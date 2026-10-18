import sys
from typing import List, Dict, Any
import dashscope
from dotenv import load_dotenv
from langchain.messages import HumanMessage

from app.conf.embedding_config import embedding_config
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log
from app.query_process.agent.state import QueryGraphState

load_dotenv(override=True)

# 动态 TopK 的上下限与断崖截断阈值
RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.5
RERANK_GAP_ABS: float = 0.3


@step_log('step_1_data_validates')
def step_1_data_validates(state: QueryGraphState):
    """取出粗排结果与联网搜索结果。"""
    web_search_docs = state.get("web_search_docs", [])
    rrf_chunks = state.get("rrf_chunks", [])


    return web_search_docs, rrf_chunks


@step_log('step_2_merge_data')
def step_2_merge_data(state,web_search_docs, rrf_chunks) -> list[dict]:
    """把向量检索与联网结果归一为统一的文档结构。"""

    rewritten_query = state.get("rewritten_query", "")

    rerank_list = []
    for chunk in web_search_docs:
        rerank_list.append({
            'title':chunk.get('title'),
            'text':chunk.get('snippet'),
            'url':chunk.get('url'),
            'type':'web',
            'score':0.0
        })
    for chunk in rrf_chunks:
        rerank_list.append({
            'title':chunk.get('title'),
            'text':chunk.get('content'),
            'url':None,
            'type':'milvus',
            'score':0.0
        })

    logger.info(f'完成两路数据统一格式处理，rrf路原数据条数{len(rrf_chunks)},web路原数据条数{len(web_search_docs)}')
    return rerank_list

def count_qwen_tokens_sdk(text: str) -> int:
    """返回 Qwen 分词器下的 token 数。"""
    return len(dashscope.tokenizers.get_tokenizer("qwen-flash").encode(text))

@step_log('step_3_rerank')
def step_3_rerank(state,rerank_list,max_retry=3) -> list[float]:
    """对每条候选文档打分，超长时先摘要，返回按分数降序排列的结果。"""

    rewritten_query = state.get('rewritten_query') or state.get('original_query')

    invoke_list = [[rewritten_query,chunk.get('text')] for chunk in rerank_list]

    # 单条超过 512 token 会被 reranker 拒绝，先压缩再打分
    for item in invoke_list:
        token_count = count_qwen_tokens_sdk(str(item[0]) + str(item[1]))
        retry = 0
        while token_count > 512 and retry < max_retry:
            llm_client = get_llm_client()
            prompt = load_prompt('rerank_summary',text = item)
            # content 传 [字符串] 会被序列化为内容块数组并被服务端拒绝，必须是纯字符串
            message = [
                HumanMessage(content=prompt)
            ]
            item[1] = llm_client.invoke(message).content

            token_count = count_qwen_tokens_sdk(str(item[0]) + str(item[1]))
            retry += 1
        if token_count > 512:
            logger.warning(f"摘要{max_retry}次后token数仍为{token_count}，按当前内容直接打分")

    rerank_model = get_reranker_model()
    scores_list = rerank_model.compute_score(invoke_list,normalize=True)

    for chunk,score in zip(rerank_list,scores_list):
        chunk['score'] = round(score,4)

    rerank_list.sort(key=lambda x: x['score'], reverse=True)

    return rerank_list
       
@step_log('step_4_topk_chunk')
def step_4_topk_chunk(chunk_list_score_sorted):
    """按分数断崖动态截断，返回最终进入答案生成的文档。"""

    min_topk = RERANK_MIN_TOPK
    max_topk = RERANK_MAX_TOPK
    max_gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS

    topk = min(max_topk,len(chunk_list_score_sorted))

    if topk > min_topk:
        # 上界必须用 topk，候选不足 max_topk 时直接按 max_topk 遍历会越界
        for i in range(min_topk-1,topk-1):
            score_1 = chunk_list_score_sorted[i].get('score',0.0)
            score_2 = chunk_list_score_sorted[i+1].get('score',0.0)
            abs_score = score_1-score_2
            gap_ratio = (score_1-score_2)/(score_1+1e-7)
            if abs_score > max_gap or gap_ratio > max_gap_ratio:
                topk = i+1
                break
    return chunk_list_score_sorted[:topk]


@node_log('node_rerank')
def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """合并多路召回结果，重排后写入 state['reranked_docs']。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    web_search_docs, rrf_chunks = step_1_data_validates(state)

    rerank_list = step_2_merge_data(state,web_search_docs, rrf_chunks)

    chunk_list_score_sorted = step_3_rerank(state,rerank_list,max_retry=4)

    topk_chunk_list = step_4_topk_chunk(chunk_list_score_sorted)
    state['reranked_docs'] = topk_chunk_list

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)



    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档"}
    ]

    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('type')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")


