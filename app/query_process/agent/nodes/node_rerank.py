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

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）：分数相对前一条降低超过该比例即截断
RERANK_GAP_RATIO: float = 0.5
# 断崖阈值（绝对）：与上一条分差超过该值即截断（分数已sigmoid归一化到0~1）
RERANK_GAP_ABS: float = 0.3



# 1. 数据校验
@step_log('step_1_data_validates')
def step_1_data_validates(state: QueryGraphState):
    """
    获取rrf_node粗排序的结果以及mcp网络搜索结果
    :param state:
    :return:
    """
    web_search_docs = state.get("web_search_docs", [])
    rrf_chunks = state.get("rrf_chunks", [])


    return web_search_docs, rrf_chunks


# 2.合并两路数据（[[问题, 答案, 原始文档],...]）
@step_log('step_2_merge_data')
def step_2_merge_data(state,web_search_docs, rrf_chunks) -> list[dict]:
    """
    进行两路数据融合! 统一格式方便后续数据处理!
    {title: , text: content or snippet , url : mcp专属 , type: milvus or web , score : 0.0}
    :param rrf_chunks:
    :param web_search_docs:
    :return: [{title: , text: content or snippet , url : mcp专属 , type: milvus or web , score : 0.0}]
    """

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

# 定义计算token函数（序列长度）
def count_qwen_tokens_sdk(text: str) -> int:
    return len(dashscope.tokenizers.get_tokenizer("qwen-flash").encode(text))

# 3. 使用reranker模型进行问题和答案的重排序（打分）
@step_log('step_3_rerank')
def step_3_rerank(state,rerank_list,max_retry=3) -> list[float]:
    """
    对[问题, 答案]逐条打分，并回填分数、按分数降序排序
    :param max_retry: 单条超长时最多摘要几次，避免摘要后仍超长导致死循环
    :return: 带score字段且已按score降序排序的rerank_list
    """

    # 1.获取重写的问题
    rewritten_query = state.get('rewritten_query') or state.get('original_query')

    # 2. 拼接reranker模型输入:[[问题，答案],...]
    invoke_list = [[rewritten_query,chunk.get('text')] for chunk in rerank_list]

    # 3. 校验每条的token数是否超出模型的最大长度限制（超长则调用llm摘要压缩）
    for item in invoke_list:
        token_count = count_qwen_tokens_sdk(str(item[0]) + str(item[1]))
        retry = 0
        while token_count > 512 and retry < max_retry:
            # 超出限制，调用llm进行总结摘要
            llm_client = get_llm_client()
            prompt = load_prompt('rerank_summary',text = item)
            # content 必须是纯字符串或内容块列表（[{"type":"text","text":...}]），
            # 传 [字符串] 会被序列化成 content:["..."]，服务端按内容块枚举解析直接 400
            message = [
                HumanMessage(content=prompt)
            ]
            item[1] = llm_client.invoke(message).content

            token_count = count_qwen_tokens_sdk(str(item[0]) + str(item[1]))
            retry += 1
        if token_count > 512:
            logger.warning(f"摘要{max_retry}次后token数仍为{token_count}，按当前内容直接打分")

    # 4. 调用rerank模型获取分数列表
    rerank_model = get_reranker_model()
    scores_list = rerank_model.compute_score(invoke_list,normalize=True) # 设置归一化用于取topk分数

    # 将score封装到对应原rerank_list的chunk字段里
    for chunk,score in zip(rerank_list,scores_list):
        chunk['score'] = round(score,4) # 四舍五入，去小数点后四位

    # 按分数排序
    rerank_list.sort(key=lambda x: x['score'], reverse=True)

    return rerank_list
       
@step_log('step_4_topk_chunk')
def step_4_topk_chunk(chunk_list_score_sorted):
    """
       min_topk  - topk截取数据  -  max_topk
    :param chunk_list_score_sorted:
    :return:
    """
    # 1. 接收参数
    min_topk = RERANK_MIN_TOPK #最少chunk数
    max_topk = RERANK_MAX_TOPK #最大chunk数
    max_gap_ratio = RERANK_GAP_RATIO #相对前一个分数降低百分比
    max_gap = RERANK_GAP_ABS #分差绝对值

    # max_topk不能超过待选list长度，否则无意义
    topk = min(max_topk,len(chunk_list_score_sorted))

    if topk > min_topk:
        # 上界用topk而不是max_topk：topk已收敛到列表长度，否则列表不足max_topk条时会越界
        for i in range(min_topk-1,topk-1):
            score_1 = chunk_list_score_sorted[i].get('score',0.0)
            score_2 = chunk_list_score_sorted[i+1].get('score',0.0)
            abs_score = score_1-score_2
            gap_ratio = (score_1-score_2)/(score_1+1e-7)
            if abs_score > max_gap or gap_ratio > max_gap_ratio:
                topk = i+1
                break
    # else:
        # 若 max_topk < min_topk ,即 候选列表长度 < min_topk，直接返回整个列表
    return chunk_list_score_sorted[:topk]


@node_log('node_rerank')
def node_rerank(state: QueryGraphState) -> QueryGraphState:
    # 1. 任务日志管理
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 2. 数据校验
    web_search_docs, rrf_chunks = step_1_data_validates(state)

    # 3. 合并两路数据
    rerank_list = step_2_merge_data(state,web_search_docs, rrf_chunks)

    # 4. 使用reranker模型进行问题和答案的重排序（打分）,并组装按分数排序的chunk_list,
    chunk_list_score_sorted = step_3_rerank(state,rerank_list,max_retry=4)

    # 5. 取topk分数的chunk作为最终的返回结果
    topk_chunk_list = step_4_topk_chunk(chunk_list_score_sorted)
    state['reranked_docs'] = topk_chunk_list

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)



    # for index in range(1,5):
    #     print(index)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    # todo: 注意
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档"}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
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

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")


