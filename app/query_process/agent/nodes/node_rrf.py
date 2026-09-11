import sys
from typing import List, Dict, Any

from app.conf.embedding_config import embedding_config
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log
from app.query_process.agent.state import QueryGraphState

@node_log("step_1_data_validates")
def step_1_data_validates(state:QueryGraphState):
    """
    获取参数并且校验
    :param state:
    :return:
    """

    embedding_chunks = state.get('embedding_chunks',[])
    hyde_embedding_chunks = state.get('hyde_embedding_chunks',[])

    return embedding_chunks,hyde_embedding_chunks


@node_log('step_2_rrf_list')
def step_2_rrf_list(embedding_chunks,hyde_embedding_chunks,k:int=60,top:int=5,weights=(1.0,1.0)):
    """
    进行多路融合排序 , 同源,启动算法!
    本次算法 = 1.0 / k + rank * weight
    :param param_list: [([1 {id:xx,distance:xx,entity:{chunk_id}},2,3],1.0),([1,2,3],1.0),([],1.0)]
    :param k 平滑参数,用于削弱排名的过大影响
    :param top 最终的获取数量
    :return: [entity,entity....]
    """
    # 1. 处理集合参数（添加权重）
    param_list = [(embedding_chunks,weights[0]),(hyde_embedding_chunks,weights[1])]

    # 2. 定义字典（对应chunk_id的累计得分和entity字段数据）
    score_dict = {} # chunk_id:score
    entity_dict ={} # chunk_id:entity

    # 3. 循环统计每一路得分
    for chunks,weight in param_list:
        # 3.1 循环单路得分
        for rank,chunk in enumerate(chunks,start=1):
            # rank chunks中数据已排序，直接取index
            # chunk的数据结构：{id:..,distance:...,entity:...}
            
            # 3.2 统计每个chunk_id的得分和entity
            chunk_id = chunk.get('id') or chunk['entity']['chunk_id']

            score_dict[chunk_id] = score_dict.get(chunk_id,0.0) + (1.0/(rank+k))*weight
            # 存储chunk_id -> entity
            # 每次都赋值,下一次覆盖上一次!
            # entity_dict[chunk_id] = chunk.get("entity",{})
            # 如果没有值,才赋值! 第一次已经赋值了,后面就不会更新了
            entity_dict.setdefault(chunk_id,chunk.get('entity',{}))

    # 4. 组合数据（按累计得分降序排序）
    rerank_list = sorted(score_dict.items(),key=lambda x:x[1],reverse=True)
    final_entity_list = [entity_dict[chunk_id] for chunk_id,_ in rerank_list][:top]
    return final_entity_list


@node_log('node_rrf')
def node_rrf(state:QueryGraphState):
    # 1. 任务日志管理
    add_running_task(state['session_id'],'node_rrf')

    # 2. 数据校验
    embedding_chunks,hyde_embedding_chunks = step_1_data_validates(state)

    # 3. rrf重排序（排名+权重）
    final_entity_list = step_2_rrf_list(embedding_chunks,hyde_embedding_chunks)
    state['rrf_chunks'] = final_entity_list

    add_done_task(state['session_id'],'node_rrf')

    return state
    



# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "小象Z2黑白激光打印机怎么操作？",
        "rewritten_query": "小象Z2黑白激光打印机的具体操作步骤是什么？",
        "item_names": ["联想至像小象Z2黑白激光打印机"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")