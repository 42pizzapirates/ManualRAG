import sys
import os

from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger, node_log, step_log
from dotenv import load_dotenv,find_dotenv
from app.utils.task_utils import add_done_task,add_running_task

# load_dotenv 加载 .env 文件  find_dotenv() .env文件在项目下的任何位置都可以加载!
load_dotenv(find_dotenv())

# 1. 数据校验
@node_log("step_1_data_validates")
def step_1_item_names_validate(state):
    """
    获取参数并且校验
    :param state:
    :return:
    """
    item_names = state.get('item_names')
    rewritten_query = state.get('rewritten_query')
    if not item_names or not rewritten_query:
        logger.error('item_names或rewritten_query不存在')
        raise ValueError('item_names或rewritten_query不存在')
    return item_names,rewritten_query

# 2. 将重写问题进行向量化,并执行混合检索
@node_log("step_2_rewritten_query_vector")
def step_2_rewritten_query_vector(item_names,rewritten_query):

    vector_dict = generate_embeddings([rewritten_query])

    dense_vector = vector_dict['dense'][0]
    sparse_vector = vector_dict['sparse'][0]

    # 1. 获取milvus客户端
    milvus_client = get_milvus_client()

    # 2.设置检索过滤字段
    expr_str = f'item_name in {item_names}'
    # 3. 封装检索请求对象列表
    reqs = create_hybrid_search_requests(dense_vector,sparse_vector,expr=expr_str)

    result = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.8,0.2),
        norm_score=True,
        limit=7,
        output_fields=["chunk_id","item_name","content","title","parent_title","part","file_title"]
    )
    return result[0] if len(result) > 0 else []

def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    print("---量内容检索 开始处理---")
    #1. 日志和任务处理
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 2. 参数获取和校验(item_names / rewritten_query)
    item_names, rewritten_query = step_1_item_names_validate(state)
    # 4. 向量化并进行混合检索(过滤条件/双向量和权重设置/输出字段控制)
    result_list = step_2_rewritten_query_vector(item_names,rewritten_query)
    # 5. 返回结果即可
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": result_list}

if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "联想至像小象Z2黑白激光打印机 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",  # 模拟改写后的查询
        "item_names": ["联想至像小象Z2黑白激光打印机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        print(f"\n>>> 测试完成！检索到 {chunks} 条结果")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)