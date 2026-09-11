import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from mpmath import limit
from pymilvus import AnnSearchRequest

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger, node_log

load_dotenv(override=True)

# 1. 数据校验
@node_log("step_1_data_validates")
def step_1_item_names_validate(state:QueryGraphState):
    item_names = state['item_names']
    rewritten_query = state['rewritten_query'
                        ]
    if not item_names or len(item_names) == 0:
        logger.error(f'item_names为空，无法进行向量检索')
        raise ValueError(f'item_names为空，无法进行向量检索')
    if not rewritten_query:
        logger.warning(f'rewritten_query为空！给予origin_query')
        rewritten_query = state['original_query']
    
    return item_names,rewritten_query


# 2. 调用模型获取预测答案
@node_log("step_2_llm_predict_answer")
def step_2_llm_predict_answer(rewritten_query):

    # 2.1 获取大模型
    llm_client = get_llm_client()

    # 2.2 拼接提示词
    prompt = load_prompt('hyde_prompt',rewritten_query=rewritten_query)
    message = [
        HumanMessage(content=prompt)
    ]

    # 调用模型
    return llm_client.invoke(message).content


# 3. 将问题与预测答案拼接，进行向量化检索
@ node_log('step_3_concat_vector_search')
def step_3_concat_vector_search(item_names,rewritten_query,predict_answer):
    # print(f'拼接后的输入：{rewritten_query+predict_answer}/n')
    vector_dict = generate_embeddings([rewritten_query+' '+predict_answer])

    dense_vector = vector_dict['dense'][0]
    sparse_vector = vector_dict['sparse'][0]

    expr = f'item_name in {item_names}'
    reqs = create_hybrid_search_requests(dense_vector,sparse_vector,expr=expr)

    milvus_client = get_milvus_client()
    result_list = hybrid_search( 
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs = reqs,
        ranker_weights=(0.8,0.2),
        norm_score=True, # 将分调整到 0 - 1 之间比较  [cosine 0 -1  ip  -1 0 1 ]
                         # IP 整体分都偏低   -1 0 1  -> 0 - 1 ->  -0.5 + 1 / 2
                         # cosine 分基本不变
        limit=5,
        output_fields=['chunk_id','item_name','content',"title","parent_title","part","file_title"]
    )

    return result_list[0] if len(result_list) > 0 else []


@node_log('node_search_embedding_hyde')
def node_search_embedding_hyde(state):
    """
    节点功能：进行向量内容检索
    """
    print("---量内容检索 开始处理---")
    #1. 日志和任务处理
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 2. 参数获取和校验(item_names / rewritten_query)
    item_names, rewritten_query = step_1_item_names_validate(state)
    # 3. 调用llm获取预测回答
    predict_answer = step_2_llm_predict_answer(rewritten_query)
    # 4. 重写提问+预测回答拼接，再进行向量化检索
    result_list = step_3_concat_vector_search(item_names,rewritten_query,predict_answer)
    # 5. 返回结果即可
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": result_list}

if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "联想至像小象Z2黑白激光打印机 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",  # 模拟改写后的查询
        "item_names": ["联想至像小象Z2黑白激光打印机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding_hyde 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding_hyde(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("hyde_embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        print(f"\n>>> 测试完成！检索到 {chunks} 条结果")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)