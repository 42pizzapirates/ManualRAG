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

@node_log("step_1_data_validates")
def step_1_item_names_validate(state:QueryGraphState):
    """校验商品名与重写查询，缺失时回退原始问题。"""
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


@node_log("step_2_llm_predict_answer")
def step_2_llm_predict_answer(rewritten_query):
    """让模型生成假设性答案，用于扩展检索语义。"""

    llm_client = get_llm_client()

    prompt = load_prompt('hyde_prompt',rewritten_query=rewritten_query)
    message = [
        HumanMessage(content=prompt)
    ]

    return llm_client.invoke(message).content


@ node_log('step_3_concat_vector_search')
def step_3_concat_vector_search(item_names,rewritten_query,predict_answer):
    """把原问题与假设答案拼接后检索，只保留已确认商品对应的切片。"""
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
        norm_score=True,  # 把 IP 分数归一化到 0-1
        limit=5,
        output_fields=['chunk_id','item_name','content',"title","parent_title","part","file_title"]
    )

    return result_list[0] if len(result_list) > 0 else []


@node_log('node_search_embedding_hyde')
def node_search_embedding_hyde(state):
    """HyDE 召回：问题 + 假设答案检索，结果写入 state。"""
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, rewritten_query = step_1_item_names_validate(state)
    predict_answer = step_2_llm_predict_answer(rewritten_query)
    result_list = step_3_concat_vector_search(item_names,rewritten_query,predict_answer)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": result_list}

if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "联想至像小象Z2黑白激光打印机 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
        "item_names": ["联想至像小象Z2黑白激光打印机"],
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding_hyde 节点...")
    try:
        result = node_search_embedding_hyde(test_state)
        logger.info(f"检索结果汇总：{result}")
        chunks = result.get("hyde_embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        print(f"\n>>> 测试完成！检索到 {chunks} 条结果")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
