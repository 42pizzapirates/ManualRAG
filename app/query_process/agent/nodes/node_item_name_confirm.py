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
from app.core.logger import logger, node_log, step_log

load_dotenv(find_dotenv())

@step_log('step_1_data_validates')
def step_1_data_validates(state:QueryGraphState):
    """校验并返回原始查询与 session_id。"""

    original_query = state.get('original_query')
    session_id = state.get('session_id')
    if not original_query or not session_id:
        logger.error('session_id和original_query不能为空！')
        raise ValueError('session_id和original_query不能为空！')

    return original_query,session_id

@step_log('step_2_chat_history')
def step_2_chat_history(session_id):
    """读取会话历史，作为查询重写与商品识别的上下文。"""
    return get_recent_messages(session_id)

@step_log('step_3_llm_item_names_and_rewrite')
def step_3_llm_item_names_and_rewrite(history_message_list,original_query):
    """结合历史对话重写查询，并抽取用户提到的商品名称。

    :return: {"item_names": [...], "rewritten_query": "..."}
    """

    client = get_llm_client(json_mode=True)

    history_text = ''
    for msg in history_message_list:
        role = msg.get('role', '')
        content = msg.get('rewritten_query') if role == 'user' else msg.get('text')
        # item_names 可能为 None（save_chat_message 默认值），join 前需兜底
        item_names_text = '、'.join(msg.get('item_names') or [])
        history_text += f'角色:{role},内容:{content or ""},关联主体:{item_names_text}\n'
    promt = load_prompt('rewritten_query_and_itemnames',history_text=history_text,query=original_query)

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=promt)
    ]

    response = client.invoke(messages)
    # JsonOutputParser 可兼容模型把 JSON 包在 ```json 代码块里的情况
    llm_json_dict = JsonOutputParser().parse(response.content)

    # 模型偶发漏字段，补齐默认值避免 KeyError 中断流程
    if 'rewritten_query' not in llm_json_dict:
        logger.warning(f'模型重写问题失败，给rewritten_query赋予原始问题：{original_query}')
        llm_json_dict['rewritten_query'] = original_query
    if 'item_names' not in llm_json_dict:
        logger.warning('模型识别商品失败，给item_names赋予空列表')
        llm_json_dict['item_names'] = []

    return llm_json_dict

@step_log('step_4_vector_query_item_name')
def step_4_vector_query_item_name(item_names):
    """对每个商品名执行稠密 + 稀疏混合检索，返回 {原词: 候选列表}。"""

    vector_dict = {}
    item_name_dict = generate_embeddings(item_names)
    for index in  range(len(item_names)):
        dense_vector = item_name_dict['dense'][index]
        sparse_vecor = item_name_dict['sparse'][index]

        reqs = create_hybrid_search_requests(dense_vector,sparse_vecor)
        milvus_client = get_milvus_client()
        response = hybrid_search(
            client = milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs = reqs,
            ranker_weights = (0.8,0.2),
            norm_score = True,  # 把 IP 分数归一化到 0-1，便于与阈值比较
            output_fields=['item_name']
        )

        current_item_name_list = []
        # hybrid_search 失败时返回 None，按空结果继续处理
        for item in (response[0] if response else []):
            current_item_name_list.append(
                {
                  'item_name':item.get('entity',{}).get('item_name',''),
                  'score':item.get('distance',0.0)
                }
            )
        vector_dict[item_names[index]] = current_item_name_list

    return vector_dict


@step_log('step_5_select_item_name_list')
def step_5_select_item_name_list(vector_dict):
    """按分数阈值把候选商品分为已确认与待选两组。

    >= 0.65 取最高分作为确认项；0.50-0.65 取前两名作为待选项。
    """

    confirmed_item_name_list = []
    option_item_name_list = []
    for item_name,item_name_list in vector_dict.items():
        item_name_list.sort(key=lambda x:x['score'],reverse=True)
        high_list = [item for item in item_name_list if item['score'] >= 0.65]
        low_list = [item for item in item_name_list if 0.50 <= item['score'] < 0.65]
        if len(high_list) > 0:
            confirmed_item_name_list.append(high_list[0]['item_name'])
            continue
        if len(low_list) > 0:
            option_item_name_list.extend([item['item_name'] for item in low_list[:2]])

    return{
        'confirmed_item_name_list':confirmed_item_name_list,
        'option_item_name_list':option_item_name_list
    }

@step_log('step_6_deal_state')
def step_6_deal_state(state:QueryGraphState,final_result,rewritten_query):
    """把检索结果写回 state；未唯一命中时直接生成澄清话术。"""
    confirmed_item_name_list = final_result.get('confirmed_item_name_list',[])
    option_item_name_list = final_result.get('option_item_name_list',[])

    # 即使未确认商品，也要保留重写结果供历史记录使用
    state['rewritten_query'] = rewritten_query

    if len(confirmed_item_name_list) > 0 :
        state['item_names'] = confirmed_item_name_list
        if 'answer' in state:
            del state['answer']
        return

    if len(option_item_name_list) > 0:
        option_name_str = '、'.join(option_item_name_list)
        state['answer'] = f'您是想问以下哪个产品：{option_name_str}?请明确型号。'
        return
    state['answer'] = f'抱歉，未找到相关产品，请提供准确型号以便我为您查询！'

@step_log('step_7_save_user_chat_message')
def step_7_save_user_chat_message(state:QueryGraphState):
    """保存用户本轮消息。"""
    save_chat_message(
        session_id = state['session_id'],
        role='user',
        text = state['original_query'],
        rewritten_query=state['rewritten_query'],
        item_names=state.get('item_names', [])
    )

@node_log('node_item_name_confirm')
def node_item_name_confirm(state:QueryGraphState):
    """识别并确认用户问题中的商品名称，写入 state['item_names']。"""

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    original_query,session_id = step_1_data_validates(state)
    hiistory_message_list = step_2_chat_history(session_id)
    # 下游答案生成节点需要复用它拼装上下文
    state['history'] = hiistory_message_list
    llm_result_dict = step_3_llm_item_names_and_rewrite(hiistory_message_list,original_query)
    item_names = llm_result_dict['item_names']
    rewritten_query = llm_result_dict['rewritten_query']
    final_result = {}
    if item_names and len(item_names) > 0:
        vector_dict = step_4_vector_query_item_name(item_names)
        final_result = step_5_select_item_name_list(vector_dict)

    step_6_deal_state(state,final_result,rewritten_query)

    step_7_save_user_chat_message(state)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "烫金机和华为手机以及苹果手机好不好用?",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
