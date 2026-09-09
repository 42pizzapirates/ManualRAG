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



load_dotenv(find_dotenv(),override=True)


def step_1_data_validate(state:QueryGraphState):
    """
    获取session_id和原始查询内容,并且进行校验处理!
    :param state:
    :return:
    """

    original_query = state.get('original_query')
    session_id = state.get('session_id')
    if not original_query or not session_id:
        logger.error(f'session_id和original_query不能为空')
        raise ValueError(f'session_id和original_query不能为空')

    return original_query,session_id


def step_2_chat_history(session_id):
    """历史聊天记录"""
    return get_recent_messages(session_id)

def step_3_llm_itemnames_and_rewrite(history_message_list,original_query):
    """
    进行item_name和rewritten_query识别
    :param history_message_list:
    :param original_query:
    :return:  {item_names:[],rewritten_query:重写问题}
    """
    # 1. 初始化模型对象
    client = get_llm_client(json_mode=True)
    # 2. 进行提示词处理
    # 构建历史聊天记录
    history_text = ''
    for msg in history_message_list:
        role = msg.get('role', '')
        content = msg.get('rewritten_query') if role == 'user' else msg.get('text')
        # item_names 可能为 None（save_chat_message 默认值），join 前需兜底
        item_names_text = '、'.join(msg.get('item_names') or [])
        history_text += f'角色:{role},内容:{content or ""},关联主体:{item_names_text}\n'

    prompt = load_prompt('rewritten_query_and_itemnames',history_text = history_text,query = original_query)
    # 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]

    # 3. 调用模型,获取json格式输出（client.invoke 返回 AIMessage，需经解析器转成 dict）
    llm_json_dict = JsonOutputParser().invoke(client.invoke(messages))

    if not 'rewritten_query' in llm_json_dict:
        logger.warning(f'模型重写问题失败，给rewritten_query赋予原始问题：{original_query}')
        llm_json_dict['rewritten_query'] = original_query
    if not 'item_names' in llm_json_dict:
        logger.warning(f'模型识别商品失败，给item_names赋予空列表')
        llm_json_dict['item_names'] = []
    # 4. 返回响应结果
    return llm_json_dict

def step_4_vector_query_item_name(item_names):
    """
       a -> 向量 -> 两个annSearchRequest -> hy_search -> 结果
       b -> 向量 -> 两个annSearchRequest -> hy_search -> 结果
       c -> 向量 -> 两个annSearchRequest -> hy_search -> 结果
       d -> 向量 -> 两个annSearchRequest -> hy_search -> 结果
    :param item_names:  lm : [a,b,c,d]
    :return:
       {
          a : [{item_name:xx,score:0.8}]
       }
    """
    vector_dict = {}
    # 1. 将llm查询到的item_names转化成向量
    # result = {dense:[],sparse:[]}
    item_name_dict = generate_embeddings(item_names)
    # 2. 循环每个item_name对应稠密和稀疏向量，进行混合查询
    for index in range(len(item_names)):
        # index 获取当前item_name 对应的稠密和稀疏向量
        dense_vector = item_name_dict['dense'][index]
        sparse_vector = item_name_dict['sparse'][index]

        # 将稠密和稀疏向量分别转成annSearchRequest（单向量检索）
        # 调用milvus封装的方法，创建annSearchRequest（默认limit：5）
        # 稠密向量 metric 必须与建库索引一致（kb_item_names 建索引时用的是 IP，默认的 COSINE 会报 metric type not match）
        reqs = create_hybrid_search_requests(dense_vector,sparse_vector,dense_params={"metric_type":"IP"})

        # 定义rerank工具，调用混合检索结果，进行重排序(加权或rff)
        milvus_client = get_milvus_client()
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs = reqs,
            ranker_weights=(0.8,0.2),
            norm_score=True, # 将分调整到 0 - 1 之间比较  [cosine 0 -1  ip  -1 0 1 ]
                              # IP 整体分都偏低   -1 0 1  -> 0 - 1 ->  -0.5 + 1 / 2
                              # cosine 分基本不变
            output_fields=['item_name']
        )

        # 3. 解析结果:嵌套列表（单向量索引结果为列表，混合索引结果为嵌套列表）
        # entity返回样本的指定字段信息(键值对)
        """
           response = [
              [
                 {
                    id:xx,
                    distance:0.8,
                    entity:{item_name:向量中item_name}
                 },
                  {
                    id:xx,
                    distance:0.8,
                    entity:{item_name:向量中item_name}
                 },
              ]
           ]
        """
        current_item_name_list = []
        # hybrid_search 执行失败会返回 None，此处按空结果处理
        for item in (response[0] if response else []):
            """
            {
                id:xx,
                distance:0.8,
                entity:{item_name:向量中item_name}
             }
            """ 
            current_item_name_list.append(
                {
                    'item_name':item.get('entity',{}).get('item_name',''),
                    'score':item.get('distance',0)
                }
            )          

        # 4. 组装为最终的dict
        vector_dict[item_names[index]] = current_item_name_list

    return vector_dict

def step_5_select_item_name_list(vector_dict):
    """
    作用: 在向量查询的列表中,选出确定和可选的item_name列表
    final_result = {confirmed_item_name_list:[],options_item_name_list:[]}

    { A: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
      B: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
      C: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]

    原则1: 每个模型提供item_name只能对应一个确认的向量查询item_name
    原则2: 当没有确认的item_name我们提取可以选的item_name 可选的每个提供至多两个2
    原则3: 他们最终不会区分,都会一起加入到 confirmed_item_name_list | options_item_name_list
        确认的规则:
             大于 0.65 入选 (自己调整)
             只有一个 -> 就选他
             有多个  -> 选item_name (lm)  -> 最高分
        可选的规则: (没有确定)
             大于0.55  <=    <小于0.65
             提供top2
        分更低: 就是无用的数据 不需要处理...

    :param vector_dict:
    :return:
    """
    # 1.准备两个集合 确认 和 可选的
    confirmed_item_name_list = [] 
    options_item_name_list = []

    # 2. 遍历每个item_name对应的向量数据
    for item_name,item_name_list in vector_dict.items():
        # 3. 排序(保证按相似度分数排列)
        item_name_list.sort(key=lambda x : x['score'],reverse=True)
        # 4. 截取确认的集合和可选的集合
        high_list = [item for item in item_name_list if item['score'] >= 0.65]
        low_list = [item for item in item_name_list if 0.5 <= item['score'] <0.65]
        # 5. 确认集合长度并选择处理方式

        if len(high_list) > 0:
            confirmed_item_name_list.append(high_list[0]['item_name'])
            continue
        if len(low_list) > 0:
            options_item_name_list.extend(item['item_name'] for item in low_list[:2])

    # 返回结果
    return {
        'confirmed_item_name_list':confirmed_item_name_list,
        'options_item_name_list':options_item_name_list
    }

def step_6_deal_state(state,final_result,rewritten_query):
    """
    修改state : answer  item_names rewritten_query
    :param state:
    :param final_result:
    :param rewritten_query:
    :return:
    """
    confirmed_item_name_list = final_result.get('confirmed_item_name_list',[])
    options_item_name_list = final_result.get('options_item_name_list',[])
    # 1. 无论是否确认到商品名,都记录模型重写后的问题(step_7保存聊天记录需要)
    state['rewritten_query'] = rewritten_query
    # 2. 判断confirmed列表中有无数据
    if len(confirmed_item_name_list) > 0:
        # 成功
        state['item_names'] = confirmed_item_name_list
        if 'answer' in state:
            del state['answer']
        return

    # 3.confirmed列表为空，判断可选列表
    if len(options_item_name_list) > 0:
        option_name_str = '、'.join(options_item_name_list)
        state['answer'] = f'您是想问以下哪个产品：{option_name_str}?请明确型号'
        return
    # 4，均为空，无可选主体
    state['answer'] = '抱歉，未找到相关产品，请提供准确型号以便我为您查询。'

def step_7_save_user_chat_message(state):
    # 进行用户聊天记录保存
    save_chat_message(
        session_id=state['session_id'],
        role='user',
        text=state['original_query'],
        rewritten_query=state['rewritten_query'],
        item_names=state.get('item_names',[])
    )

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    # 1.日志任务管理
    add_running_task(state['session_id'],sys._getframe().f_code.co_name,state['is_stream'])

    # 2. 取值和校验
    original_query,session_id = step_1_data_validate(state)

    # 3. 获取历史聊天记录(seesion_id) -> list[message]
    history_message_list = step_2_chat_history(session_id)

    # 4. 调用模型进行问题重写和item_names识别
    llm_result_dict = step_3_llm_itemnames_and_rewrite(history_message_list,original_query)
    item_names = llm_result_dict['item_names']
    rewritten_query = llm_result_dict['rewritten_query']

    final_result = {}
    if item_names and len(item_names) > 0:
        # 5. 通过模型提供的item_names 分别进行向量数据库混合检索
        # 模型 item_names = [A,B,C]
        # 向量库 A -> 匹配的item_name 每个name对应分 B -> 匹配的item_name  C -> 匹配的item_name
        """
          { A: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
            B: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
            C: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
        """
        vector_dict = step_4_vector_query_item_name(item_names)

        # 6.确认最终的item_name列表
        final_result = step_5_select_item_name_list(vector_dict)

    # 7.判断confirmed列表和option列表，处理最终的answer并更新state
    step_6_deal_state(state,final_result,rewritten_query)

    # 8. 保存被刺对话的记录（user）
    step_7_save_user_chat_message(state)

    add_done_task(state['session_id'],sys._getframe().f_code.co_name,state['is_stream'])
    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "烫金机和华为手机以及苹果手机好不好用?",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")


