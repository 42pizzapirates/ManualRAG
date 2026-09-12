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
    """
    获取session_id和原始查询内容,并且进行校验处理!
    :param state:
    :return:
    """

    original_query = state.get('original_query')
    session_id = state.get('session_id')
    if not original_query or not session_id:
        logger.error('session_id和original_query不能为空！')
        raise ValueError('session_id和original_query不能为空！')

    return original_query,session_id

@step_log('step_2_chat_history')
def step_2_chat_history(session_id):
    """历史聊天记录"""
    return get_recent_messages(session_id)

@step_log('step_3_llm_item_names_and_rewrite')
def step_3_llm_item_names_and_rewrite(history_message_list,original_query):
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
    promt = load_prompt('rewritten_query_and_itemnames',history_text=history_text,query=original_query)

    # 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=promt)
    ]

    # 3.调用模型，使用json格式化输出
    response = client.invoke(messages)
    # json_mode已保证返回JSON，用JsonOutputParser解析，可兼容模型偶尔包裹的```json代码块
    llm_json_dict = JsonOutputParser().parse(response.content)

    # 4. 模型偶发漏字段，兜底避免 KeyError 打穿整条链路
    if 'rewritten_query' not in llm_json_dict:
        logger.warning(f'模型重写问题失败，给rewritten_query赋予原始问题：{original_query}')
        llm_json_dict['rewritten_query'] = original_query
    if 'item_names' not in llm_json_dict:
        logger.warning('模型识别商品失败，给item_names赋予空列表')
        llm_json_dict['item_names'] = []

    return llm_json_dict

@step_log('step_4_vector_query_item_name')
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
    # 1. 将llm查询到的所有item_names转化为向量
    # result = {dense:[],sparse:[]}
    item_name_dict = generate_embeddings(item_names)
    # 2. 循环每个item_name对应的稠密和稀疏向量，进行混合检索
    for index in  range(len(item_names)):
    # index 获取当前item_name对应的稠密和稀疏向量
        dense_vector = item_name_dict['dense'][index]
        sparse_vecor = item_name_dict['sparse'][index]
        # 将稠密和稀疏向量分别转成annSearchRequest
        # 调用了mivlus封装的方法,内部帮我们创建两个AnnSearchRequest 默认: limit = 5
        # todo: 相似度全部改为 IP [当初创建集合索引的类型]

        reqs = create_hybrid_search_requests(dense_vector,sparse_vecor)
        # 正常定义一个rerank工具
        # 调用混合检索函数进行搜索结果
        milvus_client = get_milvus_client()
        response = hybrid_search(
            client = milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs = reqs,
            ranker_weights = (0.8,0.2),
            norm_score = True,  # 将分调整到 0 - 1 之间比较  [cosine 0 -1  ip  -1 0 1 ]
                              # IP 整体分都偏低   -1 0 1  -> 0 - 1 ->  -0.5 + 1 / 2
                              # cosine 分基本不变
            output_fields=['item_name']
        )

    # 3. 解析结果
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
                  'score':item.get('distance',0.0)
                }
            )
        # 组装到最终的dict作为单个item_name的检索结果
        vector_dict[item_names[index]] = current_item_name_list

    # 5. 返回最终结果
    return vector_dict


@step_log('step_5_select_item_name_list')
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

    # 1. 准备两个集合 确认 和 可选的
    confirmed_item_name_list = []
    option_item_name_list = []
    # 2. 遍历每个item_name对应向量数据
    for item_name,item_name_list in vector_dict.items():
        # 3. 按分数排序
        item_name_list.sort(key=lambda x:x['score'],reverse=True)
        # 4. 截取确认的集合和可选的集合
        high_list = [item for item in item_name_list if item['score'] >= 0.65]
        low_list = [item for item in item_name_list if 0.50 <= item['score'] < 0.65]
        # 5. 确认集合长度和选择（条件边）
        # 只获取分数最高的
        if len(high_list) > 0:
            confirmed_item_name_list.append(high_list[0]['item_name'])
            continue
        # 6.可选集合（在确认集合没有选中的场景下，进行可选集合处理）
        if len(low_list) > 0:
            option_item_name_list.extend([item['item_name'] for item in low_list[:2]])

    # 7.返回结果
    return{
        'confirmed_item_name_list':confirmed_item_name_list,
        'option_item_name_list':option_item_name_list
    }

@step_log('step_6_deal_state')
def step_6_deal_state(state:QueryGraphState,final_result,rewritten_query):
    """
    修改state : answer  item_names rewritten_query
    :param state:
    :param final_result:
    :param rewritten_query:
    :return:
    """
    confirmed_item_name_list = final_result.get('confirmed_item_name_list',[])
    option_item_name_list = final_result.get('option_item_name_list',[])

    # 1. 无论是否确认到商品名,都记录模型重写后的问题(step_7保存聊天记录需要)
    state['rewritten_query'] = rewritten_query

    # 2. 判断确定列表有无数据
    if len(confirmed_item_name_list) > 0 :
        # 成功
        state['item_names'] = confirmed_item_name_list
        if 'answer' in state:
            del state['answer']
        return

    # 3. 没有确定列表，有可选列表
    if len(option_item_name_list) > 0:
        # 没有确认的有可选的
        # 名字、xxxx
        option_name_str = '、'.join(option_item_name_list)
        state['answer'] = f'您是想问以下哪个产品：{option_name_str}?请明确型号。'
        return
    # 4. 确定与可选均为空，返回提示
    state['answer'] = f'抱歉，未找到相关产品，请提供准确型号以便我为您查询！'

@step_log('step_7_save_user_chat_message')
def step_7_save_user_chat_message(state:QueryGraphState):
    # 进行用户聊天记录保存
    save_chat_message(
        session_id = state['session_id'],
        role='user',
        text = state['original_query'],
        rewritten_query=state['rewritten_query'],
        item_names=state.get('item_names', [])
    )

@node_log('node_item_name_confirm')
def node_item_name_confirm(state:QueryGraphState):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """

    # 1. 日志和任务处理 
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # 2. 取值和校验
    original_query,session_id = step_1_data_validates(state)
    # 3. 获取历史聊天记录
    hiistory_message_list = step_2_chat_history(session_id)
    # 回填state：本节点是入口节点，下游 node_answer_output 组装答案提示词时要用
    state['history'] = hiistory_message_list
    # 4. 调用模型进行问题重写和item_name识别
    llm_result_dict = step_3_llm_item_names_and_rewrite(hiistory_message_list,original_query)
    item_names = llm_result_dict['item_names']
    rewritten_query = llm_result_dict['rewritten_query']
    # {确认item_name:[] , 可选的item_name:[]}
    final_result = {}
    if item_names and len(item_names) > 0:
        # 5. 通过模型提供的item_names分别进行向量数据库混合检索
        # 模型 item_names = [A,B,C]
        # 向量库 A -> 匹配的item_name 每个name对应分 B -> 匹配的item_name  C -> 匹配的item_name
        """
          { A: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
            B: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
            C: [{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8},{item_name:匹配到的item_name,score:0.8}]
        """
        vector_dict = step_4_vector_query_item_name(item_names)
        # 6. 确认item_name列表
        # final_result = {confirmed_item_name_list:[],options_item_name_list:[]}
        final_result = step_5_select_item_name_list(vector_dict)

    # 7. 判断确定和可选的列表,最终处理answer以及给state
    step_6_deal_state(state,final_result,rewritten_query)

    # 8. 保存本次对话的记录（user）
    step_7_save_user_chat_message(state)

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
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