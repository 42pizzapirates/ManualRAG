import sys
import time

from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, step_log, node_log
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

@step_log('step_1_validate_answer')
def step_1_validate_answer(state:QueryGraphState):
    """若前置节点已生成澄清答案，则直接输出并返回 True。"""
    answer = state.get('answer')
    is_stream = state.get('is_stream',False)
    session_id = state.get('session_id')

    if answer:
        if is_stream:
            for ch in answer:
                push_to_session(session_id,SSEEvent.DELTA,{'delta':ch})
        set_task_result(session_id,'answer',answer)
        return True
    return False

@step_log('step_2_data_validates')
def step_2_data_validates(state:QueryGraphState):
    """取出答案生成所需的上下文并校验关键字段。"""
    history = state.get('history',[])
    reranked_docs = state.get('reranked_docs',[])
    item_names = state.get('item_names',[])
    rewritten_query = state.get('rewritten_query',[])

    if not reranked_docs or len(reranked_docs) ==0 or not rewritten_query:
        logger.error('reranked_docs 或 rewritten_query 为空!无法继续使用模型进行答案生成!')
        raise ValueError('reranked_docs 或 rewritten_query 为空!无法继续使用模型进行答案生成!')

    return history,reranked_docs,item_names,rewritten_query

@step_log('step_3_make_prompt')
def step_3_make_promt(history,reranked_docs,item_names,rewritten_query):
    """把检索结果、历史对话和商品信息填入答案提示词模板。"""

    context_chunk_list = []

    for number,chunk in enumerate(reranked_docs,start=1):
        context_chunk_list.append(
            f'第{number}块：标题：{chunk['title']} 匹配度得分：{chunk['score']} 来源：{'网络搜索' if chunk['type'] == 'web' else '向量查询'}'
            f'\n'
            f'内容：{chunk['text']}'
        )

    history_text = '没有历史聊天记录！'
    if history and len(history) > 0:
        history_text = ""
        for msg in history:
            history_text += \
                (f"角色:{msg['role']},内容:{msg['rewritten_query'] if msg['role'] == 'user' else msg['text']}"
                f",关联主体: {'、'.join(msg.get('item_names') or [])}\n")

    item_name_str = '本次关联主体：'+'、'.join(item_names) if item_names and len(item_names) > 0 else '没有关联主体！'

    prompt = load_prompt('answer_out',context = context_chunk_list,
                         history = history_text ,item_names = item_name_str,question = rewritten_query)
    return prompt

@step_log('step_4_call_llm_final_answer')
def step_4_call_llm_final_answer(state:QueryGraphState,prompt):
    """生成最终答案；流式模式下逐段推送 delta 事件。"""

    is_stream = state.get('is_stream',False)
    session_id = state.get('session_id')

    llm_client = get_llm_client()
    final_result = ''
    if is_stream:
        for chunk in llm_client.stream(prompt):
            delta_content = chunk.content
            final_result += delta_content
            push_to_session(session_id,SSEEvent.DELTA,{'delta':delta_content})
    else:
        response = llm_client.invoke(prompt)
        content = response.content
        final_result = content

    set_task_result(session_id,'answer',final_result)
    state['answer'] = final_result


@step_log('step_5_extract_chunk_images')
def step_5_extract_chunk_images(state:QueryGraphState,reranked_docs):
    """从检索结果的 url 字段和正文 Markdown 中收集去重后的图片地址。"""

    image_urls = []
    reg = re.compile(r'\!\[.*?\]\((.*?)\)')

    for doc in reranked_docs:
        url = doc.get('url')
        text = doc.get('text')
        if url:
            if url.endswith((".png", ".jpg",".gif",".jpeg",".svg")):
                if url not in image_urls:
                    image_urls.append(url)
        if text:
            image_str_list = reg.findall(text)
            for image_url in image_str_list:
                if image_url not in image_urls:
                    image_urls.append(image_url)

    state['image_urls'] = image_urls

@step_log('step_6_save_chat_histroy')
def step_6_save_chat_histroy(state:QueryGraphState):
    """保存助手回复及关联商品、图片，供后续轮次使用。"""
    save_chat_message(
        session_id = state['session_id'],
        role = 'assistant',
        text = state['answer'],
        rewritten_query = state.get('rewritten_query') or state.get('original_query'),
        item_names = state.get('item_names',[]),
        image_urls = state.get('image_urls',[])

    )

@node_log('node_answer_outpu')
def node_answer_output(state:QueryGraphState):
    """生成最终答案，支持整段或流式输出。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    has_answer = step_1_validate_answer(state)
    if not has_answer:
        history,reranked_docs,item_names,rewritten_query = step_2_data_validates(state)
        prompt = step_3_make_promt(history,reranked_docs,item_names,rewritten_query)
        step_4_call_llm_final_answer(state,prompt)
        step_5_extract_chunk_images(state,reranked_docs)

    step_6_save_chat_histroy(state)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "type": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "type": "milvus",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "user", "text": "HAK 180 烫金机","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"}
    ]

    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None
    }

    try:
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
        else:
            print("[FAIL] Prompt 未构建")

        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
