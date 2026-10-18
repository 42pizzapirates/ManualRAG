import asyncio
import os
import json
import sys
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger, node_log, step_log

DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

@step_log('step_1_data_validate')
def step_1_data_validate(state):
    """返回可用的查询文本，缺失重写结果时回退原始问题。"""
    rewritten_query = state['rewritten_query']
    if not rewritten_query or len(rewritten_query) == 0:
        logger.warning('rewritten_query为空！改为使用original_query!')
        rewritten_query = state['original_query']

    return rewritten_query


@step_log('step_2_openai_mcp_call')
async def step_2_openai_mcp_call(rewritten_query:str,count:int=5):
    """通过 MCP 适配器调用百炼联网搜索工具。"""
    mcp_client = MCPServerStreamableHttp(
        name = 'web_search',
        params={
            'url' : DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,
            'headers' : {'Authorization':DASHSCOPE_API_KEY},
            'timeout' : 200,
            'sse_read_timeout' : 200
        }
    )

    try:
        # connect / list_tools / cleanup 均为协程，必须 await
        await mcp_client.connect()
        tool_llist = await mcp_client.list_tools()
        logger.info(f'可用工具列表：{tool_llist}')
        mcp_result = await mcp_client.call_tool(
            tool_name= 'bailian_web_search',
            arguments={
                'query' :rewritten_query,
                'count' :count
            }

        )
        return mcp_result
    finally:
        await mcp_client.cleanup()

@node_log('node_web_search_mcp')
def node_web_search_mcp(state):
    """执行联网搜索并把结果写入 state['web_search_docs']。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = step_1_data_validate(state)

    result = asyncio.run(step_2_openai_mcp_call(rewritten_query))

    text_dict = json.loads(result.content[0].text)
    pages = text_dict.get('pages',[])
    # 搜索结果可能含 GBK 无法编码的字符，Windows 控制台直接打印会抛 UnicodeEncodeError
    logger.info(f'联网搜索返回结果条数：{len(pages)}')

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {'web_search_docs':pages}

from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id":"xxxx",
        "is_stream":False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置"
    }

    result_state = node_web_search_mcp(test_state)

    # 只打印摘要，避免搜索结果中的非 GBK 字符导致控制台报错
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")
    print(f"联网搜索返回条数: {result_state.get('web_search_docs', [])}")
