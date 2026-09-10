import asyncio
import os
import json
import sys
# from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger, node_log, step_log

DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

# 1. 数据校验
@step_log('step_1_data_validate')
def step_1_data_validate(state):
    """
    获取参数并且校验
    :param state:
    :return:
    """
    rewritten_query = state['rewritten_query']
    if not rewritten_query or len(rewritten_query) == 0:
        logger.warning('rewritten_query为空！改为使用original_query!')
        rewritten_query = state['original_query']

    return rewritten_query


# 2. 封装MCP异步路由函数
@step_log('step_2_openai_mcp_call')
async def step_2_openai_mcp_call(rewritten_query:str,count:int=5):
    """
    使用openai的方式调用mcpserver提供的工具
    :param rewritten_query:
    :param count:
    :return:
    """
    #1.通过适配器创建mcp_client
    mcp_client = MCPServerStreamableHttp(
        name = 'web_search', # 随便写
        params={
            'url' : DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,
            'headers' : {'Authorization':DASHSCOPE_API_KEY},
            'timeout' : 200,
            'sse_read_timeout' : 200
        }
    )

    try:
        # 2. 连接mcp服务（connect/list_tools/cleanup 均为协程，必须 await）
        await mcp_client.connect()
        # 3. 调用工具
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
        # 4.释放本次连接资源
        await mcp_client.cleanup()

@node_log('node_web_search_mcp')
def node_web_search_mcp(state):
    # 1. 任务日志管理
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 2. 数据校验
    rewritten_query = step_1_data_validate(state)

    # 3. 封装mcp调用流程(异步函数），获取调用结果
    result = asyncio.run(step_2_openai_mcp_call(rewritten_query))

    # 4. 结果解析
    text_dict = json.loads(result.content[0].text)
    pages = text_dict.get('pages',[])
    # 不直接打印pages内容：搜索结果含™等GBK无法编码的字符，Windows控制台会抛UnicodeEncodeError
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

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果（只打印摘要，搜索结果原文含非GBK字符，直接打印会抛UnicodeEncodeError）
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")
    print(f"联网搜索返回条数: {len(result_state.get('web_search_docs', []))}")
