from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT


from mimetypes import guess_type

from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_graph

# 定义fastapi对象
app = FastAPI(title='query service',description='掌柜智库查询服务')

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# 接口1：返回html页面
# url: /query/html 无参数 直接响应html文件
@app.get('/query/html')
def return_query_html():
    # 1. 拼接地址
    html_path_obj = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    # 2. 判断文件是否存在
    if not html_path_obj.exists():
        logger.error(f'html不存在，无法返回页面！')
        raise HTTPException(status_code=404,DETAIL=f'html不存在，无法返回页面')
    # 3.返回文件数据（ Fileresponse直接返回文件数据）
    return FileResponse(
        path=html_path_obj,
        media_type=guess_type(html_path_obj.name)[0]
    )


# 接口2 健康检查
# /health 无参数 响应健康状态
@app.get('/health')
def healt():
    return{
        'ok':True
    }


# 定义任务请求接口的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query:str = Field(...,description='查询内容')
    session_id:str = Field(None,description='会话ID')
    is_stream: bool = Field(False,description='是否流式返回')



# 接口三 任务请求接口
# /query
# method = post
# 参数json: {query:"提问内容",session_id:当前会话的id,is_stream:流式处理}
# 响应json: 流式: { "message": "结果正在处理中...", "session_id": "xxx-uuid" }
#          非流式: { "message": "处理完成！", "session_id": "xxx", "answer": "回答内容...", "done_list": [] }
"""
   task_utils 
       容器:
            done_list = {session_id: []}
            running_list = {session_id: []}
            task_status = {session_id : status}
            _tasks_result = {session_id:{key : answer}}
       方法:
             add_running_task  / get 
             add_done_task     / get 
             update_task_status / get 
             set_task_result   /  get 
   sse_utils 
       容器: 
            _session_stream: Dict[str, queue.Queue] = {}
       方法:
            push_to_session 
            create_sse_queue 
"""


"""
  /query 
  作用: 触发查询流程的执行  query_graph的执行.... (意图确认 / 多路召回 / 粗排序 / 精排序 / 结果输出...)
  形式: is_stream = true 异步执行以上过程   is_stream = false 同步执行异常过程 
  步骤:
     1. 接收三个核心参数 query session_id is_stream 
     2. 判断是否是流式处理
        是
           创建sse的队列 create_sse_queue(session_id)  #[sse -> queue ]
           异步执行结果 backgroundtask.add_task(方法,query session_id is_stream)
           结果  { "message": "结果正在处理中...", "session_id": "xxx-uuid" }
        否
          直接调用同步执行graph方法(query session_id is_stream)
          task_utils  get_task_result(session_id,"answer")   [node_answer_output / set_task_result(session_id,"answer",result)]
          结果 -> {} -> { "message": "处理完成！", "session_id": "xxx", "answer": "回答内容...", "done_list": [get_done_list()] } 
          
  执行方法封装:
     run_query_graph
     1. 处理一下异常 try except
     2. 准备 create_query_default_state
        update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream) -> sse 
     3. 执行图即可 query_app.invoke(initial_state)
        update_task_status(session_id,TASK_STATUS_COMPLETED,is_stream) -> sse 
        异常
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)
"""


def run_query_graph(session_id:str,query:str,is_stream:bool):
    """执行main_graph"""
    try:

        # 清空原有的任务消息队列
        clear_task(session_id)

        # 更新会话状态
        update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)

        # 执行检索任务
        initial_state = (
            create_query_default_state(session_id=session_id,
                                        original_query = query,
                                        is_stream = is_stream
                                       )
        )
        query_graph.invoke(initial_state)
        update_task_status(session_id,TASK_STATUS_COMPLETED,is_stream)


        # final事件最后推送，因为其会关闭本次传输流
        # 图片URL暂未从检索结果中带出，先留空（原先写死的演示地址指向旧服务器和旧桶名，已移除）
        image_urls = []
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                'answer':get_task_result(session_id,'answer'),
                'status':'completed',
                'image_urls':image_urls            }
        )

    except Exception as e:
        logger.exception(f'执行{session_id}对应查询问题：{query}，任务执行失败，错误信息：{e}')
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)


@app.post('/query')
async def query(query_request:QueryRequest,backgroundtasks:BackgroundTasks):

    is_stram = query_request.is_stream
    session_id = query_request.session_id
    query = query_request.query
    #判定是否异步执行（流式）
    if is_stram:

        # 异步流式，创建sse的消息队列(用于数据传输接口返回流式数据)
        create_sse_queue(session_id)
        # 添加异步任务（挂起）
        backgroundtasks.add_task(run_query_graph,session_id,query,is_stram)
        # 返回响应结果
        return {'message':'结果正在处理中...','session_id':session_id}
    else:
        # 同步执行（等待任务结果）
        run_query_graph(session_id,query,is_stram)
        # 获取结果
        answer = get_task_result(session_id,'answer')
        # 等待结束以后，返回结果
        return {
            'message':'处理完成！',
            'session_id':session_id,
            'answer':answer,
            'done_list':get_done_task_list(session_id)
        }

# 接口四 流式返回任务结果
@app.get('/stream/{session_id}')
async def stream_query_result(session_id:str,request:Request):
    return StreamingResponse(
        # 生成器工具函数
        sse_generator(session_id,request),
        # 返回结果类型
        media_type='text/event-stream'
    )

# 接口五 查询历史聊天记录
# /history/{session_id}?limit=xx 可能传
@app.get('/history/{session_id}')
def get_history(session_id:str,limit:int=10):
    records = get_recent_messages(session_id,limit=limit)
    items = []
    for r in records:
        items.append({
            'id':str(r.get('_id')) if r.get('_id') is not None else "",
            'session_id':r.get('session_id',''),
            'role':r.get('role',''),
            'text':r.get('text',''),
            'rewritten_query':r.get('rewritten_query',''),
            'item_names':r.get('item_names',[]),
            'ts':r.get('ts')
        })

    return {
        'session_id':session_id,
        'items':items
    }

# 接口六：清空历史聊天记录
# /history/{session_id} delete
@app.delete('/history/{session_id}')
def delete_history(session_id:str):
    # 删除
    delete_count = clear_history(session_id)
    return{
        'message':f'删除:{session_id}会话对应的聊天记录成功！',
        'delete_count':delete_count
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )