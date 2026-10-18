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

app = FastAPI(title='query service',description='掌柜智库查询服务')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get('/query/html')
def return_query_html():
    html_path_obj = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    if not html_path_obj.exists():
        logger.error(f'html不存在，无法返回页面！')
        raise HTTPException(status_code=404,DETAIL=f'html不存在，无法返回页面')
    return FileResponse(
        path=html_path_obj,
        media_type=guess_type(html_path_obj.name)[0]
    )


@app.get('/health')
def healt():
    return{
        'ok':True
    }


class QueryRequest(BaseModel):
    """查询请求参数。"""
    query:str = Field(...,description='查询内容')
    session_id:str = Field(None,description='会话ID')
    is_stream: bool = Field(False,description='是否流式返回')



def run_query_graph(session_id:str,query:str,is_stream:bool):
    """执行查询图，并负责任务状态与终态事件。"""
    try:
        clear_task(session_id)

        update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)

        initial_state = (
            create_query_default_state(session_id=session_id,
                                        original_query = query,
                                        is_stream = is_stream
                                       )
        )
        state = query_graph.invoke(initial_state)
        update_task_status(session_id,TASK_STATUS_COMPLETED,is_stream)


        # final 事件会关闭 SSE 连接，必须最后推送
        image_urls = state['image_urls']
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
        # 必须先写入错误再改状态：update_task_status 会立即推送 progress 事件
        set_task_result(session_id,'error',str(e))
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)


@app.post('/query')
async def query(query_request:QueryRequest,backgroundtasks:BackgroundTasks):

    is_stram = query_request.is_stream
    session_id = query_request.session_id
    query = query_request.query
    if is_stram:

        create_sse_queue(session_id)
        backgroundtasks.add_task(run_query_graph,session_id,query,is_stram)
        return {'message':'结果正在处理中...','session_id':session_id}
    else:
        run_query_graph(session_id,query,is_stram)
        answer = get_task_result(session_id,'answer')
        # status/error 用于区分正常结束与执行失败
        return {
            'message':'处理完成！',
            'session_id':session_id,
            'answer':answer,
            'done_list':get_done_task_list(session_id),
            'status':get_task_status(session_id),
            'error':get_task_result(session_id,'error')
        }

@app.get('/stream/{session_id}')
async def stream_query_result(session_id:str,request:Request):
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type='text/event-stream'
    )

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
            # 前端刷新后依赖该字段渲染历史图片
            'image_urls':r.get('image_urls') or [],
            'ts':r.get('ts')
        })

    return {
        'session_id':session_id,
        'items':items
    }

@app.delete('/history/{session_id}')
def delete_history(session_id:str):
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
