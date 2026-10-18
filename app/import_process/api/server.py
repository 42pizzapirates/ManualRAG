from mimetypes import guess_type
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.import_process.agent.main_graph import import_graph

app = FastAPI(title='import service',description='导入文件处理！')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get('/import/html')
def return_import_html():
    """返回导入页面。"""
    html_file_obj = PROJECT_ROOT / 'app' / 'import_process' / 'page' / 'import.html'
    if not html_file_obj.exists():
        logger.error(f'本次查询没有找到对应的html文件，返回404异常！')
        raise HTTPException(status_code=404,detail='没有找到对应的html文件')
    return FileResponse(
        path=html_file_obj,
        media_type=guess_type(html_file_obj.name)[0]
    )

def invoke_import_graph(task_id:str,local_dir:str,local_file_path:str):
    """异步执行导入流程，并维护任务状态。"""
    try:
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        state = get_default_state()
        state['task_id'] = task_id
        state['local_dir'] = local_dir
        state['local_file_path'] = local_file_path
        import_graph.invoke(state)
        update_task_status(task_id,TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id,TASK_STATUS_FAILED)
        logger.exception(f'task_id={task_id},文件解析失败：{str(e)}')


@app.post('/upload')
async def upload(backgroundtasks:BackgroundTasks,files:List[UploadFile]=File(...)):
    """保存上传文件并为每个文件启动一个异步导入任务。"""
    task_ids = []
    local_dir_obj = PROJECT_ROOT / 'output' / datetime.now().strftime("%Y%m%d")
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        local_dir_obj = local_dir_obj / task_id
        local_dir_obj.mkdir(parents=True,exist_ok=True)

        local_file_path_obj = local_dir_obj / file.filename

        local_file_path_obj.write_bytes(await file.read())

        backgroundtasks.add_task(invoke_import_graph,
                                task_id = task_id,
                                local_dir=str(local_dir_obj),
                                local_file_path=str(local_file_path_obj)  
                                 )

    return {
        'code':200,
        'message':'文件上传成功，正在解析处理中',
        'task_ids':task_ids
    }

@app.get('/status/{task_id}',summary='任务状态查询',description='根据TaskID查询单个文件的处理结果')
async def get_task_progress(task_id:str):
    """返回任务状态、已完成节点和运行中节点，供前端轮询。"""

    task_status_info:Dict[str,Any] = {
        'code':200,
        'task_id':task_id,
        'status':get_task_status(task_id),
        'done_list':get_done_task_list(task_id),
        'running_list':get_running_task_list(task_id)

    }

    logger.info(
        f'[{task_id}]任务状态查询，当前状态：{task_status_info['status']},已完成节点：{task_status_info['done_list']}'
    )

    return task_status_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
