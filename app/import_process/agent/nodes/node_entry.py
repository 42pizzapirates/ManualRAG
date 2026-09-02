import os.path
import sys
from pathlib import Path

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

'''
节点作用：传入文件路径，判断文件类型，更新状态
入参: local_file_path,task_id
出参：is_md_read_enabled is_pdf_read_enabled md_path pdf_path file_title
步骤：
    0. 日志动作 @node_log + 任务列表记录（进行中，已完成）
    1. 获取state中的文件路径，task_id
    2. 进行文件校验，判断文件是否为空
    3. 根据地址判断文件类型，修改对应的state字段
    4. 识别文件地址，获取文件名
    5. 返回结果和状态

'''
@node_log('node_entry')
def node_entry(state:ImportGraphState) -> ImportGraphState:
    file_path = state['local_file_path']

    # 0. 设置为进行中的任务
    add_running_task(state['task_id'],'node_entry')
    
    # 1.获取state中的数据
    file_path = state['local_file_path']

    # 2.判断file_path是否为空，若为空则服务降级（更新日志）
    if not file_path:
        logger.warning('节点:node_entry,获取文件路径，路径为空！直接跳转END节点')
        add_done_task(state['task_id'],'node_entry')
        return state

    # 3.根据file_path判断文件类型，修改对应state
    if file_path.endswith('md'):
        state['md_path'] = file_path
        state['is_md_read_enabled'] = True
    elif file_path.endswith('pdf'):
        state['pdf_path'] = file_path
        state['is_pdf_read_enabled'] = True
    else:
        logger.warning(f'节点:node_entry,获取文件路径{file_path}，不支持该文件类型！直接跳转END节点')
        add_done_task(state['task_id'],'node_entry')
        return state
    # 4.根据file_path获取文件名称
    file_path_obj = Path(file_path)
    state['file_title'] = file_path_obj.stem
    # 执行完毕\
    return state

if __name__ == '__main__':
      # 单元测试：覆盖不支持类型、MD、PDF三种场景
        logger.info("===== 开始node_entry节点单元测试 =====")
    
        # 测试1: 不支持的TXT文件
        test_state1 = create_default_state(
            task_id="test_task_001",
            local_file_path="联想海豚用户手册.txt"
        )
        node_entry(test_state1)
    
        logger.info(f"测试1结果: {test_state1}")
    
        # 测试2: MD文件
        test_state2 = create_default_state(
            task_id="test_task_002",
            local_file_path="小米用户手册.md"
        )
        node_entry(test_state2)
    
        logger.info(f"测试2结果: {test_state2}")
    
        # 测试3: PDF文件
        test_state3 = create_default_state(
            task_id="test_task_003",
            local_file_path="万用表的使用.pdf"
        )
        node_entry(test_state3)
    
        logger.info("===== 结束node_entry节点单元测试 =====")