import os.path
import sys
from pathlib import Path

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

"""
  节点作用: 接收传入的文件地址(local_file_path)识别文件类型,修改对应的state
  入参:  local_file_path / task_id
  出参:  is_md_read_enabled is_pdf_read_enabled  md_path  pdf_path  file_title 
  步骤:
       0. 日志动作  @node_log + 任务列表记录 (进行中,已完成)
       1. 获取state中数据 local_file_path task_id
       2. 进行文件校验 local_file_path 是否为空
       3. 根据地址判断文件类型,修改对应的state参数即可
       4. 识别文件地址对应的文件名称
       5. 返回结果和状态 
"""
@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """

    # 0. 设置成进行中任务
    add_running_task(state["task_id"],"node_entry")

    # 1. 获取state中数据 local_file_path task_id
    local_file_path = state["local_file_path"]
    # 2. 校验是否为空(不需要考虑是否真的有文件,读取文件内容的时候再考虑)
    if not local_file_path:
        # 为空 [1.抛出异常 2.服务降级(鲁棒性)]
        # 降级处理  warning
        logger.warning(f"节点:node_entry,获取文件输入地址,发现地址为空!直接跳转到END节点")
        add_done_task(state["task_id"], "node_entry")
        return state
    # 3. 根据地址判断文件类型,修改对应的state参数即可
    # md   md_path  is_md_read_enabled
    # pdf  pdf_path is_pdf_read_enabled
    # 两种都不是 服务降级
    if local_file_path.endswith(".md"):
        # 修改地址
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
    else:
        # 服务降级处理
        logger.warning(f"虽然local_file_path有值{local_file_path},不是md或者pdf类型,所以无法识别,直接跳转到END节点!")
        add_done_task(state["task_id"], "node_entry")
        return state
    # 4. 是md/pdf并且已经修改对应的值,识别文件的名字
    # os.path老版本的地址处理
    #file_name= os.path.basename(local_file_path).split(".")[0] #-> xxx.pdf
    # pathlib.Path进行处理
    local_file_path_obj = Path(local_file_path)
    #file_name_1 = local_file_path_obj.name  # xxx.pdf
    file_name = local_file_path_obj.stem  # xxx
    #file_name_1 = local_file_path_obj.suffix # .pdf
    state["file_title"] = file_name

    # 执行完毕了
    add_done_task(state["task_id"],"node_entry")
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