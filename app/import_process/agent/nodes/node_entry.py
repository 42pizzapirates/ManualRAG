import os.path
import sys
from pathlib import Path

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """识别输入文件类型，设置对应解析开关；不支持的类型直接结束流程。"""

    add_running_task(state["task_id"],"node_entry")

    local_file_path = state["local_file_path"]
    if not local_file_path:
        logger.warning(f"节点:node_entry,获取文件输入地址,发现地址为空!直接跳转到END节点")
        add_done_task(state["task_id"], "node_entry")
        return state
    if local_file_path.endswith(".md"):
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
    else:
        logger.warning(f"虽然local_file_path有值{local_file_path},不是md或者pdf类型,所以无法识别,直接跳转到END节点!")
        add_done_task(state["task_id"], "node_entry")
        return state
    local_file_path_obj = Path(local_file_path)
    file_name = local_file_path_obj.stem
    state["file_title"] = file_name

    add_done_task(state["task_id"],"node_entry")
    return state

if __name__ == '__main__':

    logger.info("===== 开始node_entry节点单元测试 =====")

    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    logger.info(f"测试1结果: {test_state1}")

    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    logger.info(f"测试2结果: {test_state2}")

    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")
