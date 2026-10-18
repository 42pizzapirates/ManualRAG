from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """导入流程在节点之间流转的状态。"""
    task_id: str

    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str

    md_content: str
    chunks: list
    item_name: str

    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """复制默认状态并应用字段覆盖。"""

    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state

def get_default_state() -> ImportGraphState:
    """返回默认状态的深拷贝，避免修改全局值。"""
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    state1 = get_default_state()
    state2 = create_default_state(task_id = "0001")
