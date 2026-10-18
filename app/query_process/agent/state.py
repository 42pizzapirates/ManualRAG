from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """查询流程在图节点之间流转的状态。"""
    session_id: str
    original_query: str

    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list

    rrf_chunks: list
    reranked_docs: list

    prompt: str
    answer: str

    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
    image_urls: List[str]


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": []
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """复制默认状态并应用字段覆盖。"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """返回默认状态的深拷贝。"""
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """深拷贝现有状态并应用字段覆盖。"""
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)
