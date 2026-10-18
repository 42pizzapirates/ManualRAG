from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger


query_graph = StateGraph(QueryGraphState)

query_graph.add_node('node_item_name_confirm',node_item_name_confirm)
query_graph.add_node('node_search_embedding',node_search_embedding)
query_graph.add_node('node_search_embedding_hyde',node_search_embedding_hyde)
query_graph.add_node('node_web_search_mcp',node_web_search_mcp)
query_graph.add_node('node_rrf',node_rrf)
query_graph.add_node('node_rerank',node_rerank)
query_graph.add_node('node_answer_output',node_answer_output)


query_graph.set_entry_point('node_item_name_confirm')

def node_item_name_confirm_after_router(state:QueryGraphState):
    """已生成澄清答案时直接收尾，否则并发进入三路召回。"""
    if state['answer']:
        logger.warning(f'node_item_name_confirm无法继续向后执行：{state['answer']}')
        return 'node_answer_output'

    return 'node_search_embedding','node_search_embedding_hyde','node_web_search_mcp'

query_graph.add_conditional_edges('node_item_name_confirm',
                                    node_item_name_confirm_after_router,
                                    {
                                        'node_answer_output':'node_answer_output',
                                        'node_search_embedding':'node_search_embedding',
                                        'node_search_embedding_hyde':'node_search_embedding_hyde',
                                        'node_web_search_mcp':'node_web_search_mcp'
                                    }

                                  )

query_graph.add_edge('node_search_embedding','node_rrf')
query_graph.add_edge('node_search_embedding_hyde','node_rrf')
query_graph.add_edge('node_web_search_mcp','node_rrf')
query_graph.add_edge('node_rrf','node_rerank')
query_graph.add_edge('node_rerank','node_answer_output')
query_graph.add_edge('node_answer_output',END)


query_graph = query_graph.compile()
