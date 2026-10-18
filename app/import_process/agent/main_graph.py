from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus


load_dotenv()

workflow = StateGraph(ImportGraphState)

workflow.add_node('node_entry',node_entry)
workflow.add_node('node_pdf_to_md',node_pdf_to_md)
workflow.add_node('node_md_img',node_md_img)
workflow.add_node('node_document_split',node_document_split)
workflow.add_node('node_item_name_recognition',node_item_name_recognition)
workflow.add_node('node_bge_embedding',node_bge_embedding)
workflow.add_node('node_import_milvus',node_import_milvus)

workflow.set_entry_point('node_entry')


def md_or_pdf(state:ImportGraphState):
    """按文件类型选择解析入口。"""
    if state['is_md_read_enabled']:
        return 'node_md_img'
    elif state['is_pdf_read_enabled']:
        return 'node_pdf_to_md'
    else:
        return END

# path_map 让路由返回值在图结构展示中仍对应到真实节点名
workflow.add_conditional_edges('node_entry',md_or_pdf,path_map={'node_md_img':'node_md_img',
                                                                'node_pdf_to_md':'node_pdf_to_md',
                                                                END:END})
    

workflow.add_edge('node_pdf_to_md','node_md_img')
workflow.add_edge('node_md_img','node_document_split')
workflow.add_edge('node_document_split','node_item_name_recognition')
workflow.add_edge('node_item_name_recognition','node_bge_embedding')
workflow.add_edge('node_bge_embedding','node_import_milvus')
workflow.add_edge('node_import_milvus',END)


import_graph = workflow.compile()




if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os

    logger.info("===== 开始执行知识图谱导入全流程测试 =====")
    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)

    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")
    else:
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",
            "local_file_path": test_pdf_path,
            "local_dir": test_output_dir,
            "is_pdf_read_enabled": False,
            "is_md_read_enabled": False
        })
        try:
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus→kg")

            final_state = None
            for step in import_graph.stream(test_state, stream_mode="values"):
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step

            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "未生成")

                logger.info(f"📄 PDF转MD内容预览（前150字符）：{md_content}...")
                logger.info(f"📝 文档切分总切片数：{chunk_count}")
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"🗄️  所有切片是否完成Milvus入库（含chunk_id）：{'是' if has_chunk_id else '否'}")
                logger.info(f"🧠 知识图谱导入ID：{kg_id}")
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            logger.exception(f"===== 全流程测试运行失败 =====")
    logger.info("===== 知识图谱导入全流程测试结束 =====")





