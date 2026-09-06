# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库


# 初始化环境变量，必须在配置读取前执行，确保后续节点能获取到环境变量的配置信息
load_dotenv()

# 1.定义状态图对象，并且指定全局state类型
workflow = StateGraph(ImportGraphState)

# 2.添加节点
workflow.add_node('node_entry',node_entry)
workflow.add_node('node_pdf_to_md',node_pdf_to_md)
workflow.add_node('node_md_img',node_md_img)
workflow.add_node('node_document_split',node_document_split)
workflow.add_node('node_item_name_recognition',node_item_name_recognition)
workflow.add_node('node_bge_embedding',node_bge_embedding)
workflow.add_node('node_import_milvus',node_import_milvus)

# 3.指定入口节点
# workflow.add_edge(START,'node_entry')
workflow.set_entry_point('node_entry')



# 4.设置入口节点后的条件边
# 在node_entry后，用于判断文件类型，并连接至对应节点
# node_entry -> state -> is_md_read_enabled = True -> node_md_img
# is_pdf_enabled = True -> node.odf_to_md
# 均为false -> END

def md_or_pdf(state:ImportGraphState):
    if state['is_md_read_enabled']:
        return 'node_md_img'
    elif state['is_pdf_read_enabled']:
        return 'node_pdf_to_md'
    else:
        return END


'''
添加条件边
    参数1：源节点
    参数2：路由函数
    参数3：path_map
        不可省略：路由函数返回的字符串不等于节点名称，
                需要打印图结构时，必须用path_map将返回值映射到节点名称

'''
workflow.add_conditional_edges('node_entry',md_or_pdf,path_map={'node_md_img':'node_md_img',
                                                                'node_pdf_to_md':'node_pdf_to_md',
                                                                END:END})
    

# 5.设置普通边
workflow.add_edge('node_pdf_to_md','node_md_img')
workflow.add_edge('node_md_img','node_document_split')
workflow.add_edge('node_document_split','node_item_name_recognition')
workflow.add_edge('node_item_name_recognition','node_bge_embedding')
workflow.add_edge('node_bge_embedding','node_import_milvus')
workflow.add_edge('node_import_milvus',END)


# 6.编译状态图对象
import_graph = workflow.compile()




if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os

    # 全流程测试：验证PDF导入→Milvus入库→KG导入完整链路
    logger.info("===== 开始执行知识图谱导入全流程测试 =====")
    # 1. 构造测试文件路径（复用你项目的doc目录，和pdf2md测试文件一致）
    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    # 2. 构造输出目录（存放MD/图片等中间文件）
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # 不存在则创建

    # 3. 校验测试PDF文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")
    else:
        # 4. 构造测试状态（贴合实际业务入参，开启PDF解析开关）
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # 测试任务ID
            "local_file_path": test_pdf_path,  # 测试PDF文件路径
            "local_dir": test_output_dir,  # 中间文件输出目录
            "is_pdf_read_enabled": False,  # 开启PDF解析（核心开关）
            "is_md_read_enabled": False  # 关闭MD解析
        })
        try:
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus→kg")

            # 5. 执行LangGraph全流程（流式执行，打印节点执行进度）
            final_state = None
            for step in import_graph.stream(test_state, stream_mode="values"):
                # 打印当前执行完成的节点（流式输出更直观）
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step  # 保存最终状态

            # 6. 全流程执行完成，结果预览和核心指标打印
            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")
                # 提取核心结果指标
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]  # MD内容前150字符
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "未生成")  # KG导入生成的ID（按实际业务字段调整）

                # 打印核心指标
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





