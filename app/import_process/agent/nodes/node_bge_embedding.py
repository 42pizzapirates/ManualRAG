import os
from typing import Any, List, Dict

from dotenv import load_dotenv

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger,node_log,step_log

@step_log("step_1_validate_chunks")
def step_1_validate_chunks(state:ImportGraphState) -> list:
    """返回非空的 chunks。"""
    chunks = state['chunks']

    if not chunks:
        logger.error(f'发现chunks为空，无法继续业务处理！')
        raise ValueError(f'发现chunks为空，无法继续业务处理！')
    return chunks

@step_log('step_2_embedding_chunks')
def step_2_embedding_chunks(chunks):
    """分批生成稠密与稀疏向量，向量化文本优先拼接商品名。"""

    chunks_vector = []
    total = len(chunks)
    step = 5
    for index in range(0,total,step):
        try:
            step_chunks = chunks[index:index+step]
            vector_str_list = []
            for item in step_chunks:
                # 商品名前置可提升与查询主体的语义匹配
                item_name = item['item_name']
                content = item['content']
                item_str = f'主体：{item_name},内容:{content}' if item_name else content
                vector_str_list.append(item_str)

            result = generate_embeddings(vector_str_list)

            for i,chunk in enumerate(step_chunks):
                chunk_new = chunk.copy()
                chunk_new['dense_vector'] = result['dense'][i]
                chunk_new['sparse_vector'] = result['sparse'][i]
                chunks_vector.append(chunk_new)

        except Exception as e:
            logger.warning(f'index={index}步骤发生错误，跳过')
            continue
    return chunks_vector

@node_log('node_bge_embedding')
def node_bge_embedding(state:ImportGraphState) -> ImportGraphState:
    """为所有 chunk 生成向量并写回 state。"""
    add_running_task(state['task_id'],'node_bge_embedding')

    chunks = step_1_validate_chunks(state)

    chunks = step_2_embedding_chunks(chunks)

    state['chunks'] = chunks

    add_done_task(state['task_id'],'node_bge_embedding')

    return state

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",
        "chunks": [
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        result_state = node_bge_embedding(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果:{result_chunks}")


    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")


