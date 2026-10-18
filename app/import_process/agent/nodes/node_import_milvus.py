from pymilvus import DataType
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger,node_log,step_log
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

@step_log('step_1_validate_chunks')
def step_1_validate_chunks(state):
    """校验 chunks 非空。"""
    chunks = state['chunks']
    if not chunks or len(chunks) == 0:
        logger.error(f'chunks为空,无法继续业务!')
        raise ValueError(f'chunks为空,无法继续业务!')

@step_log('step_2_import_milvus')
def step_2_import_milvus(state:ImportGraphState):
    """确保集合存在，按 item_name 清理旧数据后写入本次切片。"""
    milvus_client = get_milvus_client()

    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 关闭动态字段：Milvus 3.0 会为隐藏的 $meta 字段生成 shredding_data，
        # v3.0.0 的路径重复拼接缺陷会导致重启后段加载失败；本集合字段已全部显式声明
        schema = milvus_client.create_schema(
            auto_id = True,
            enable_dynamic_field = False
        )

        schema.add_field(field_name='chunk_id',datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='title',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='parent_title',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='part',datatype=DataType.INT16)
        schema.add_field(field_name='content',datatype=DataType.VARCHAR,max_length = 65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            index_name="dense_vector_index",
            metric_type="IP"  # 向量已做 L2 归一化，内积等价于余弦相似度
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}  # 跳过零值维度
        )

        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )


    # 按 item_name 幂等清理，保证重复导入不产生脏数据
    item_name = state['item_name']
    milvus_client.delete(collection_name=milvus_config.chunks_collection,
                         filter=f'item_name=="{escape_milvus_string(item_name)}"')


    result = milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=state['chunks']
    )

    insert_count  = result.get('insert_count',0)
    logger.info(f'插入数据成功！，总条数：{insert_count}')
    ids = result.get('ids',[])
    logger.info(f'插入主键回显：{ids}')


@ node_log('node_import_milvus')
def node_import_milvus(state:ImportGraphState) -> ImportGraphState:
    """把带向量的 chunks 写入 Milvus。"""

    add_running_task(state['task_id'],'node_import_milvus')

    step_1_validate_chunks(state)

    step_2_import_milvus(state)

    add_done_task(state['task_id'],'node_import_milvus')

    return state


if __name__ == '__main__':
    import sys
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            result_state = node_import_milvus(test_state)

            chunks = result_state.get("chunks", [])
            logger.info(f"返回结果:{chunks}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
