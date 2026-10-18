from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from app.conf.milvus_config import milvus_config
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# 单次识别使用的切片数与长度上限，避免超出模型上下文
DEFAULT_ITEM_NAME_CHUNK_K = 5
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
CONTEXT_TOTAL_MAX_CHARS = 10000


def step_1_check_content(state:ImportGraphState):
    """校验并返回 chunks 与 file_title。"""
    chunks = state['chunks']
    file_title = state['file_title']

    if not chunks:
        logger.error(f'chunks为空！终止业务')
        raise ValueError(f'chunks为空！终止业务')
    if not file_title:
        logger.warning(f'file_title为空！设为默认值')
        file_title = 'default_title'
    return chunks,file_title

def step_2_document_context(chunks) -> str:
    """截取前 K 个切片并拼接为模型可读的上下文字符串。"""
    current_chunks = chunks[:DEFAULT_ITEM_NAME_CHUNK_K]
    chunks_str_list = []
    for idx,item in enumerate(current_chunks,start=1):
        content = item['content'][:SINGLE_CHUNK_CONTENT_MAX_LEN]
        chunks_str_list.append(f'切片:{idx},标题:{item['title']},内容:{content}')

    chunk_str = '\n'.join(chunks_str_list)

    final_chunk_str = chunk_str[:CONTEXT_TOTAL_MAX_CHARS]

    return final_chunk_str

def step_3_lm_call(context,file_title) -> str:
    """调用模型识别商品名称，返回原始文本。"""
    llm = get_llm_client() 

    system_prompt_str = load_prompt('product_recognition_system')
    user_prompt_str = load_prompt("item_name_recognition",file_title = file_title,context = context)

    message = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=user_prompt_str)
    ]

    item_name = llm.invoke(message).content

    return item_name

def step_4_insert_milvs(item_name,file_title,dense_vector,sparse_vector):
    """将商品名称及其向量写入 Milvus，同名记录会被覆盖。"""

    milvus_client = get_milvus_client()

    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 关闭动态字段：Milvus 3.0 会为隐藏的 $meta 字段生成 shredding_data，
        # v3.0.0 的路径重复拼接缺陷会导致重启后段加载失败；本集合字段已全部显式声明
        schema = milvus_client.create_schema(
            auto_id = True,
            enable_dynamic_field = False
        )

        schema.add_field(field_name='pk',datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length = 512)

        schema.add_field(field_name='dense_vector',datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name='sparse_vector',datatype=DataType.SPARSE_FLOAT_VECTOR)


        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name='dense_vector',
            index_type='AUTOINDEX',
            index_name='dense_vector_index',
            metric_type='IP'  # 向量已做 L2 归一化，内积等价于余弦相似度
        )

        index_params.add_index(
            field_name='sparse_vector',
            index_type='SPARSE_INVERTED_INDEX',
            index_name='sparse_vector_index',
            metric_type='IP',
            params={'inverted_index_algo':"DAAT_MAXSCORE"}  # 跳过零值维度
        )

        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )

    # 同名商品只保留最新记录；转义后拼入 filter，避免引号破坏表达式
    milvus_client.delete(collection_name=milvus_config.item_name_collection,filter=f'item_name == "{escape_milvus_string(item_name)}"')

    data = [
        {
            'file_title':file_title,
            'item_name':item_name,
            'dense_vector':dense_vector,
            'sparse_vector':sparse_vector
        }
    ]
    milvus_client.insert(collection_name=milvus_config.item_name_collection,data=data)

@node_log('node_item_name_recognition')
def node_item_name_recognition(state:ImportGraphState) -> ImportGraphState:
    add_running_task(state['task_id'],'node_item_name_recognition')      

    chunks,file_title = step_1_check_content(state)

    context = step_2_document_context(chunks)

    item_name = step_3_lm_call(context,file_title)
    if not item_name or not item_name.strip():
        logger.warning(f'item_name识别为空，回退使用file_title：{file_title}')
        item_name = file_title

    state['item_name'] = item_name
    for chunk in state['chunks']:
        chunk['item_name'] = item_name

    result = generate_embeddings([item_name])
    dense_vector = result['dense'][0]
    sparse_vector = result['sparse'][0]

    step_4_insert_milvs(item_name,file_title,dense_vector,sparse_vector)

    add_done_task(state['task_id'],'node_item_name_recognition')    

    return state


def test_node_item_name_recognition():
    """独立运行该节点，验证识别、向量化和 Milvus 写入链路。"""
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",
            "file_title": "华为Mate60 Pro手机使用说明书",
            "file_name": "华为Mate60Pro说明书.pdf",
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        result_state = node_item_name_recognition(mock_state)

        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            item_name = result_state.get('item_name')
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{escape_milvus_string(item_name)}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


if __name__ == "__main__":
    test_node_item_name_recognition()
