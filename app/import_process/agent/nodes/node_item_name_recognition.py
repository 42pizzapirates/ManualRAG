from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import generate_embeddings
# 5. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 6. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger, node_log
# 7. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认10000
CONTEXT_TOTAL_MAX_CHARS = 10000


def step_1_check_content(state:ImportGraphState):
    """
    获取chunks切片和对应file_title
    :param state:
    :return:
    """
    chunks = state['chunks']
    file_title = state['file_title']

    # 非空校验
    if not chunks:
        logger.error(f'chunks为空！终止业务')
        raise ValueError(f'chunks为空！终止业务')
    if not file_title:
        logger.warning(f'file_title为空！设为默认值')
        file_title = 'default_title'
    return chunks,file_title

def step_2_document_context(chunks) -> str:
    """
    根据切片内容,拼接context,提供给模型进行识别item_name
       chunks = [{title,content,file_title,part,parent_title},{},{}]   -> top k 切片 -> 文本 (context) -> user 提示词 -> lm
       chunks[:DEFAULT_ITEM_NAME_CHUNK_K]  -> 文本
          切片:1,标题:xxx,内容:xxx \n
          切片:2,标题:xxx,内容:xxx \n
          切片:3,标题:xxx,内容:xxx \n
          切片:4,标题:xxx,内容:xxx \n
          切片:5,标题:xxx,内容:xxx \n
      别超过我们最大的长度限额: CONTEXT_TOTAL_MAX_CHARS
    :param chunks:
    :return:
    """
    # 1. 截取topk个切片
    current_chunks = chunks[:DEFAULT_ITEM_NAME_CHUNK_K]
    chunks_str_list = []
    # 2. 循环处理
    for idx,item in enumerate(current_chunks,start=1):
        # 3. 进行每个chunk -> 字符串（单切片内容截断，防止长切片占满上下文）
        content = item['content'][:SINGLE_CHUNK_CONTENT_MAX_LEN]
        chunks_str_list.append(f'切片:{idx},标题:{item['title']},内容:{content}')

    # 4. 组合成完整字符串
    chunk_str = '\n'.join(chunks_str_list)

    # 5. 检查最大长度
    final_chunk_str = chunk_str[:CONTEXT_TOTAL_MAX_CHARS]

    # 6. 返回
    return final_chunk_str

def step_3_lm_call(context,file_title) -> str:
    """
     调用 lm , 总结和获取 item_name,如果没有返回! 使用file_title
    :param context:
    :param file_title:
    :return:
    """
    # 1. 获取模型对象
    llm = get_llm_client() 

    # 2.拼接提示词
    system_prompt_str = load_prompt('product_recognition_system')
    user_prompt_str = load_prompt("item_name_recognition",file_title = file_title,context = context)

    message = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=user_prompt_str)
    ]

    # 3.模型调用
    item_name = llm.invoke(message).content

    return item_name

def step_4_insert_milvs(item_name,file_title,dense_vector,sparse_vector):
    """
    将四个参数插入到对应milvus中即可! 一个实体  item_name, file_title, dense_vector, sparse_vector
    """

    # 1. 连接milvus客户端
    milvus_client = get_milvus_client()

    # 2. 创建表对应shchema[有哪些列和列的类型描述]

    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 2. 创建表对应shchema[有哪些列和列的类型描述]
        # 关闭动态字段：动态字段会产生隐藏的 JSON 字段($meta)，Milvus 3.0 的 JSON Shredding 会为其
        # 生成 shredding_data 文件（v3.0.0 存在路径重复拼接的缺陷，重启后段加载失败）。本集合的
        # 所有字段均已显式声明，无需动态字段。
        schema = milvus_client.create_schema(
            auto_id = True, # id 数字 自增长
            enable_dynamic_field = False # 不接受未声明的字段
        )

        schema.add_field(field_name='pk',datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length = 512)
        schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length = 512)

        # 稠密向量有固定的类型，需要根据嵌入式模型是否开启精度压缩确定， 默认FP32=FLOAT_VECTOR
        # dim = 向量的维度 bge-m3 = 1024
        schema.add_field(field_name='dense_vector',datatype=DataType.FLOAT_VECTOR,dim=1024)
        # datatype=DataType.SPARSE_FLOAT_VECTOR 稀疏向量专用的数据类型
        schema.add_field(field_name='sparse_vector',datatype=DataType.SPARSE_FLOAT_VECTOR)


        # 3. 创建collection

        # 3.1 创建对应的索引
        index_params = milvus_client.prepare_index_params()

        # 3.2 添加索引
        index_params.add_index(
            field_name='dense_vector',
            index_type='AUTOINDEX', # 根据数据量自动切换索引类型（只支持稠密向量）
            index_name='dense_vector_index',
            metric_type='IP' # 稠密向量 COSINE IP L2 如果没有归一化 -> COSINE 归一化
        )

        index_params.add_index(
            field_name='sparse_vector',
            index_type='SPARSE_INVERTED_INDEX', # 稀疏向量仅一种索引方式
            index_name='sparse_vector_index',
            metric_type='IP', #稀疏向量 IP 考虑词的权重和模型
            params={'inverted_index_algo':"DAAT_MAXSCORE"} # 跳过0 比较有值的位置
        )

        # 3.3 创建集合collection
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )

    # 4. 写入item_name的向量数据

    # 4.1 先删除 对应item_name的原数据（转义防filter解析失败）
    milvus_client.delete(collection_name=milvus_config.item_name_collection,filter=f'item_name == "{escape_milvus_string(item_name)}"')

    # 4.2 再存储
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
    # 1. 日志任务添加
    add_running_task(state['task_id'],'node_item_name_recognition')      

    # 2. 检查chunks和file_title
    chunks,file_title = step_1_check_content(state)

    # 3. 获取chunks拼接的上下文
    context = step_2_document_context(chunks)

    # 4. 获取chunks对应的item_name（识别失败返回空时回退为file_title）
    item_name = step_3_lm_call(context,file_title)
    if not item_name or not item_name.strip():
        logger.warning(f'item_name识别为空，回退使用file_title：{file_title}')
        item_name = file_title

    state['item_name'] = item_name
    for chunk in state['chunks']:
        chunk['item_name'] = item_name

    # 5. 获取item_name的稀疏和稠密向量
    result = generate_embeddings([item_name])
    dense_vector = result['dense'][0]
    sparse_vector = result['sparse'][0]

    # 6. 整合数据，插入milvus
    step_4_insert_milvs(item_name,file_title,dense_vector,sparse_vector)

    add_done_task(state['task_id'],'node_item_name_recognition')    

    return state


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
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

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{escape_milvus_string(item_name)}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()