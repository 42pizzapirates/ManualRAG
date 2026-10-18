import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 超过该长度的块进入二次切分；CHUNK_SIZE/OVERLAP 控制二次切分粒度
CHUNK_MAX_SIZE = 500
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

@step_log('step_1_validate_clean')
def step_1_validate_clean(state:ImportGraphState) -> Tuple[str,str]:
    """读取并清洗 Markdown 内容，补齐缺失的标题。"""
    md_content = state['md_content']
    file_title = state['file_title']
    md_path = state['md_path']

    if not md_content:
        logger.warning(f'没有从state读取到md_content内容，从md_path加载！')
        if md_path:
            md_content = Path(md_path).read_text(encoding='utf-8')
            state['md_content'] = md_content
        if not md_content:
            logger.error(f'无法从md_path获取md_content,终止任务！')
            raise ValueError(f'无法从md_path获取md_content,终止任务！')
    if not file_title:
        logger.warning(f'没有获取到file_title,给予默认值')
        if md_path:
            file_title = Path(md_path).stem
        if not file_title:
            file_title = 'default'
        state['file_title']= file_title

    # 统一换行符，避免切分时出现不可见的 \r
    md_content = md_content.replace('\r\n','\n').replace('\r','\n')

    return md_content,file_title

@step_log('step_2_data_chunk_title')
def step_2_data_chunk_title(md_content, file_title) -> List[Dict[str,str]]:
    """按 Markdown 标题切分正文，忽略代码块内的伪标题。"""

    reg = re.compile(r'^\s*#{1,6}\s.+')

    lines = md_content.split('\n')

    chunks = []
    current_title = None
    current_title_lines = []
    is_code_block = False
    chunk_size = 0

    for line in lines:
        if line.startswith('```') or line.startswith('~~~'):
            is_code_block = not is_code_block
            current_title_lines.append(line)
            continue

        if reg.match(line) and not is_code_block:

            if current_title:
                chunks.append({
                    'content':'\n'.join(current_title_lines),
                    'title':current_title,
                    'file_title':file_title
                })
            current_title = line 
            current_title_lines = [current_title]
            chunk_size += 1
        else:
            current_title_lines.append(line)

    if current_title:
        chunks.append({
                    'content':'\n'.join(current_title_lines),
                    'title':current_title,
                    'file_title':file_title
                })
    # 无标题文档整体作为一个块
    if chunk_size == 0:
        chunks.append({
            'content':md_content,
            'title':'default',
            'file_title':file_title
        })
        chunk_size += 1

    logger.info(f'完成语义切割，切块数量:{chunk_size},内容示例:{chunks[:3]}')
    return chunks

@step_log('step_3_data_refine_chunk')
def step_3_data_refine_chunk(chunks) -> List[Dict[str,str]]:
    """对超长块做二次切分，并补充 part 与 parent_title 字段。"""

    spliter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n'],
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )


    final_chunks = []
    for chunk in chunks:
        content = chunk['content']
        if len(content) <= CHUNK_MAX_SIZE:
            chunk['part'] = 1
            chunk['parent_title'] = chunk['title']
            final_chunks.append(chunk)
        else:
            spliter_chunks = spliter.split_text(content)
            for idx, text in enumerate(spliter_chunks,start=1):
                final_chunks.append({
                    'content':text,
                    'title': f'{chunk['title']}_{idx}',
                    'file_title':chunk['file_title'],
                    'part':idx,
                    'parent_title':chunk['title']
               })
    return final_chunks

@ step_log('step_5_backup_data')
def step_4_backup_data(chunks,md_path):
    """把切分结果备份到源文件同目录的 chunks.json。"""
    chunks_json_path_obj = Path(md_path).parent / 'chunks.json'
    chunks_json_path_obj.write_text(json.dumps(chunks,ensure_ascii=False,indent=4),encoding='utf-8')



@node_log('node_document_split')
def node_document_split(state:ImportGraphState) -> ImportGraphState:
    """按标题切分文档并写入 state['chunks']。"""
    add_running_task(state['task_id'],'node_document_split')

    md_content,file_title = step_1_validate_clean(state)

    chunks = step_2_data_chunk_title(md_content,file_title)

    final_chunks = step_3_data_refine_chunk(chunks)

    step_4_backup_data(final_chunks,state['md_path'])

    state['chunks'] = final_chunks

    add_done_task(state['task_id'],'node_document_split')
    return state

if __name__ == '__main__':
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
