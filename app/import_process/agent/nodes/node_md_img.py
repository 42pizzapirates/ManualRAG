import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

import mimetypes

from minio.deleteobjects import DeleteObject
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_core.output_parsers import StrOutputParser
from app.lm.lm_utils import get_llm_client
from langchain.messages import HumanMessage
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
from app.core.logger import logger, node_log, step_log
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.core.load_prompt import load_prompt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_img(filename:str) -> bool:
    """判断文件名是否为受支持的图片格式，后缀不区分大小写。"""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


@step_log('step_1_get_content')
def step_1_get_content(state) -> Tuple[str,Path,Path]:
    """读取 Markdown 内容，返回内容、路径对象及同级 images 目录。"""

    md_content = state['md_content']
    md_path = state['md_path']

    if not md_path:
        logger.error(f'md_path为空！无法继续')
        raise ValueError(f'md_path为空！无法继续')


    md_path_obj = Path(md_path)

    if not md_content:
        logger.warning('md_content为空，根据地址读取！')
        state['md_content'] = md_path_obj.read_text(encoding='utf-8')

    imgs_path_obj = md_path_obj.parent / 'images'
    return state['md_content'],md_path_obj,imgs_path_obj



@step_log('step_2_scan_imgs')
def step_2_scan_imgs(md_content,imgs_path_obj:Path) -> List[Tuple[str,str,Tuple[str,str]]]:
    """返回每张图片的文件名、绝对路径及前后各 100 字符的上下文。"""

    imgs_context_list = []

    for img in imgs_path_obj.iterdir():
        img_name = img.name
        if not is_supported_img(img_name):
            logger.warning(f'不支持该类型文件，文件名：{img.name}')
            continue

        # MinerU 导出同时存在 Markdown 图片语法和 HTML 表格内的 <img> 引用
        rep = re.compile(r"\!\[.*?\]\(.*?"+ re.escape(img_name)+r".*?\)")
        html_rep = re.compile(r'<img[^>]*?src=["\'][^"\']*' + re.escape(img_name) + r'["\'][^>]*>')

        match_obj = rep.search(md_content) or html_rep.search(md_content)

        if not match_obj:
            logger.warning(f'未匹配到该图片，跳过，图片名称：{img_name}')
            continue


        start,end = match_obj.span()

        pre_context = md_content[max(start-100,0):start]
        post_context = md_content[end:max(end+100,len(md_content))]

        imgs_context_list.append((img_name,str(img),(pre_context,post_context)))
    return imgs_context_list


@step_log('step_3_vlm_summary')
def step_3_vlm_summary(imgs_context_list,stem) -> Dict[str,str]:
    """逐张调用视觉模型生成图片描述，返回 {图片名: 描述}。"""

    img_summaries_dict = {}

    vm_model = get_llm_client(lm_config.lv_model)

    for img_name,img_path,context in imgs_context_list:

        apply_api_rate_limit()

        img_context_prompt = load_prompt('image_summary',root_folder=stem,image_content=context)
        img_path_obj = Path(img_path)
        img_data = base64.b64encode(img_path_obj.read_bytes()).decode(encoding='utf-8')
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # 以 data URL 传输本地图片，避免依赖公网可访问地址
                        "url": f"data:{mimetypes.guess_type(img_name)[0]};base64,{img_data}"
                    },
                },
                {"type": "text", "text":img_context_prompt},
            ]
        )

        img_summaries_dict[img_name] = vm_model.invoke([message]).content

    return img_summaries_dict
     

@step_log('step_4_imgs_upload_replace')
def step_4_imgs_upload_replace(imgs_context_list,img_summaries_dict,md_content,stem) -> str:
    """上传图片到 MinIO，并将 Markdown 中的本地图片引用替换为描述和公网地址。"""

    minio_client = get_minio_client()

    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # 前缀不能以 / 开头，必须以 / 结尾
        prefix=f'{minio_config.minio_img_dir[1:]}/{stem}/',
        recursive=True
    )
    delete_obj_list = [DeleteObject(obj.object_name) for obj in object_list]

    # remove_objects 是惰性迭代，必须消费返回值才会真正执行删除
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_obj_list
    )
    for error in errors:
        logger.warning(f'删除失败！原因：{error}')

    img_url_dict = {}
    for img_name,img_path,_ in imgs_context_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f'{minio_config.minio_img_dir}/{stem}/{img_name}',
                file_path=img_path,
                content_type=mimetypes.guess_type(img_name)[0]
            )
            img_url = (f'{minio_config.public_base_url}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{img_name}')
            logger.debug(f'图片：{img_name}上传成功！回显地址：{img_url}')
            img_url_dict[img_name] = img_url
        except Exception as e:
            logger.warning(f'上传图片{img_name}失败，跳过！')
            continue


    img_info = {}
    if not img_url_dict:
        logger.warning('图片全部上传失败！')
        return md_content
    for img_name,img_url in img_url_dict.items():
        img_info[img_name] = (img_summaries_dict[img_name],img_url)

    for img_name,(img_summary,img_url) in img_info.items():

        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        html_rep = re.compile(r'<img[^>]*?src=["\'][^"\']*' + re.escape(img_name) + r'["\'][^>]*>')
        # 通过 lambda 传入替换文本，避免描述中的正则元字符被二次解析
        md_content = rep.sub(lambda _:f'![{img_summary}]({img_url})',md_content)
        # HTML 块内的 Markdown 语法不会被渲染，需要单独替换 img 标签
        md_content = html_rep.sub(lambda _:f'<img src="{img_url}" alt="{img_summary}"/>',md_content)

    return md_content

@step_log('step_5_backup_md')
def step_5_backup_md(new_md_content,md_path_obj:Path) -> str:
    """将处理后的内容写入同目录的 *_new.md，返回新文件路径。"""
    new_md_path_obj = md_path_obj.with_name(f'{md_path_obj.stem}_new.md')
    new_md_path_obj.write_text(new_md_content,encoding='utf-8')
    return str(new_md_path_obj)

@node_log('node_md_img')
def node_md_img(state:ImportGraphState) -> ImportGraphState:
    add_running_task(state['task_id'],'node_md_img')
    md_content,md_path_obj,imgs_path_obj =step_1_get_content(state)

    img_context_list = step_2_scan_imgs(md_content,imgs_path_obj)

    img_summaries_dict = step_3_vlm_summary(img_context_list,md_path_obj.stem)

    print(f'以获取图片总结描述信息：{img_summaries_dict}')

    new_md_content = step_4_imgs_upload_replace(img_context_list,img_summaries_dict,md_content,md_path_obj.stem)

    new_md_path = step_5_backup_md(new_md_content,md_path_obj)

    state['md_content'] = new_md_content
    state['md_path'] = new_md_path

    add_done_task(state['task_id'],'node_md_img')

    return state

if __name__ == "__main__":
    """单独运行该文件时执行 MD 图片处理全流程。"""
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")








