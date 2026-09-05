import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

import mimetypes

# MinIO相关依赖
from minio.deleteobjects import DeleteObject
# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_core.output_parsers import StrOutputParser
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger, node_log, step_log
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_img(filename:str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


"""
1. 进行任务的管理 running_task
2. 进行参数校验和获取材料 step_1_get_content
3. 扫描图片的上下文 step_2_scan_images
4. 调用视觉模型,传入图片的对应的上下文,获取图的文本内容 step_3_image_summary
5. 将图片文件传入到minio服务器,并且进行md中的图片替换(描述 网络地址) step_4_upload_images_replace
6. 将新的md内容进行备份处理 step_5_backup_md_file
7. 更新state信息 md_path md_content
8. 进行任务的管理 done_task
"""

@step_log('step_1_get_content')
def step_1_get_content(state) -> Tuple[str,Path,Path]:
    """
            入参: state [md_content md_path]
           出参: md_content , md_path_obj:Path , images_dir_obj : Object
           步骤:
                1. 获取md_content 和 md_path
                2. 非空校验 (为空就异常 ValueError)
                3. md_path 转成 md_path_obj Path
                4. md_path_obj 获取 images所在的文件夹  .parent / images
    """

    # 1.获取基本信息
    md_content = state['md_content']
    md_path = state['md_path']

    # 2. 非空校验
    if not md_path:
        logger.error(f'md_path为空！无法继续')
        raise ValueError(f'md_path为空！无法继续')


    # 3. md_path转为Path对象
    md_path_obj = Path(md_path)

    if not md_content:
        logger.warning('md_content为空，根据地址读取！')
        state['md_content'] = md_path_obj.read_text(encoding='utf-8')

    # 4. 获取图片文件夹路径
    imgs_path_obj = md_path_obj.parent / 'images'
    return state['md_content'],md_path_obj,imgs_path_obj



@step_log('step_2_scan_imgs')
def step_2_scan_imgs(md_content,imgs_path_obj:Path) -> List[Tuple[str,str,Tuple[str,str]]]:
    """
        作用: 找到图片的本身信息和图片的上下文信息
        入参: md_content ,  images_dir_obj
        出参: [("图片名.xx","图片地址",(上文 -> 100,下文 -> 100))]
        步骤:
            1. 获取图片文件夹中所有图片对象
            2. 循环遍历图片对象
            3. 定义对应的正则规则 ![](图片名)
            4. md中进行正在匹配 Match = re.search(md)
            5. Match对象获取图片的坐标信息 ( .span() .star() .end()  .group() )
            6. md中截取上文和下文 md_content [ star-100 : star]   [end , end+100]
            7. 拼接结果内容 (对象.name , str(对象) , (上文,下文))
    """

    # 准备一个列表接收整体结果
    imgs_context_list = []

    #循环遍历
    # img_file 即每个图片的Path对象
    for img in imgs_path_obj.iterdir():
        # img为Path对象
        img_name = img.name
        if not is_supported_img(img_name):
            logger.warning(f'不支持该类型文件，文件名：{img.name}')
            continue

        # 定义对应的正则规则
        # re.escape(image_name \. \?)避免数据中存在特殊符号，导致正则爆炸
        # MinerU导出的md中图片有两种引用形式：markdown语法 ![](images/x.jpg) 和 HTML表格内的 <img src="images/x.jpg"/>
        rep = re.compile(r"\!\[.*?\]\(.*?"+ re.escape(img_name)+r".*?\)")
        html_rep = re.compile(r'<img[^>]*?src=["\'][^"\']*' + re.escape(img_name) + r'["\'][^>]*>')

        # 对md进行正则匹配（markdown和HTML两种形式任一命中即可）
        match_obj = rep.search(md_content) or html_rep.search(md_content)

        # 非空校验
        if not match_obj:
            logger.warning(f'未匹配到该图片，跳过，图片名称：{img_name}')
            continue


        # 获取匹配的文本坐标,.span() -> (start,end)  | star() start  |  end() -> end  |  group() -> 匹配到的字符串  | .string 获取元数据
        start,end = match_obj.span()

        # 获取上下文 100字符
        # 注意下标越界问题
        pre_context = md_content[max(start-100,0):start]
        post_context = md_content[end:max(end+100,len(md_content))]

        # 拼接最终内容
        imgs_context_list.append((img_name,str(img),(pre_context,post_context)))
    return imgs_context_list


@step_log('step_3_vlm_summary')
def step_3_vlm_summary(imgs_context_list,stem) -> Dict[str,str]:
    """
        作用: 使用视觉模型识别图片描述内容
       入参: [("图片名.xx","图片地址",(上文 -> 100,下文 -> 100))]   md_path_obj.stem (图片所在的文件夹的名字)
       出参: dict {图片名 : 图片的总结和描述}
       步骤:
             1. 准备工作
                获取模型客户端对象
             2. 循环上次获取图片的上下文
             3. 封装提示词
             4. 使用模型执行
             5. 获取返回结果 装到 dict中
             6. 返回结果
    """

    # 准备存放结果的字典
    img_summaries_dict = {}

    # 1. 获取模型对象
    vm_model = get_llm_client(lm_config.lv_model)

    # 2. 循环总结每份图片
    for img_name,img_path,context in imgs_context_list:

        # 设置模型请求窗口限制（滑动窗孔）
        apply_api_rate_limit()

        # 3. 构建提示词
        # 获取图片描述信息
        img_context_prompt = load_prompt('image_summary',root_folder=stem,image_content=context)
        # 获取图片内容
        # base64.b64encode(字节) 普通字节转成base64的字节 .decode(encoding="utf-8") 转成字符串
        img_path_obj = Path(img_path)
        img_data = base64.b64encode(img_path_obj.read_bytes()).decode(encoding='utf-8')
        # 整合提示词
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # 1. 图片地址 必须公网地址 http://xxxx [图片上传到minio]
                        # 2. base64的字符串  字节 -> 字符 -> 服务器 -> 字符 -> 字节
                        "url": f"data:{mimetypes.guess_type(img_name)[0]};base64,{img_data}"
                    },
                },
                # 本次图片的上下文文本描述
                {"type": "text", "text":img_context_prompt},
            ]
        )

        img_summaries_dict[img_name] = vm_model.invoke([message]).content

    return img_summaries_dict
     

@step_log('step_4_imgs_upload_replace')
def step_4_imgs_upload_replace(imgs_context_list,img_summaries_dict,md_content,stem) -> str:
    """
    作用: 将图片传递到minio,拼接网络地址,联合混合模型返回的图片描述,一起替换md_content内容
    入参: md_content  [("图片名.xx","图片地址",(上文 -> 100,下文 -> 100))] dict {图片名 : 图片的总结和描述}  md_path_obj.stem
    出参: md_content 替换后的  ![](本地地址) -> ![summary](网络地址)
    步骤:
        1.获取minio的客户端对象 minio_client
        2.删除原文件名对应的minio对象
        3.循环"图片地址"地址向minio服务器传递文件,并且拼接(端点/桶名/对象名)和记录地址 {图片名:图片地址}
        4. {图片名:图片地址}   {图片名 : 图片的总结和描述} -> 合并到一起 -> {图片名 : (图片描述和总结,图片地址)}
        5. md_content内容替换,循环处理和替换即可 正则 |  rep.sub("要替换进入的内容",md_content)
        6. 最终返回md_content
    """

    # 1. 获取minio的客户端对象 minio_client
    minio_client = get_minio_client()

    # 2.删除源文件名对应的minio对象
    # 先查询，再删除
    object_list = minio_client.list_objects(
        # 参数1：要删除桶的名字
        bucket_name=minio_config.bucket_name,
        # 参数2：前缀 ([1:]) 不能使用 /开头 必须使用 /结尾
        prefix=f'{minio_config.minio_img_dir[1:]}/{stem}/',
        # 参数3：递归查询
        recursive=True
    )
    delete_obj_list = [DeleteObject(obj.object_name) for obj in object_list]

    # 底层是懒执行! 必须调用结果! 否则不执行删除!!!
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_obj_list
    )
    for error in errors:
        logger.warning(f'删除失败！原因：{error}')

    # 3.循环"图片地址"地址向minio服务器传递文件,并且拼接(端点/桶名/对象名)和记录地址 {图片名:图片地址}
    img_url_dict = {}
    for img_name,img_path,_ in imgs_context_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f'{minio_config.minio_img_dir}/{stem}/{img_name}',
                file_path=img_path,
                content_type=mimetypes.guess_type(img_name)[0]
            )
            # 拼接图片的网络地址，端点+桶+对象名
            img_url = (f'http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{img_name}')
            logger.debug(f'图片：{img_name}上传成功！回显地址：{img_url}')
            # 存储图片和对应网络地址对
            img_url_dict[img_name] = img_url
        except Exception as e:
            logger.warning(f'上传图片{img_name}失败，跳过！')
            continue


    # 4.  {图片名:图片地址}   {图片名 : 图片的总结和描述} -> 合并到一起 -> {图片名 : (图片描述和总结,图片地址)}
    # {图片名: (图片描述和总结, 图片地址)}
    img_info = {}
    if not img_url_dict:
        logger.warning('图片全部上传失败！')
        return md_content
    for img_name,img_url in img_url_dict.items():
        img_info[img_name] = (img_summaries_dict[img_name],img_url)

    # 5.替换md_content内容，循环正则替换
    for img_name,(img_summary,img_url) in img_info.items():

        # rep .findall finditer match search sub
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        html_rep = re.compile(r'<img[^>]*?src=["\'][^"\']*' + re.escape(img_name) + r'["\'][^>]*>')
                # 参数1: 要替换入的内容
                # 参数2: 要对哪个文档进行替换
                # 返回值: 替换后的新文档内容
                # 第一种方案,替换入的字符串也可能出现被正则识别的元字符,也会会出现异常!
                # md_content =  rep.sub(f"![{image_summary}]({image_url})",md_content)
                # 第二种方案,使用lambda表达式处理, 正则就不会处理,只会使用原生的正则返回值!
        md_content = rep.sub(lambda _:f'![{img_summary}]({img_url})',md_content)
        # HTML表格内的<img>标签同样替换，保留HTML结构（markdown语法在HTML块内不会被渲染）
        md_content = html_rep.sub(lambda _:f'<img src="{img_url}" alt="{img_summary}"/>',md_content)

    return md_content

@step_log('step_5_backup_md')
def step_5_backup_md(new_md_content,md_path_obj:Path) -> str:
    """
    将新的md_content内容写入到本地磁盘! xx.md xx_new.md
    :param new_md_content:
    :param md_path_obj:
    :return: 新md的地址 str
    """
    # with_name -> 磁盘没关系,获取一个新的path / rename 重命名 修改磁盘
    new_md_path_obj = md_path_obj.with_name(f'{md_path_obj.stem}_new.md')
    # 备份
    new_md_path_obj.write_text(new_md_content,encoding='utf-8')
    return str(new_md_path_obj)

@node_log('node_md_img')
def node_md_img(state:ImportGraphState) -> ImportGraphState:
    #1.记录进行状态
    add_running_task(state['task_id'],'node_md_img')
    # 2. 准备和校验
    md_content,md_path_obj,imgs_path_obj =step_1_get_content(state)

    # 3. 扫描图片上下文
    img_context_list = step_2_scan_imgs(md_content,imgs_path_obj)

    # 4. 调用模型获取图片总结描述
    img_summaries_dict = step_3_vlm_summary(img_context_list,md_path_obj.stem)

    print(f'以获取图片总结描述信息：{img_summaries_dict}')

    # 5. 上传图片到minio，并替换md_content
    new_md_content = step_4_imgs_upload_replace(img_context_list,img_summaries_dict,md_content,md_path_obj.stem)

    # 6. 备份新的md文件
    new_md_path = step_5_backup_md(new_md_content,md_path_obj)

    # 7. 更新state
    state['md_content'] = new_md_content
    state['md_path'] = new_md_path

    add_done_task(state['task_id'],'node_md_img')

    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    # print(mimetypes.guess_type("mv.jpg"))
    # print(mimetypes.guess_type("mv.png"))
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")








