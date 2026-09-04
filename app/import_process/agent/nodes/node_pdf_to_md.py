import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from typing import Tuple
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config

@step_log("step_1_validate_paths")
def step_1_validate_paths(state) -> Tuple[Path,Path]:
    """
       入参: state
      出参: pdf_path_obj [Path]  local_dir_obj [Path]
      步骤:
         1. state获取对应的地址
         2. 进行非空校验(pdf_path -> none -> 结束 | local_dir 给与默认地址)
         3. 将两个参数转成Path (str -> Path )
         4. 判断pdf_path_obj是否有文件,local_dir_path 是否存在文件夹
            没有文件->抛出异常
            没有文件夹 -> 创建文件mkdir
         5. 返回两个路径地址
    """
    # 1.获取state中定义的地址
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 2.做非空判断
    if not pdf_path:
        # pdf地址不存在
        logger.error(f'pdf_path值为空,无法读取文件，直接抛出异常')
        raise ValueError('pdf_path的参数值为空,无法读取！')
    if not local_dir:
        logger.warning(f'没有传入local_dir地址,给予默认值！')
        local_dir = PROJECT_ROOT / 'output' # Path
        state['local_dir'] = str(local_dir)
    
    # 3. 将地址转换为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 4.判断文件是否存在
    if not pdf_path_obj.exists():
        # pdf地址不存在，抛出异常
        logger.error(f'pdf_path:{pdf_path_obj},无文件存在！')
        raise FileNotFoundError(f'pdf_path:{pdf_path_obj},无文件存在！')
    if not local_dir_obj.exists():
        logger.warning(f'local_dir:{local_dir_obj}地址没有文件夹，创建文件夹')
        # parents=True  可以创建多层结构文件夹
        # exist_ok=True 存在也不报错! 没有才会创建 
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_pull")
def step_2_upload_and_pull(pdf_path_obj) -> str:
    """
            入参: pdf_path_obj
           出参: zip_url (str)
           步骤:
             1. 参数校验 (minerU -> 检查下miner url和key)
             2. 申请上传地址 (minerU) [batch_id]
             3. 向执行地址进行上传文件
             4. 轮询获取返回结果(zip_url) [batch_id]
             5. 返回zip_url
    """
    #1. 校验minerU核心参数
    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    # 2. 申请上传地址和批量id
    url = f'{mineru_config.base_url}/file-urls/batch'
    header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {mineru_config.api_key}"
    }

    data = {
        "files": [
            {"name": f"{pdf_path_obj.stem}"}
        ],
        "model_version": "vlm"
    }
    # requests  get  delete -> 没有请求体 使用url路径传递参数 url?id=xx&id=xx   post put -> 有请求体
    # headers 请求头(token)
    # json 请求体数据 [字符串json]
    # data 请求体数据 [字节数据 上传文件]
    response = requests.post(url,headers=header,json=data)
    # 结果处理和解析 上传地址和批量ID
    # 请求结果 2层判定 1. http原始的状态码 (1 2 [200] 3 4 5xx) 2. 业务状态码 ( {code:0})
    # http状态码的含义: 1xx 中继状态 (请求没有完成)  2xx 成功状态(200 完整请求成功 202 端点续传成功)  3xx 重定向状态(302 304 支付宝支付)
    # 4xx 客户端错误(404 405 400) [前端]  5xx 服务器错误(500 502 504) [后端]
    # http原始的状态码
    htt_status_code = response.status_code
    if htt_status_code !=200:
        logger.error(f'申请上传地址失败,返回状态码为:{htt_status_code},请检查minerU配置!')
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{htt_status_code},请检查minerU配置!")
    # response .json() [json字符串 -> 响应体的字符数据,转成 -> 字典类型] .text [获取原始的文本类型] .content [获取字节数据(返回二进制文件)]
    result_dict = response.json()
    if result_dict['code'] != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")
        raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")
    # 获取上传地址
    file_upload_url = result_dict['data']['file_urls'][0]
    # 获取批量id
    batch_id = result_dict['data']['batch_id']

    # 3.向指定地址上传文件
    # Path  read_text writer_text  read_bytes writer_bytes

    data = pdf_path_obj.read_bytes()
    # 获取session对象
    with requests.Session() as session:
        session.trust_env = False # 纯净版的请求头,不随意携带代理的参数
        # 获取session对象
        upload_response = session.put(file_upload_url,data=data)
        # 检查网络状态
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    # 4. 轮询获取相应结果
    # 轮询就是死循环,终止条件: 1. 拿到结果  2. 失败了 3. 超时
    # 准备请求数据

    poll_url = f'{mineru_config.base_url}/extract-results/batch/{batch_id}'
    timeout = 600 #单位秒。1页pdf0.5-1秒时间
    interval_time = 3 # 设置轮询地时间间隔
    start_time = time.time()

    while True:
        # 4.1 判断是否超时
        if time.time() - start_time >= 600 :
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")
        # 4.2 进行轮询请求（鲁棒性）
        try:
            poll_response = requests.get(poll_url,headers=header)  
        except Exception as e:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        # 4.3 判断网络请求状态码
        http_poll_status_code = poll_response.status_code

        # 不等于200，若为5xx则给予再次请求机会（mineru服务端异常）
        if http_poll_status_code != 200:
            if 500 <= http_poll_status_code <600:
                logger.warning(f"可有修复的网络异常,状态码为:{http_poll_status_code}")
                time.sleep(interval_time)
                continue

        # 4.4 判断业务状态码
        poll_response_dict = poll_response.json()
        if poll_response_dict['code'] !=0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")

        # 4.5 判断具体地转化状态
        extract_result = poll_response_dict['data']['extract_result'][0]
        extract_result_state = extract_result['state']
        # 1. done 终结了 获取url地址 2. failed 终结了 失败  3. 其他 进行中 给机会
        if extract_result_state =='done':
            extract_result_url = extract_result['full_zip_url']
            # 空值校验
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")    
            # 返回压缩地址
            return extract_result_url

        elif extract_result_state =='failed':
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log('step3_download_and_extract')
def step_3_download_and_extract(zip_url,local_dir_path_obj,stem) ->Path:
    """
         1. 向指定zip地址发起请求获取响应response
         2. 将响应数据写到本地磁盘 [local_dir_path/pdf_path_obj.stem/stem.zip]
         3. 先清空解压文件夹的原文件
         4. 再次解压即可(避免出现脏数据)
         5. 检查是否存在md文件
         6. 进行md文件的命名确定 [xx.pdf -> full.md -> xx.md]
         7. 返回md_path_obj地址
    
         zip_url -> get -> content -> local_dir / stem _ result .zip  -> 清空 local_dir/stem 文件夹 ->
                                                                          local_dir/stem 文件夹中 -> local_dir/stem/stem.md
    """
    # 1. 先向指定地址发起请求获取下载的文件内容zip
    """
          参数: 
             url 
             headers 
             json 
             data 
             timeout 请求的超时时间 秒
          下载内容
    """
    response = requests.get(zip_url,timeout=30)

    #2. 将下载内容写到本地磁盘 local_dir / 文件名_result.zip
    md_path_obj = local_dir_path_obj / f'{stem}_result.zip'
    md_path_obj.write_bytes(response.content)

    # 3. 准备解压对应的文件夹 local_dir / 文件名 [先删除原文件夹内容，再解压内容]
    # local_dir /hk180/ 解压后的文件
    extract_path_obj = local_dir_path_obj / stem # 解压后的文件夹地址
    if extract_path_obj.exists():
        #递归删除文件夹下所有文件
        shutil.rmtree(extract_path_obj)
    # 创建好对应的文件夹
    extract_path_obj.mkdir(parents = True,exist_ok = True)

    #shutil.unpack_archive 参数1：压缩包路径，参数二：要解压到的路径
    shutil.unpack_archive(md_path_obj,extract_path_obj)

    # 参数1: 压缩后的文件名  xxx
    # 参数2: 压缩的格式 zip tar tar.gz  -> xxx.zip
    # 参数3: 要压缩的文件或者文件夹
    # shutil.make_archive("xx","zip",extract_path_obj)

    # 4. 检查解压后的文件夹是否存在md文件
    # 文件夹对象 . rglob ( "*.md" ) 匹配文件夹中所有的md文件,返回的是迭代器
    # [ Path ]
    md_file_list = list(extract_path_obj.rglob('*.md'))
    if not md_file_list or len(md_file_list) == 0:
        logger.error(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
        raise FileNotFoundError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    # 5. 检查md文件的命名，统一改为文件名.md
    target_md_obj = None
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_obj = md_file
            return target_md_obj

    # full.md
    for md_file in md_file_list:
        if md_file.name.lower() == "full.md":
            target_md_obj = md_file
            break
    # 随机.md
    if not target_md_obj:
        # 没有获取full.md
        # 获取第一个md文件即可
        target_md_obj = md_file_list[0]   

     # 统一修改文件名称 [一定不叫原名称.md]
        # stem.md
        # Path rename 当前的文件对象,修改成指定的文件对象 [修改磁盘中的文件]
        # Path with_name 修改路径的地址 但是不会修改磁盘  c:/full.md  -> c:/hk180.md [不会修改磁盘]
    
    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f'{stem}.md'))
    return final_md_path_obj


"""
节点作用: node_pdf_to_md  将pdf转成md,并且保存和存储,同时修改state相关的参数
入参:  [pdf_path:str :Path   local_dir:str :Path 默认的存储文件地址(项目/output) ]
出参:  [md_path:str  md_content:str]
步骤:
   1. 日志+进行中的任务记录 add_running_task
   2. step_1_validate_paths 校验pdf和输出地址
   3. step_2_upload_and_poll minerU进行交互
   4. step_3_download_and_extract 下载提取和解压
   5. 根据md地址读取对应md_content内容,并且更新state
   6. 日志+完成的任务记录  add_done_task
"""

@node_log('node_pdf_to_md')
def node_pdf_to_md(state:ImportGraphState) -> ImportGraphState:

#1. 日志+进行中的任务记录
    add_running_task(state['task_id'],'node_pdf_to_md')

    # 2. 校验pdf文件和输出路径
    pdf_path_obj,local_dir_path_obj = step_1_validate_paths(state)

    # 3. 向mineru上传文件并且轮询获取结果
    zip_url = step_2_upload_and_pull(pdf_path_obj)

    # 4. 下载处理好的文件压缩包并解压，获取md文件地址
    md_path_obj = step_3_download_and_extract(zip_url,local_dir_path_obj,pdf_path_obj.stem)

    # 5. 根据md地址读取对应md_content内容,并且更新state
    state['md_path'] = str(md_path_obj)
    state['md_content'] = md_path_obj.read_text(encoding='utf-8')

    # 6. 日志+完成的任务记录

    add_done_task(state['task_id'],'node_pdf_to_md')

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")


