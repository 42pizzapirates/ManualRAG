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
    """校验 PDF 路径并确保输出目录存在。"""
    pdf_path = state['pdf_path']
    output_dir = state['local_dir']

    if not pdf_path:
        logger.error('pdf地址不存在！')
        raise ValueError('pdf地址不存在！')
    if not output_dir:
        logger.warning(f'没有传入outpu_dir地址，给予默认值')
        output_dir = PROJECT_ROOT / 'output'
        state['local_dir'] = output_dir
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(output_dir)

    if not pdf_path_obj.exists():
        logger.error(f'{pdf_path}文件不存在！')
        raise FileNotFoundError(f'{pdf_path}文件不存在！')
    if not output_dir_obj.exists():
        logger.warning(f'{output_dir_obj}路径不存在！自动创建路径')
        output_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj,output_dir_obj


@step_log("step_2_upload_and_pull")
def step_2_upload_and_pull(pdf_path_obj) -> str:
    """上传 PDF 到 MinerU 并轮询解析结果，返回结果压缩包地址。"""

    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error('mineru配置错误，请检查相关环境配置！')
        raise ValueError('mineru配置错误，请检查相关环境配置！')

    # MinerU 采用“申请预签名地址 -> 直传 -> 按 batch_id 轮询”的两阶段流程
    url = f'{mineru_config.base_url}/file-urls/batch'
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key}"
    }
    data = {
        'files':[
            {'name':pdf_path_obj.name}
        ],
        'model_version':'vlm'
    }


    response = requests.post(url,headers=header,json=data)

    # 同时校验 HTTP 状态码和 MinerU 业务状态码
    htt_status_code = response.status_code
    if htt_status_code != 200:
        logger.error(f'申请上传地址失败，返回状态码为{htt_status_code},检查mineru配置')
        raise RuntimeError(f'申请上传地址失败，返回状态码为{htt_status_code},检查mineru配置')

    result_dict = response.json()
    if result_dict['code'] != 0:
        logger.error(f'申请地址网络状态正常,但业务失败！错误码{result_dict['code']}')
        raise RuntimeError(f'申请地址网络状态正常,但业务失败！错误码{result_dict['code']}')

    upload_url = result_dict['data']['file_urls'][0]

    batch_id = result_dict['data']['batch_id']

    data = pdf_path_obj.read_bytes()

    with requests.Session() as session:
        # 预签名地址由 MinerU 签发，禁用环境代理以免签名被代理改写
        session.trust_env = False
        upload_response = session.put(upload_url,data=data)
        if upload_response.status_code != 200:
            logger.error(f'上传文件失败，返回状态码{upload_response.status_code}')
            raise RuntimeError(f'上传文件失败，返回状态码{upload_response.status_code}')

    poll_url = f'{mineru_config.base_url}/extract-results/batch/{batch_id}'
    time_out = 600  
    interval_time = 3
    start_time = time.time()
    while True:
        if time.time() - start_time > time_out:
            logger.error(f'轮询超时，检查mineru配置！')
            raise RuntimeError(f'轮询超时，检查mineru配置！')
        try:
            poll_response = requests.get(poll_url,headers=header)
        except Exception as e:
            logger.warning(f'请求出现异常！稍后重试！')
            time.sleep(interval_time)
            continue
        http_poll_status_code = poll_response.status_code
        # 5xx 视为服务端临时故障，可重试；其余非 200 直接失败
        if http_poll_status_code != 200:
            if 500 <= http_poll_status_code <600:
                logger.warning(f'可能修复的网络异常：状态码：{http_poll_status_code}')
                time.sleep(interval_time)
                continue
            else:
                logger.error(f'不可修复的网络状态异常,状态码：{http_poll_status_code}')
                raise RuntimeError(f'不可修复的网络状态异常,状态码：{http_poll_status_code}')
            
        poll_response_dict = poll_response.json()
        if poll_response_dict['code'] != 0:
            logger.error(f'轮询业务异常，错误码：{poll_response_dict['code']}')
            raise RuntimeError(f'轮询业务异常，错误码：{poll_response_dict['code']}')

        extract_result = poll_response_dict['data']['extract_result'][0]
        extract_result_state = extract_result['state']
        if extract_result_state == 'done':
            extract_result_url = extract_result['full_zip_url']
            if not extract_result_url:
                logger.error(f'已经完成了解析，但url地址为空！')
                raise RuntimeError(f'已经完成了解析，但url地址为空！')
            return extract_result_url
        elif extract_result_state == 'failed':
            logger.error(f'文件解析失败！失败信息：{extract_result['err_msg']}')
            raise RuntimeError(f'文件解析失败！失败信息：{extract_result['err_msg']}')
        else:
            logger.warning(f'文件解析尚未完成，状态：{extract_result_state}')
            time.sleep(interval_time)
            continue



             
@step_log('step3_download_and_extract')
def step_3_download_and_extract(zip_url,local_dir_path_obj,stem) ->Path:
    """下载并解压 MinerU 结果，返回统一命名为 {stem}.md 的文件路径。"""

    response = requests.get(zip_url,timeout=30)

    md_path_obj = local_dir_path_obj / f'{stem}_result.zip'
    md_path_obj.write_bytes(response.content)

    # 先清理旧目录，避免上次解析残留文件混入本次结果
    extract_path_obj = local_dir_path_obj / stem
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)
    extract_path_obj.mkdir(parents=True,exist_ok=True)
    shutil.unpack_archive(md_path_obj,extract_path_obj)


    md_file_list = list(extract_path_obj.rglob('*.md'))
    if not md_file_list or len(md_file_list) ==0:
        logger.error(f'文件解压失败，{extract_path_obj}没有md文件')
        raise FileNotFoundError(f'文件解压失败，{extract_path_obj}没有md文件')
    


    # MinerU 输出可能是 full.md 或随机文件名，统一改为源文件名
    target_md_obj = None
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_obj = md_file
            return target_md_obj
    for md_file in md_file_list:
        if md_file.name.lower() == 'full.md':
            target_md_obj = md_file
            break
    if not target_md_obj:
        target_md_obj = md_file_list[0]


    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f'{stem}.md'))
    return final_md_path_obj


@node_log('node_pdf_to_md')
def node_pdf_to_md(state:ImportGraphState) -> ImportGraphState:
    """解析 PDF 并把 Markdown 内容与路径写回 state。"""

    add_running_task(state['task_id'],'node_pdf_to_md')

    pdf_path_obj,output_path_obj = step_1_validate_paths(state)

    zip_url = step_2_upload_and_pull(pdf_path_obj)


    md_path_obj = step_3_download_and_extract(zip_url,output_path_obj,pdf_path_obj.stem)

    state['md_content'] = md_path_obj.read_text(encoding='utf-8')
    state['md_path'] = str(md_path_obj)

    add_done_task(state['task_id'],'node_pdf_to_md')

    return state

if __name__ == "__main__":

    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180使用说明书.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")


