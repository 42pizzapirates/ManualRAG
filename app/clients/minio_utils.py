import os
import json

from minio import Minio

from app.conf.minio_config import minio_config
from app.core.logger import logger

_minio_client = None


def _create_minio_client() -> Minio:
    """使用配置中的端点和凭证创建 MinIO 客户端。"""
    return Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure
    )


def _set_bucket_policy(bucket_name: str) -> str:
    """生成仅允许匿名读取对象的桶策略。"""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    return json.dumps(policy)


def _create_bucket_ready(client: Minio):
    """确保桶存在；新建桶时开放匿名读取，供前端直接加载图片。"""
    bucket_name = minio_config.bucket_name
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
        logger.info(f"MinIO桶 {bucket_name} 已创建，并设置访问策略")
    else:
        logger.info(f"MinIO桶 {bucket_name} 已存在，无需重复创建")


def get_minio_client() -> Minio:
    """返回进程内共享的 MinIO 客户端，首次调用时初始化桶。

    无锁实现，仅适用于单线程初始化场景。
    """
    global _minio_client

    if _minio_client is None:
        logger.info("开始初始化MinIO客户端（首次调用，执行懒加载）")
        client = _create_minio_client()
        _create_bucket_ready(client)
        _minio_client = client
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    return _minio_client
