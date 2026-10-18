from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    minio_img_dir: str
    minio_secure: bool
    public_base_url: str

# 回显地址默认沿用 endpoint，换域名或启用 HTTPS 时用 MINIO_PUBLIC_BASE_URL 覆盖
_minio_endpoint = os.getenv("MINIO_ENDPOINT")
minio_config = MinIOConfig(
    endpoint=_minio_endpoint,
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True",
    public_base_url=os.getenv("MINIO_PUBLIC_BASE_URL") or f"http://{_minio_endpoint}"
)
