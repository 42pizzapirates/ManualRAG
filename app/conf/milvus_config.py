from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MilvusConfig:
    milvus_url: str
    milvus_user: str
    milvus_password: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str

milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    milvus_user=os.getenv("MILVUS_USER"),
    milvus_password=os.getenv("MILVUS_PASSWORD"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
