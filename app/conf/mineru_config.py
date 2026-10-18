from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MineruConfig:
    base_url: str
    api_key : str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_key=os.getenv("MINERU_API_TOKEN")
)
