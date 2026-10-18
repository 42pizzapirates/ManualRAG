from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps:int = 0)->Path:
    """返回当前文件向上第 ps 级的目录。"""
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    """定位项目根目录：环境变量优先，其次向上查找标识文件。"""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量PROJECT_ROOT未配置")


PROJECT_ROOT = get_project_root(".env")
