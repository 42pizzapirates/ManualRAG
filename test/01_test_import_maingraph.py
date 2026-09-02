import json

from app.import_process.agent.main_graph import import_graph as kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

# 1. 创建state
state = create_default_state(task_id="007",local_file_path="xxx.pdf")
# 2. 执行编译后的图对象
result = kb_import_app.invoke(state)
# json.dumps 将字典数据转成字符串输出,可以进行格式化处理
# json.dump  将字典写入磁盘文件中!
# json.loads 将字符串加载成python字典
# json.load  将文件磁盘的字符串加载成python字典
logger.info(f"执行结果: {json.dumps(result, indent=4, ensure_ascii=False)}")
# 3. 查看编译的图结构
logger.info(f"图编译结构:{kb_import_app.get_graph().print_ascii()}")