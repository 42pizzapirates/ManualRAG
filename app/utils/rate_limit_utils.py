import time
from collections import deque
from typing import Deque
from app.core.logger import logger

_GLOBAL_REQUEST_TIMES: Deque[float] = deque()


def apply_api_rate_limit(
        max_requests: int = 500,
        window_seconds: int = 60
) -> None:
    """滑动窗口限流；达到上限时阻塞至窗口释放出名额。"""
    current_time = time.time()
    while _GLOBAL_REQUEST_TIMES and current_time - _GLOBAL_REQUEST_TIMES[0] >= window_seconds:
        _GLOBAL_REQUEST_TIMES.popleft()
    if len(_GLOBAL_REQUEST_TIMES) >= max_requests:
        sleep_duration = window_seconds - (current_time - _GLOBAL_REQUEST_TIMES[0])
        if sleep_duration > 0:
            logger.debug(f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
            time.sleep(sleep_duration)
            # 等待期间可能有请求过期，重新清理后再继续
            current_time = time.time()
            while _GLOBAL_REQUEST_TIMES and current_time - _GLOBAL_REQUEST_TIMES[0] >= window_seconds:
                _GLOBAL_REQUEST_TIMES.popleft()
    _GLOBAL_REQUEST_TIMES.append(current_time)
    logger.debug(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(_GLOBAL_REQUEST_TIMES)}")
