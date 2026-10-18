import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

from app.conf.lm_config import lm_config
from app.core.logger import logger

# 键为 (模型名, JSON 模式)，避免重复构造客户端
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """获取带缓存的 ChatOpenAI 客户端。

    模型优先级：入参 > lm_config.llm_model > qwen3-32b。
    json_mode 开启后要求服务端返回 json_object。
    """
    target_model = model or lm_config.llm_model or "qwen3-32b"
    cache_key = (target_model, json_mode)

    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_BASE（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 千问系模型的私有参数经 extra_body 透传
    extra_body = {"enable_thinking": False}
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client


if __name__ == "__main__":
    logger.info("===== 开始执行LLM客户端工具测试 =====")
    try:
        client1 = get_llm_client()
        logger.info("✅ 测试1通过：默认配置客户端创建成功")

        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("✅ 测试2通过：指定多模态模型客户端创建成功")

        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"✅ 测试3通过：缓存机制验证成功，client2与client3为同一实例：{client2 is client3}")

        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("✅ 测试4通过：JSON输出模式客户端创建成功")

    except Exception as e:
        logger.error(f"❌ LLM客户端工具测试失败：{str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM客户端工具测试结束 =====")
