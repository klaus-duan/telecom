from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str

    redis_url: str
    redis_prefix: str
    session_ttl_seconds: int

    milvus_uri: str
    milvus_token: str
    milvus_collection: str
    milvus_top_k: int

    qwen_api_key: str
    qwen_base_url: str
    qwen_chat_model: str
    qwen_embed_model: str

    postgres_dsn: str

    router_mode: str  # heuristic|llm


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


def get_settings() -> Settings:
    # 本地开发可用 python-dotenv 载入 .env；生产用环境变量/secret
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except Exception:
        pass

    return Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_prefix=os.getenv("REDIS_PREFIX", os.getenv("APP_ENV", "dev")),
        session_ttl_seconds=_get_int("SESSION_TTL_SECONDS", 2 * 60 * 60),
        milvus_uri=(os.getenv("MILVUS_URI") or os.getenv("ZILLIZ_URI") or "").strip(),
        milvus_token=(os.getenv("MILVUS_TOKEN") or os.getenv("ZILLIZ_TOKEN") or "").strip(),
        milvus_collection=os.getenv("MILVUS_COLLECTION", "qa_collection"),
        milvus_top_k=_get_int("MILVUS_TOP_K", 5),
        qwen_api_key=(os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip(),
        # Qwen 通常提供 OpenAI 兼容接口；你可以按控制台给的地址覆盖
        qwen_base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        qwen_chat_model=os.getenv("QWEN_CHAT_MODEL", "qwen-plus"),
        qwen_embed_model=os.getenv("QWEN_EMBED_MODEL", "text-embedding-v2"),
        postgres_dsn=(os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or "").strip(),
        router_mode=os.getenv("ROUTER_MODE", "heuristic"),
    )
