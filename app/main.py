from __future__ import annotations


from fastapi import FastAPI
from app.api.routes import make_router
from app.core.config import get_settings
from app.integrations.milvus_retriever import MilvusRetriever
from app.integrations.postgres_store import PostgresStore
from app.integrations.qwen_openai import QwenClient
from app.integrations.redis_memory import RedisMemory
from app.graphs.rag_graph import GraphDeps, build_graph



def create_app() -> FastAPI:
    settings = get_settings()

    if not settings.redis_url:
        raise RuntimeError("Missing REDIS_URL")

    if not settings.qwen_api_key:
        raise RuntimeError("Missing QWEN_API_KEY (or DASHSCOPE_API_KEY)")

    if not settings.milvus_uri or not settings.milvus_token:
        raise RuntimeError("Missing MILVUS_URI/MILVUS_TOKEN")

    qwen = QwenClient(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
    # 使用与建库脚本一致的 BAAI/bge-large-zh 作为 embedding_fn
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-large-zh')
    if torch.cuda.is_available():
        model = model.to('cuda')
    class STWrapper:
        def __init__(self, model):
            self.model = model
        def encode_documents(self, texts):
            texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    embedding_fn = STWrapper(model)

    def embed_texts(texts: list[str]) -> list[list[float]]:
        vecs = embedding_fn.encode_documents(texts)
        return [list(map(float, v)) for v in vecs]
    retriever = MilvusRetriever(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        collection=settings.milvus_collection,
        embed_fn=embed_texts,
        top_k=settings.milvus_top_k,
    )

    class LLMWrapper:
        def chat(self, *, messages):
            return qwen.chat(model=settings.qwen_chat_model, messages=messages)

        def chat_with_tools(self, *, messages, tools, tool_executor):
            return qwen.chat_with_tools(
                model=settings.qwen_chat_model,
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
            )

    memory = RedisMemory.from_url(
        settings.redis_url, prefix=settings.redis_prefix, ttl_seconds=settings.session_ttl_seconds
    )

    pg_store = None
    if settings.postgres_dsn:
        pg_store = PostgresStore(settings.postgres_dsn)

    graph = build_graph(
        GraphDeps(router_mode=settings.router_mode, retriever=retriever, llm=LLMWrapper())
    )

    app = FastAPI(title="AI Agent (LangGraph + RAG)")
    app.include_router(make_router(memory=memory, graph=graph, settings=settings, pg_store=pg_store))

    @app.get("/health")
    def health():
        return {"ok": True, "env": settings.app_env}
    return app


app = create_app()
