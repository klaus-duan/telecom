from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

from pymilvus import MilvusClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedDoc:
    id: int | str
    score: float
    question: str | None
    knowledge: str

    def to_citation(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "question": self.question,
        }


class MilvusRetriever:
    def __init__(
        self,
        *,
        uri: str,
        token: str,
        collection: str,
        embed_fn,
        top_k: int = 5,
    ) -> None:
        self._uri = uri
        self._token = token
        self._client: MilvusClient | None = None
        self._collection = collection
        self._embed_fn = embed_fn
        self._top_k = top_k

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, token=self._token)
        return self._client

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        try:
            vec = self._embed_fn([query])[0]
        except Exception as e:
            logger.exception("MilvusRetriever embed failed: %s", e)
            return []

        try:
            client = self._get_client()
            res = client.search(
                collection_name=self._collection,
                data=[vec],
                anns_field="question_emb",
                limit=self._top_k,
                output_fields=["id", "question", "knowledge"],
            )
        except Exception as e:
            # 典型：gRPC DEADLINE_EXCEEDED / 网络不可达 / token/uri 错误
            logger.exception("MilvusRetriever search failed: %s", e)
            return []

        docs: list[RetrievedDoc] = []
        # pymilvus search 返回二维 list：每个 query 对应一个 hits 列表
        hits = res[0] if res else []
        for h in hits:
            # 兼容不同返回格式：Hit 对象 / dict
            if isinstance(h, dict):
                score_val = h.get("distance", h.get("score", 0.0))
                entity = h.get("entity") or h.get("fields") or h
            else:
                score_val = getattr(h, "distance", getattr(h, "score", 0.0))
                entity = getattr(h, "entity", None) or {}

            if hasattr(entity, "to_dict"):
                try:
                    entity = entity.to_dict()
                except Exception:
                    pass

            if not isinstance(entity, dict):
                # 最后兜底：尝试从属性读取
                entity = {
                    "id": getattr(entity, "id", None),
                    "question": getattr(entity, "question", None),
                    "knowledge": getattr(entity, "knowledge", None),
                }

            kid = entity.get("id")
            q = entity.get("question")
            kb = entity.get("knowledge") or ""
            docs.append(
                RetrievedDoc(
                    id=kid,
                    score=float(score_val),
                    question=q if isinstance(q, str) else None,
                    knowledge=str(kb),
                )
            )
        return docs
