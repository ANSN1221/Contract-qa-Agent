"""检索：BM25 与向量双路召回，用加权 RRF 融合。

为什么用 RRF 而不是加权分数：BM25 分数无上界、余弦相似度在 0–1 之间，
加权求和必须先做归一化，而归一化方式本身又是一个需要调的超参。
RRF 只看排名，天然免疫量纲问题。

为什么 BM25 权重更高：在本项目的语料上实测，向量召回对「投标保证金」这类
精确术语的问题会返回「反商业贿赂承诺书」这类语义相近但无用的条款，
而关键词召回能精确命中。向量负责语义泛化，关键词负责术语命中，
权重留成环境变量，D10 做消融时可以直接改。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache

from contract_qa.config import Settings, get_settings
from contract_qa.index import BM25_FILE, COLLECTION_NAME, ClauseRecord, load_records, tokenize
from contract_qa.llm import embed_texts

RRF_K = 60
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1.0"))
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", "0.5"))


@dataclass
class Hit:
    clause_id: str
    score: float            # 融合分（RRF）
    dense_rank: int | None  # 向量召回排名，1 起
    bm25_rank: int | None   # 关键词召回排名，1 起
    record: ClauseRecord


class Retriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.records = {record.clause_id: record for record in load_records(self.settings)}
        self._bm25 = None
        self._clause_ids: list[str] = []
        self._collection = None

    # ---- 懒加载：只用 BM25 时不需要向量库，反之亦然 ----

    def _load_bm25(self):
        if self._bm25 is None:
            path = self.settings.processed_dir / BM25_FILE
            if not path.exists():
                raise FileNotFoundError(f"缺少 BM25 索引 {path}，请先运行 contract-qa index")
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            self._bm25 = payload["bm25"]
            self._clause_ids = payload["clause_ids"]
        return self._bm25

    def _load_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    # ---- 两路召回 ----

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        index = self._load_bm25()
        scores = index.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._clause_ids, scores, strict=False), key=lambda item: item[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]

    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        collection = self._load_collection()
        vector = embed_texts([query], self.settings)[0]
        result = collection.query(query_embeddings=[vector], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances, strict=False)]

    # ---- 融合 ----

    def search(self, query: str, top_k: int = 8, candidates: int = 50) -> list[Hit]:
        dense = self.dense_search(query, candidates)
        sparse = self.bm25_search(query, candidates)

        dense_rank = {cid: rank for rank, (cid, _) in enumerate(dense, start=1)}
        bm25_rank = {cid: rank for rank, (cid, _) in enumerate(sparse, start=1)}

        fused: dict[str, float] = {}
        for ranks, weight in ((dense_rank, DENSE_WEIGHT), (bm25_rank, BM25_WEIGHT)):
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + weight / (RRF_K + rank)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            Hit(
                clause_id=cid,
                score=score,
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                record=self.records[cid],
            )
            for cid, score in ordered
            if cid in self.records
        ]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
