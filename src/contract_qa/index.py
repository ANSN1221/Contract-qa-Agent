"""建索引：Chroma 向量索引 + BM25 关键词索引。

两份索引共享同一套 clause_id，检索阶段才能做融合。
BM25 用中文二元切分，不引入分词依赖——对条款这种短文本，二元词足够，
也不会因为自定义词典缺失把「合同价款」这类词切碎。
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from contract_qa.config import Settings, get_settings
from contract_qa.llm import embed_texts

COLLECTION_NAME = "clauses"
BM25_FILE = "bm25.pkl"
_ID_PATTERN = re.compile(r"[0-9a-zA-Z_.]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中文取二元词，英文数字保留原词。"""
    lowered = text.lower()
    tokens: list[str] = []
    for chunk in _CJK_PATTERN.findall(lowered):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    tokens.extend(_ID_PATTERN.findall(lowered))
    return tokens


@dataclass
class ClauseRecord:
    clause_id: str
    doc_id: str
    doc_title: str
    clause_no: str
    chapter_path: list[str]
    page_start: int
    page_end: int
    text: str

    def metadata(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "clause_no": self.clause_no,
            "chapter_path": " > ".join(self.chapter_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    @property
    def citation_label(self) -> str:
        parts = [self.doc_id]
        if self.chapter_path:
            parts.append(self.chapter_path[-1])
        if self.clause_no:
            parts.append(self.clause_no)
        return " · ".join(parts)


def load_records(settings: Settings | None = None) -> list[ClauseRecord]:
    """从 data/processed 读全部条款，并用 manifest 补上文档标题。"""
    settings = settings or get_settings()
    manifest_path = settings.processed_dir / "manifest.json"
    titles: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for doc in manifest.get("documents", []):
            titles[doc["doc_id"]] = doc.get("title", "")

    records: list[ClauseRecord] = []
    for path in sorted(settings.processed_dir.glob("DOC*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                ClauseRecord(
                    clause_id=row["clause_id"],
                    doc_id=row["doc_id"],
                    doc_title=titles.get(row["doc_id"], ""),
                    clause_no=row["clause_no"],
                    chapter_path=row["chapter_path"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    text=row["text"],
                )
            )
    return records


def build_bm25(records: list[ClauseRecord], settings: Settings | None = None) -> Path:
    """建 BM25 索引并落盘。返回文件路径。"""
    from rank_bm25 import BM25Okapi

    settings = settings or get_settings()
    corpus = [tokenize(record.text) for record in records]
    index = BM25Okapi(corpus)
    target = settings.processed_dir / BM25_FILE
    payload = {"clause_ids": [record.clause_id for record in records], "bm25": index}
    with target.open("wb") as fh:
        pickle.dump(payload, fh)
    return target


def build_chroma(records: list[ClauseRecord], settings: Settings | None = None) -> int:
    """建向量索引。返回写入条数。"""
    import chromadb

    settings = settings or get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    # 重建时先删旧集合，避免残留过期条款
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    vectors = embed_texts([record.text for record in records], settings)
    batch = 128
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        collection.add(
            ids=[record.clause_id for record in chunk],
            embeddings=vectors[start : start + batch],
            documents=[record.text for record in chunk],
            metadatas=[record.metadata() for record in chunk],
        )
    return len(records)
