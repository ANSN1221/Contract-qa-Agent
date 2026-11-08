"""根据 data/raw 下的实际文件生成 document-register.csv。

指标全部来自真实测量（页数、可抽取字符数、低文本页数），不手填。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REGISTER = ROOT / "data" / "document-register.csv"

LOW_TEXT_THRESHOLD = 20  # 单页可抽取字符数低于此值视为无文本页


def load_sources():
    spec = importlib.util.spec_from_file_location(
        "fetch_docs", ROOT / "scripts" / "fetch_public_documents.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_docs"] = mod
    spec.loader.exec_module(mod)
    return mod.SOURCES


def measure(path: Path) -> tuple[int, int, int]:
    doc = pymupdf.open(path)
    try:
        pages = doc.page_count
        chars = 0
        low_text_pages = 0
        for page in doc:
            text = page.get_text("text").strip()
            chars += len(text)
            if len(text) < LOW_TEXT_THRESHOLD:
                low_text_pages += 1
        return pages, chars, low_text_pages
    finally:
        doc.close()


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in load_sources():
        path = RAW / source.filename
        if not path.exists():
            raise SystemExit(f"缺少文件：{path}")
        pages, chars, low_text_pages = measure(path)
        scanned = chars < 200
        if scanned:
            status = "scan_only"
            note = f"纯扫描件，可抽取字符 {chars}；保留为解析失败样本"
        else:
            status = "ready"
            note = f"可抽取 {chars} 字符，无文本页 {low_text_pages}/{pages}"
        rows.append(
            {
                "doc_id": source.doc_id,
                "file_name": source.filename,
                "document_type": source.document_type,
                "title": source.title,
                "source_url": source.url,
                "sanitized": "false",
                "page_count": str(pages),
                "char_count": str(chars),
                "status": status,
                "notes": note,
            }
        )
    return rows


def main() -> int:
    rows = build_rows()
    fieldnames = [
        "doc_id",
        "file_name",
        "document_type",
        "title",
        "source_url",
        "sanitized",
        "page_count",
        "char_count",
        "status",
        "notes",
    ]
    with REGISTER.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if r["status"] == "ready")
    contracts = sum(1 for r in rows if r["document_type"] == "合同")
    tenders = sum(1 for r in rows if r["document_type"] == "招标文件")
    total_pages = sum(int(r["page_count"]) for r in rows)
    print(f"写入 {len(rows)} 行 -> {REGISTER}")
    print(f"合同 {contracts} / 招标文件 {tenders}")
    print(f"可解析 {ready} / 纯扫描 {len(rows) - ready}")
    print(f"合计 {total_pages} 页")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
