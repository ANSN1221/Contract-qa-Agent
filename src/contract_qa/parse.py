"""解析编排：PDF -> 行 -> 条款 -> JSONL + manifest。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from contract_qa.models import ParseResult
from contract_qa.splitter import choose_plan, split_lines
from contract_qa.text import build_lines, extract_pages, page_stats

REGISTER_NAME = "document-register.csv"


def parse_document(pdf_path: Path, doc_id: str) -> ParseResult:
    pages, page_count = extract_pages(pdf_path)
    stats = page_stats(pages)
    lines, dropped = build_lines(pages)
    plan = choose_plan(lines)

    result = ParseResult(
        doc_id=doc_id,
        source_file=pdf_path.name,
        page_count=page_count,
        page_stats=stats,
        split_mode=plan.mode,
        dropped_lines=dropped,
    )
    result.clauses, result.merged_fragments = split_lines(doc_id, lines, plan)
    return result


def load_register(register_path: Path) -> list[dict[str, str]]:
    with register_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_parse(project_root: Path, out_dir: Path | None = None) -> dict[str, object]:
    raw_dir = project_root / "data" / "raw"
    register_path = project_root / "data" / REGISTER_NAME
    out_dir = out_dir or project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_register(register_path)
    manifest: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for row in rows:
        pdf_path = raw_dir / row["file_name"]
        if not pdf_path.exists():
            skipped.append({"doc_id": row["doc_id"], "reason": "文件缺失"})
            continue

        result = parse_document(pdf_path, row["doc_id"])
        summary = result.summary()
        summary["document_type"] = row.get("document_type", "")
        summary["title"] = row.get("title", "")
        summary["status"] = row.get("status", "")

        if result.is_scanned:
            summary["clauses_written"] = 0
            summary["note"] = "纯扫描件，无可抽取文本，跳过切分"
            manifest.append(summary)
            continue

        with (out_dir / f"{row['doc_id']}.jsonl").open("w", encoding="utf-8") as fh:
            for clause in result.clauses:
                fh.write(json.dumps(clause.to_dict(), ensure_ascii=False) + "\n")
        summary["clauses_written"] = len(result.clauses)
        manifest.append(summary)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"documents": manifest, "skipped": skipped},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"manifest": manifest, "skipped": skipped, "manifest_path": str(manifest_path)}


def format_report(payload: dict[str, object]) -> str:
    docs = payload["manifest"]  # type: ignore[index]
    lines = [
        "doc_id  类型     页数  字符     条款  模式     页均条款  中位长度",
        "-" * 78,
    ]
    total_clauses = 0
    for doc in docs:  # type: ignore[union-attr]
        if doc["scanned"]:
            lines.append(
                f"{doc['doc_id']}  {doc['document_type']:5s} {doc['page_count']:>4}  "
                f"{doc['total_chars']:>6}  {'扫描':>6}  {'-':6s}  {'-':>7}  {'-':>7}"
            )
            continue
        total_clauses += int(doc["clause_count"])
        per_page = doc["clause_count"] / doc["page_count"] if doc["page_count"] else 0
        lines.append(
            f"{doc['doc_id']}  {doc['document_type']:5s} {doc['page_count']:>4}  "
            f"{doc['total_chars']:>6}  {doc['clause_count']:>6}  {doc['split_mode']:6s}  "
            f"{per_page:>7.2f}  {doc['clause_len_median']:>7}"
        )
    lines.append("-" * 78)
    lines.append(f"条款总数 {total_clauses}")
    return "\n".join(lines)
