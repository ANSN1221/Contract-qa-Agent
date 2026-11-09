"""命令行入口：check 检查环境，parse 跑解析与切分。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "docs",
    "src/contract_qa",
    "tests",
)
REGISTER_PATH = PROJECT_ROOT / "data" / "document-register.csv"


def discover_documents(raw_dir: Path) -> list[Path]:
    """Return local PDF documents in stable order."""
    if not raw_dir.exists():
        return []
    return sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def read_register() -> list[dict[str, str]]:
    if not REGISTER_PATH.exists():
        return []
    with REGISTER_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_check() -> int:
    """Check the scaffold; missing source documents are reported, not treated as an error."""
    print("合同条款智能问答 Agent · 环境检查")
    print(f"Python: {sys.version.split()[0]}")

    missing = [name for name in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / name).is_dir()]
    if missing:
        print("失败：缺少目录 " + ", ".join(missing))
        return 1

    documents = discover_documents(PROJECT_ROOT / "data/raw")
    rows = read_register()
    ready = sum(1 for row in rows if row.get("status") == "ready")
    scanned = sum(1 for row in rows if row.get("status") == "scan_only")

    print("目录结构：通过")
    print(f"data/raw PDF：{len(documents)} 份")
    print(f"登记表：{len(rows)} 行（可解析 {ready} / 纯扫描 {scanned}）")
    print("密钥配置：使用 .env，禁止提交真实密钥")

    if not documents:
        print("待办：将脱敏 PDF 放入 data/raw，并更新 data/document-register.csv")
    return 0


def run_parse(out_dir: Path | None) -> int:
    try:
        from contract_qa.parse import format_report
        from contract_qa.parse import run_parse as do_parse
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底提示
        print(f"失败：缺少依赖（{exc}）。请先安装 pymupdf：pip install pymupdf")
        return 1

    payload = do_parse(PROJECT_ROOT, out_dir)
    print(format_report(payload))
    print(f"manifest: {payload['manifest_path']}")
    if payload["skipped"]:
        print(f"跳过 {len(payload['skipped'])} 份：" + ", ".join(
            f"{item['doc_id']}({item['reason']})" for item in payload["skipped"]  # type: ignore[index]
        ))
    return 0


def run_index(rebuild: bool) -> int:
    try:
        from contract_qa.index import build_bm25, build_chroma, load_records
    except ImportError as exc:  # pragma: no cover
        print(f"失败：缺少依赖（{exc}）")
        return 1

    records = load_records()
    if not records:
        print("失败：data/processed 下没有条款，请先运行 contract-qa parse")
        return 1
    print(f"条款 {len(records)} 条")

    bm25_path = build_bm25(records)
    print(f"BM25 索引：{bm25_path}")

    from contract_qa.config import get_settings

    settings = get_settings()
    if not settings.has_embedding:
        print("跳过向量索引：.env 缺少 EMBEDDING_API_KEY（BM25 索引已可用）")
        return 0

    count = build_chroma(records, settings)
    print(f"向量索引：{count} 条 -> {settings.chroma_path}")
    return 0


def run_ask(question: str, top_k: int, show_hits: bool) -> int:
    try:
        from contract_qa.qa import answer_question
    except ImportError as exc:  # pragma: no cover
        print(f"失败：缺少依赖（{exc}）")
        return 1

    try:
        result = answer_question(question, top_k=top_k)
    except FileNotFoundError as exc:
        print(f"失败：{exc}")
        return 1
    except Exception as exc:  # 网络、配额、密钥问题统一提示，不吞掉原因
        print(f"失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"\n问题：{result.question}")
    print(f"回答：{result.answer}")
    print(f"可答：{'是' if result.answerable else '否'}")

    if result.citations:
        print("\n引用：")
        for index, cite in enumerate(result.citations, start=1):
            print(f"  [{index}] {cite.label} · 第{cite.page_start}-{cite.page_end}页")
            snippet = cite.text[:100].replace("\n", " ")
            print(f"      {snippet}……")
    if result.citation_violations:
        print(f"\n警告：模型给出了候选之外的引用 {result.citation_violations}")

    if show_hits:
        print("\n召回明细：")
        for rank, hit in enumerate(result.hits, start=1):
            r = hit.record
            print(
                f"  {rank}. {r.clause_id} rrf={hit.score:.4f} "
                f"dense={hit.dense_rank} bm25={hit.bm25_rank} — {r.clause_id} {r.clause_no}"
            )
    if result.usage:
        print(
            f"\ntokens：prompt {result.usage.prompt_tokens} / "
            f"completion {result.usage.completion_tokens}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合同条款智能问答 Agent")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="检查项目结构与数据准备状态")
    parse_cmd = subparsers.add_parser("parse", help="解析 PDF 并切分条款")
    parse_cmd.add_argument("--out", type=Path, default=None, help="输出目录，默认 data/processed")
    index_cmd = subparsers.add_parser("index", help="建 BM25 与向量索引")
    index_cmd.add_argument("--rebuild", action="store_true", help="强制重建（默认也重建）")
    ask_cmd = subparsers.add_parser("ask", help="提问并返回答案与引用")
    ask_cmd.add_argument("question", help="问题文本")
    ask_cmd.add_argument("--top-k", type=int, default=8, help="送入模型的条款数，默认 8")
    ask_cmd.add_argument("--show-hits", action="store_true", help="打印召回明细")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in (None, "check"):
        raise SystemExit(run_check())
    if args.command == "parse":
        raise SystemExit(run_parse(args.out))
    if args.command == "index":
        raise SystemExit(run_index(args.rebuild))
    if args.command == "ask":
        raise SystemExit(run_ask(args.question, args.top_k, args.show_hits))
    parser.error(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
