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

    if sys.version_info < (3, 11):
        print("失败：需要 Python 3.11 或更高版本。")
        return 1

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
        from contract_qa.parse import format_report, run_parse as do_parse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合同条款智能问答 Agent")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="检查项目结构与数据准备状态")
    parse_cmd = subparsers.add_parser("parse", help="解析 PDF 并切分条款")
    parse_cmd.add_argument("--out", type=Path, default=None, help="输出目录，默认 data/processed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in (None, "check"):
        raise SystemExit(run_check())
    if args.command == "parse":
        raise SystemExit(run_parse(args.out))
    parser.error(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
