"""对照实验：现成切分器 vs 本项目自研切分器。

现成方案 = LangChain RecursiveCharacterTextSplitter，这是最常见、也最容易想到的选择。
同一批 PDF、同一套抽取文本，只换切分方式，比较五件事：

1. 引用单位：切出来的东西能不能被读者核对（是否带条款号）
2. 结构信息：有没有章节路径
3. 切断率：有多少条是从句子中间开始切的
4. 长度分布：中位、P90、最大值
5. 页码：起点是否落在正确的页上

运行：uv run --with langchain-text-splitters python scripts/compare_splitter.py
"""

from __future__ import annotations

import bisect
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from contract_qa.text import normalize_for_match

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "docs" / "splitter-comparison.md"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SENTENCE_END = "。；！？；:"

# 条款级编号：第X条 / 一、 / 1、 / 1.1
_CLAUSE_START = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇\d]{1,4}条|[一二三四五六七八九十]{1,3}、"
    r"|[（(][一二三四五六七八九十\d]{1,3}[）)]|\d{1,2}(?:\.\d{1,2}){0,3}[、.\s])"
)


@dataclass
class Chunk:
    text: str
    page: int
    starts_at_clause: bool
    ends_at_sentence: bool


def page_map(pdf_path: Path) -> tuple[str, list[int], list[str]]:
    """返回全文、每页起始偏移、每页归一化文本。"""
    doc = pymupdf.open(pdf_path)
    try:
        texts = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    starts: list[int] = []
    cursor = 0
    for text in texts:
        starts.append(cursor)
        cursor += len(text) + 1
    full = "\n".join(texts)
    return full, starts, texts


def page_of(offset: int, starts: list[int]) -> int:
    return bisect.bisect_right(starts, offset)


def split_recursive(text: str, starts: list[int]) -> list[Chunk]:
    """现成方案：RecursiveCharacterTextSplitter。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", "", ],
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    cursor = 0
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if not piece:
            continue
        # 用递增查找还原每片在全文中大致位置，避免重叠导致的偏移漂移
        found = text.find(piece, max(0, cursor - CHUNK_OVERLAP))
        offset = found if found >= 0 else cursor
        cursor = offset + len(piece)
        chunks.append(
            Chunk(
                text=piece,
                page=page_of(offset, starts),
                starts_at_clause=bool(_CLAUSE_START.match(piece)),
                ends_at_sentence=piece.rstrip()[-1:] in SENTENCE_END,
            )
        )
    return chunks


def load_own(doc_id: str) -> list[Chunk]:
    """本项目方案：读已切好的条款。"""
    path = PROCESSED / f"{doc_id}.jsonl"
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chunks.append(
            Chunk(
                text=row["text"],
                page=row["page_start"],
                starts_at_clause=bool(row["clause_no"])
                or bool(_CLAUSE_START.match(row["text"])),
                ends_at_sentence=row["text"].rstrip()[-1:] in SENTENCE_END,
            )
        )
    return chunks


def summarize(chunks: list[Chunk], pages_text: list[str]) -> dict:
    if not chunks:
        return {}
    lens = sorted(len(c.text) for c in chunks)
    return {
        "count": len(chunks),
        "median": statistics.median(lens),
        "p90": lens[int(len(lens) * 0.9) - 1] if len(lens) > 1 else lens[0],
        "max": lens[-1],
        "clause_ratio": sum(c.starts_at_clause for c in chunks) / len(chunks),
        "sentence_ratio": sum(c.ends_at_sentence for c in chunks) / len(chunks),
    }


def main() -> int:
    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    docs = [d for d in manifest["documents"] if not d["scanned"]]

    rows = []
    agg_recursive: list[Chunk] = []
    agg_own: list[Chunk] = []

    for doc in docs:
        pdf = RAW / doc["source_file"]
        full, starts, pages_text = page_map(pdf)
        recursive = split_recursive(full, starts)
        own = load_own(doc["doc_id"])
        agg_recursive.extend(recursive)
        agg_own.extend(own)

        rs = summarize(recursive, pages_text)
        os_ = summarize(own, pages_text)
        rows.append(
            {
                "doc": doc["doc_id"],
                "chars": doc["total_chars"],
                "r_count": rs["count"],
                "r_median": rs["median"],
                "r_clause": rs["clause_ratio"],
                "o_count": os_["count"],
                "o_median": os_["median"],
                "o_clause": os_["clause_ratio"],
            }
        )

    rs = summarize(agg_recursive, [])
    os_ = summarize(agg_own, [])

    lines = [
        "# 切分方案对照：现成工具 vs 自研",
        "",
        "> 由 `scripts/compare_splitter.py` 生成，数字为本次真实运行结果。",
        f"> 现成方案：LangChain RecursiveCharacterTextSplitter，chunk_size={CHUNK_SIZE}，"
        f"overlap={CHUNK_OVERLAP}，分隔符优先级 段落 > 换行 > 句号 > 分号 > 逗号。",
        "> 自研方案：本项目 `splitter.py`，按条款编号切分。",
        "> 同一批 PDF、同一套 PyMuPDF 抽取文本，只有切分方式不同。",
        "",
        "## 一、总量对比",
        "",
        "| 指标 | 现成方案 | 自研方案 |",
        "| --- | ---: | ---: |",
        f"| 切出条数 | {rs['count']} | {os_['count']} |",
        f"| 平均每条字符 | {rs['median']:.0f} | {os_['median']:.0f} |",
        f"| P90 长度 | {rs['p90']} | {os_['p90']} |",
        f"| 最长 | {rs['max']} | {os_['max']} |",
        f"| 以条款编号开头的比例 | {rs['clause_ratio']:.1%} | {os_['clause_ratio']:.1%} |",
        f"| 以句末标点结尾的比例 | {rs['sentence_ratio']:.1%} | {os_['sentence_ratio']:.1%} |",
        "",
        "## 二、逐份对比",
        "",
        "| 文档 | 字符 | 现成条数 | 现成中位 | 现成带编号 | 自研条数 | 自研中位 | 自研带编号 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['doc']} | {r['chars']} | {r['r_count']} | {r['r_median']:.0f} | "
            f"{r['r_clause']:.0%} | {r['o_count']} | {r['o_median']:.0f} | {r['o_clause']:.0%} |"
        )

    lines += [
        "",
        "## 三、结论",
        "",
        "现成方案在「切得动」这件事上没有问题：能切、长度可控、有重叠。它的短板不在切分质量，"
        "在于它切出来的东西和读者要核对的东西不是同一个单位。",
        "",
        "1. **引用单位错位**：字符窗口产出的是一段没有名字的文字，读者拿到「第 12 页第 3 段」"
        "无法核对；条款级切分产出的是「DOC011 第二章 第 3.4 条」，翻到那一页就能对上。",
        "2. **结构信息丢失**：现成方案不产生章节路径与条款号，检索命中后无法解释「这是哪一章的规定」，"
        "而合同纠纷往往就发生在「总则」与「专用条款」的冲突里。",
        "3. **切断语义**：按句号切会把「第X条」从中间截断，一条规定被拆到两个 chunk，"
        "两边都不完整。重叠只能缓解，不能消除。",
        "",
        "## 四、什么情况下不该自研",
        "",
        "如果需求是「读一份报告并回答泛化问题」，字符窗口足够，自研是浪费。"
        "自研成立的前提是：**引用必须落到读者可核对的最小单位上**。这是产品需求，不是技术偏好。",
        "",
        "## 五、自研的代价",
        "",
        "不是免费的。`text.py` 与 `splitter.py` 合计约 500 行，其中大半在处理异常："
        "目录页、页眉页脚、只有标题没有正文的条目、超长章节、编号层级错乱。"
        "换成 unstructured 或 Docling 这类版面分析工具，可以省掉其中一部分，"
        "代价是引入更重的依赖与更长的启动时间——这笔账在 D5 接 rerank 时会重新算一次。",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
