"""条款切分。

三种模式，按文档自身的编号体系自动选择，绝不混用：

- article：「第X条」数量足够 -> 以条为引用单元，章/节作为 chapter_path
- outline: 靠 一、/1./1.1 分层 -> 逐层试切，取中位长度最接近目标的层级
- chunk  : 完全没有稳定编号 -> 按字符窗口切，强制带页码，标记为弱引用

任何一条被切出来的 Clause 都必须能回答：属于哪份文件、第几页、原文是什么。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from contract_qa.models import Clause, Line
from contract_qa.text import classify_heading, join_lines

ARTICLE_MODE_MIN_UNITS = 8
CANDIDATE_LEVELS = (4, 5, 6, 7, 8)
MIN_UNITS_FOR_CANDIDATE = 3
GOOD_MEDIAN_CHARS = 60
ACCEPTABLE_MEDIAN_CHARS = 40
COARSE_MEDIAN_CHARS = 700
TARGET_CLAUSE_CHARS = 350
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MIN_CLAUSE_CHARS = 30
HEADING_ONLY_MAX_CHARS = 30
MAX_CLAUSE_CHARS = 1200
BREAK_CHARS = "。；！？;!?；:"
BREAK_SEARCH_WINDOW = 240


@dataclass(frozen=True)
class SplitPlan:
    mode: str
    unit_level: int
    reason: str
    probe_median: int = 0
    probe_count: int = 0


def choose_plan(lines: list[Line]) -> SplitPlan:
    """选引用单元层级：从最粗的编号层往下探，直到切出来的东西有实际内容。

    不用「中位长度最接近某个目标值」做择优，因为招标文件里的表格行、
    短条目本来就是几十个字符，用长度优化会把好文档误判成坏文档。
    这里只回答一个问题：这一层是「标题层」还是「内容层」。
    """
    counts: dict[int, int] = {}
    for line in lines:
        if line.is_heading:
            counts[line.heading_level] = counts.get(line.heading_level, 0) + 1

    if counts.get(4, 0) >= ARTICLE_MODE_MIN_UNITS:
        plan = SplitPlan("article", 4, f"第X条 {counts[4]} 处，按条引用")
        probes, _ = _split_by_heading("probe", lines, plan)
        return _with_probe(plan, probes)

    available = [lv for lv in CANDIDATE_LEVELS if counts.get(lv, 0) >= MIN_UNITS_FOR_CANDIDATE]
    if not available:
        return SplitPlan("chunk", 0, f"无可用编号（{_counts_text(counts)}），退化为定长窗口")

    measured: list[tuple[int, SplitPlan, int]] = []
    for level in available:
        plan = SplitPlan("outline", level, f"候选层级 {level}")
        probes, _ = _split_by_heading("probe", lines, plan)
        median = _median([c.char_len for c in probes])
        measured.append((level, _with_probe(plan, probes), median))

    # 第一优先：最粗的一层，只要它「有内容」且「不算太粗」。
    for level, plan, median in measured:
        if GOOD_MEDIAN_CHARS <= median <= COARSE_MEDIAN_CHARS:
            return _selected(level, plan, median, counts, f"中位 {median} 字符落在 60–700 区间")

    # 第二优先：都偏粗时，选最接近目标长度的一层。
    in_range = [item for item in measured if item[2] >= ACCEPTABLE_MEDIAN_CHARS]
    if in_range:
        level, plan, median = min(in_range, key=lambda item: abs(item[2] - TARGET_CLAUSE_CHARS))
        return _selected(level, plan, median, counts, f"无 60–700 区间层级，取最接近 350 字符者")

    level, plan, median = measured[0]
    return SplitPlan(
        mode="chunk",
        unit_level=0,
        reason=f"{_counts_text(counts)}；最粗层级 {level} 中位仅 {median} 字符，改走定长窗口",
    )


def _selected(
    level: int, plan: SplitPlan, median: int, counts: dict[int, int], why: str
) -> SplitPlan:
    return SplitPlan(
        mode=plan.mode,
        unit_level=level,
        reason=(
            f"{_counts_text(counts)}；层级 {level} 试切 {plan.probe_count} 条、{why}"
        ),
        probe_median=median,
        probe_count=plan.probe_count,
    )


def _with_probe(plan: SplitPlan, clauses: list[Clause]) -> SplitPlan:
    return SplitPlan(
        mode=plan.mode,
        unit_level=plan.unit_level,
        reason=plan.reason,
        probe_median=_median([c.char_len for c in clauses]),
        probe_count=len(clauses),
    )


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _counts_text(counts: dict[int, int]) -> str:
    if not counts:
        return "无编号"
    return "、".join(f"L{level}:{counts[level]}" for level in sorted(counts))


def split_lines(doc_id: str, lines: list[Line], plan: SplitPlan) -> tuple[list[Clause], int]:
    if plan.mode == "chunk":
        return _finalize(doc_id, _split_by_chunk(doc_id, lines))
    return _split_by_heading(doc_id, lines, plan)


def _split_by_heading(doc_id: str, lines: list[Line], plan: SplitPlan) -> tuple[list[Clause], int]:
    clauses: list[Clause] = []
    stack: dict[int, str] = {}
    buffer: list[Line] = []
    current_no = ""

    def flush() -> None:
        nonlocal buffer, current_no
        if not buffer:
            return
        text = join_lines([line.text for line in buffer])
        if text:
            path = tuple(stack[level] for level in sorted(stack) if level < plan.unit_level)
            clauses.append(
                Clause(
                    doc_id=doc_id,
                    clause_id="",  # 编号在最后统一分配
                    chapter_path=path,
                    clause_no=current_no,
                    text=text,
                    page_start=buffer[0].page_no,
                    page_end=buffer[-1].page_no,
                    char_start=buffer[0].offset,
                    char_end=buffer[-1].offset + len(buffer[-1].text),
                    split_mode=plan.mode,
                )
            )
        buffer = []

    for line in lines:
        level = line.heading_level if line.is_heading else 0
        if level and level < plan.unit_level:
            flush()
            stack[level] = line.text
            for deeper in [k for k in stack if k > level]:
                del stack[deeper]
            current_no = ""
            buffer.append(line)
            continue
        if level == plan.unit_level:
            flush()
            current_no = line.heading_label or line.text[:12]
            buffer.append(line)
            continue
        buffer.append(line)

    flush()
    return _finalize(doc_id, clauses)


def _split_by_chunk(doc_id: str, lines: list[Line]) -> list[Clause]:
    clauses: list[Clause] = []
    buffer: list[Line] = []
    length = 0
    stack: dict[int, str] = {}

    def flush(keep_tail: bool) -> None:
        nonlocal buffer, length
        if not buffer:
            return
        text = join_lines([line.text for line in buffer])
        if text:
            path = tuple(stack[level] for level in sorted(stack))
            clauses.append(
                Clause(
                    doc_id=doc_id,
                    clause_id="",
                    chapter_path=path,
                    clause_no="",
                    text=text,
                    page_start=buffer[0].page_no,
                    page_end=buffer[-1].page_no,
                    char_start=buffer[0].offset,
                    char_end=buffer[-1].offset + len(buffer[-1].text),
                    split_mode="chunk",
                )
            )
        if not keep_tail:
            buffer = []
            length = 0
            return
        # 回溯保留一段重叠，避免条款被窗口边界切断后两边都检索不到
        tail: list[Line] = []
        tail_len = 0
        for line in reversed(buffer):
            tail.insert(0, line)
            tail_len += len(line.text)
            if tail_len >= CHUNK_OVERLAP:
                break
        buffer = tail
        length = tail_len

    for line in lines:
        if line.is_heading and line.heading_level <= 3:
            stack[line.heading_level] = line.text
            for deeper in [k for k in stack if k > line.heading_level]:
                del stack[deeper]
        buffer.append(line)
        length += len(line.text)
        if length >= CHUNK_SIZE:
            flush(keep_tail=True)

    flush(keep_tail=False)
    return clauses


def _finalize(doc_id: str, clauses: list[Clause]) -> tuple[list[Clause], int]:
    """合并过短碎片（页码残留、孤立标点），再统一编号。"""
    merged = 0
    kept: list[Clause] = []
    for clause in clauses:
        is_heading_only = classify_heading(clause.text[:60])[0] > 0
        if (
            kept
            and clause.char_len < MIN_CLAUSE_CHARS
            and not clause.clause_no
            and not is_heading_only
        ):
            prev = kept[-1]
            kept[-1] = Clause(
                doc_id=prev.doc_id,
                clause_id=prev.clause_id,
                chapter_path=prev.chapter_path,
                clause_no=prev.clause_no,
                text=f"{prev.text}{clause.text}",
                page_start=prev.page_start,
                page_end=clause.page_end,
                char_start=prev.char_start,
                char_end=clause.char_end,
                split_mode=prev.split_mode,
            )
            merged += 1
            continue
        kept.append(clause)

    kept = _merge_heading_only(kept)

    numbered = [
        Clause(
            doc_id=c.doc_id,
            clause_id=f"{doc_id}-C{index:04d}",
            chapter_path=c.chapter_path,
            clause_no=c.clause_no,
            text=c.text,
            page_start=c.page_start,
            page_end=c.page_end,
            char_start=c.char_start,
            char_end=c.char_end,
            split_mode=c.split_mode,
        )
        for index, c in enumerate(kept, start=1)
    ]
    return _split_oversized(numbered), merged


def _is_heading_only(clause: Clause) -> bool:
    """只有标题、没有正文的条款：如「第一章 招标公告」单独成条。

    这类条目单独送去向量化只会浪费一条索引，且检索命中后无法回答问题，
    所以要并到相邻条款里，同时保留标题文字。
    """
    text = clause.text.strip()
    if len(text) >= HEADING_ONLY_MAX_CHARS:
        return False
    if any(ch in text for ch in BREAK_CHARS):
        return False
    return classify_heading(text)[0] > 0


def _merge_heading_only(clauses: list[Clause]) -> list[Clause]:
    """标题型条款优先并给下一条（标题本就属于它），末尾的并回上一条。"""
    out: list[Clause] = []
    carry: Clause | None = None
    for clause in clauses:
        if carry is not None:
            clause = replace(
                clause,
                text=f"{carry.text}\n{clause.text}",
                page_start=carry.page_start,
                char_start=carry.char_start,
                clause_no=clause.clause_no or carry.clause_no,
            )
            carry = None
        if _is_heading_only(clause):
            carry = clause
            continue
        out.append(clause)
    if carry is not None:
        if out:
            out[-1] = replace(
                out[-1],
                text=f"{out[-1].text}\n{carry.text}",
                page_end=carry.page_end,
                char_end=carry.char_end,
            )
        else:
            out.append(carry)
    return out


def _split_oversized(clauses: list[Clause]) -> list[Clause]:
    """超长条款按标点切开，避免单条塞进一整章。

    切出来的片段沿用它所属条款的编号与页码范围，clause_id 加序号后缀，
    并在 parent_clause_id 里留下原始 id，保证引用可回溯、也不会伪造页码。
    """
    out: list[Clause] = []
    for clause in clauses:
        if clause.char_len <= MAX_CLAUSE_CHARS:
            out.append(clause)
            continue
        for part_index, piece in enumerate(_cut_text(clause.text), start=1):
            out.append(
                Clause(
                    doc_id=clause.doc_id,
                    clause_id=f"{clause.clause_id}-{part_index}",
                    chapter_path=clause.chapter_path,
                    clause_no=clause.clause_no,
                    text=piece,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    char_start=clause.char_start,
                    char_end=clause.char_end,
                    split_mode=clause.split_mode,
                    parent_clause_id=clause.clause_id,
                )
            )
    return out


def _cut_text(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CLAUSE_CHARS, len(text))
        if end < len(text):
            window_start = max(start + MAX_CLAUSE_CHARS - BREAK_SEARCH_WINDOW, start + 1)
            cut = -1
            for pos in range(end - 1, window_start - 1, -1):
                if text[pos] in BREAK_CHARS:
                    cut = pos + 1
                    break
            if cut > start:
                end = cut
        pieces.append(text[start:end].strip())
        start = end
    return [p for p in pieces if p]
