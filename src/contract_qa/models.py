"""解析链路的数据模型。

设计原则：所有 citation 需要的字段必须能在切分阶段确定，不允许事后补算。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    """清洗后的单行文本，保留它来自哪一页、在全文中的偏移量。"""

    text: str
    page_no: int
    offset: int
    is_heading: bool = False
    heading_level: int = 0
    heading_label: str = ""


@dataclass(frozen=True)
class Clause:
    """一个可被引用的最小条款单元。"""

    doc_id: str
    clause_id: str
    chapter_path: tuple[str, ...]
    clause_no: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    split_mode: str
    parent_clause_id: str = ""

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "clause_id": self.clause_id,
            "parent_clause_id": self.parent_clause_id,
            "chapter_path": list(self.chapter_path),
            "clause_no": self.clause_no,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_len": self.char_len,
            "split_mode": self.split_mode,
        }


@dataclass
class PageStat:
    page_no: int
    chars: int
    low_text: bool


@dataclass
class ParseResult:
    doc_id: str
    source_file: str
    page_count: int
    page_stats: list[PageStat] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    split_mode: str = ""
    dropped_lines: int = 0
    merged_fragments: int = 0

    @property
    def total_chars(self) -> int:
        return sum(p.chars for p in self.page_stats)

    @property
    def low_text_pages(self) -> int:
        return sum(1 for p in self.page_stats if p.low_text)

    @property
    def is_scanned(self) -> bool:
        return self.total_chars < 200

    def summary(self) -> dict[str, object]:
        lens = sorted(c.char_len for c in self.clauses)
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "low_text_pages": self.low_text_pages,
            "scanned": self.is_scanned,
            "split_mode": self.split_mode,
            "clause_count": len(self.clauses),
            "dropped_lines": self.dropped_lines,
            "merged_fragments": self.merged_fragments,
            "clause_len_min": lens[0] if lens else 0,
            "clause_len_median": lens[len(lens) // 2] if lens else 0,
            "clause_len_max": lens[-1] if lens else 0,
        }
