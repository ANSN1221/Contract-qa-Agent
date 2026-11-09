"""问答：检索候选条款 -> 让模型只依据候选作答 -> 附上可核对的引用。

D3 只做到「有引用」，D6 才会加独立的 verify 节点做引用校验。
这里先立两条规矩：候选条款外的事实不许出现；模型给出的引用必须能在候选里找到。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from contract_qa.config import Settings, get_settings
from contract_qa.llm import ChatResult, chat
from contract_qa.retrieval import Hit, Retriever, get_retriever

SYSTEM_PROMPT = """你是合同审查助手。只能依据给定的条款片段回答问题。

规则：
1. 如果给出的片段不足以回答，answer 填「未在所给条款中找到依据」，
   answerable 为 false，citations 为空数组。
2. 不许补充片段之外的法律常识、行业惯例或推测。
3. 回答里提到的每个事实，都必须能在 citations 指向的片段中找到原文依据。
4. 若条款明确说明某事项不适用、不收取、无需提供、已免除，请直接说明该结论并引用，
   不要因为「问题问的是金额而条款没给金额」就判定为不可回答。
5. 只输出 JSON，不要任何解释性文字。

输出格式：
{"answer": "……", "citations": ["条款ID"], "answerable": true}
"""


@dataclass
class Citation:
    clause_id: str
    label: str
    page_start: int
    page_end: int
    text: str


@dataclass
class QAResult:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    usage: ChatResult | None = None
    citation_violations: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens if self.usage else 0


def _format_candidates(hits: list[Hit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        record = hit.record
        where = f"{record.doc_id}"
        if record.chapter_path:
            where += f" / {record.chapter_path[-1]}"
        if record.clause_no:
            where += f" / {record.clause_no}"
        where += f" / 第{record.page_start}-{record.page_end}页"
        blocks.append(f"[{index}] 条款ID={record.clause_id}（{where}）\n{record.text}")
    return "\n\n".join(blocks)


def _to_citations(citation_ids: list[str], hits: list[Hit]) -> tuple[list[Citation], list[str]]:
    by_id = {hit.record.clause_id: hit.record for hit in hits}
    citations: list[Citation] = []
    violations: list[str] = []
    for cid in citation_ids:
        record = by_id.get(cid)
        if record is None:
            violations.append(cid)
            continue
        citations.append(
            Citation(
                clause_id=record.clause_id,
                label=record.citation_label,
                page_start=record.page_start,
                page_end=record.page_end,
                text=record.text,
            )
        )
    return citations, violations


def answer_question(
    question: str,
    top_k: int = 8,
    retriever: Retriever | None = None,
    settings: Settings | None = None,
) -> QAResult:
    settings = settings or get_settings()
    retriever = retriever or get_retriever()

    hits = retriever.search(question, top_k=top_k)
    if not hits:
        return QAResult(question=question, answer="未检索到相关条款。", answerable=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"条款片段：\n\n{_format_candidates(hits)}\n\n问题：{question}",
        },
    ]
    result = chat(messages, settings, json_mode=True)

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return QAResult(
            question=question,
            answer=f"模型输出不是合法 JSON：{result.content[:200]}",
            answerable=False,
            hits=hits,
            usage=result,
        )

    citations, violations = _to_citations(payload.get("citations") or [], hits)
    return QAResult(
        question=question,
        answer=str(payload.get("answer", "")).strip(),
        answerable=bool(payload.get("answerable", False)),
        citations=citations,
        hits=hits,
        usage=result,
        citation_violations=violations,
    )
