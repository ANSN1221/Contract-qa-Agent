"""主链路：route → rewrite → retrieve → generate。

为什么用图而不是函数串联：这条链路有分支（无关问题直接短路）、有条件重试
（JSON 解析失败要重来一次），用 if/else 写下去会变成嵌套金字塔，而且日志打点
散在各处。换成显式状态机之后，每个节点的输入输出都是同一个 state，
加 D6 的 verify 节点只需要多插一条边。

本文件只负责编排，具体检索与调用都在 retrieval / llm 里。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from contract_qa.config import Settings, get_settings
from contract_qa.llm import chat
from contract_qa.retrieval import Hit, Retriever, get_retriever

log = structlog.get_logger("contract_qa.graph")

MAX_QUERIES = 3          # 复合问题最多拆 3 个子查询，再多就超出上下文预算
GENERATE_RETRIES = 1     # malformed 只重试一次，附上错误信息


class QAState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    queries: list[str]
    hits: list[dict]
    answer: str
    answerable: bool
    citation_ids: list[str]
    attempts: int
    error: str
    prompt_tokens: int
    completion_tokens: int
    timings: dict[str, float]


ROUTE_PROMPT = """判断用户问题是否需要查阅合同条款才能回答。

只输出 JSON：
{"route": "contract_qa" 或 "out_of_scope", "reason": "一句话理由"}

route 取值：
- contract_qa：问的是合同/招标文件里的内容（金额、期限、责任、程序、标准等），需要查条款
- out_of_scope：闲聊、与合同无关的通用问题、要求你做条款之外的判断（如"这份合同合不合法"）
"""

REWRITE_PROMPT = """把用户问题改写成适合条款检索的查询。

规则：
1. 去掉口语和疑问词，保留实体与术语。例：「保证金多少钱？」→「投标保证金 金额」
2. 如果问题包含多个独立诉求，拆成多条查询，最多 3 条。
3. 不要臆造原文里没有的术语，也不要回答问题。
4. 只输出 JSON：{"queries": ["查询1", "查询2"]}
"""

GENERATE_PROMPT = """你是合同审查助手。只能依据给定的条款片段回答问题。

规则：
1. 片段不足以回答时，answer 填「未在所给条款中找到依据」，answerable 为 false，citations 为空数组。
2. 不许补充片段之外的法律常识、行业惯例或推测。
3. 回答里每个事实都必须能在 citations 指向的片段中找到原文依据。
4. 若条款明确说明某事项不适用、不收取、无需提供、已免除，直接说明结论并引用，
   不要因为「问题问金额而条款没给金额」就判为不可回答。
5. 只输出 JSON：{"answer": "……", "citations": ["条款ID"], "answerable": true}
"""


def map_citations(citation_ids: list[str], hits: list[Hit]) -> tuple[list[dict], list[str]]:
    """把模型给出的引用 id 映射成可展示的引用；候选之外的 id 记为违例，不静默丢弃。"""
    by_id = {hit.clause_id: hit for hit in hits}
    citations: list[dict] = []
    violations: list[str] = []
    for cid in citation_ids:
        hit = by_id.get(cid)
        if hit is None:
            violations.append(cid)
            continue
        record = hit.record
        citations.append(
            {
                "clause_id": record.clause_id,
                "label": record.citation_label,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "text": record.text,
            }
        )
    return citations, violations


@dataclass
class GraphResult:
    question: str
    answer: str
    answerable: bool
    citations: list[dict] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    route: str = ""
    route_reason: str = ""
    citation_violations: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _parse_json(content: str) -> dict | None:
    """容错解析：先直解，失败则截取第一个 { 到最后一个 } 再试一次。

    模型偶尔会在 JSON 前后加一句解释；严格 json.loads 会直接判失败，
    但这属于可救的格式问题，不是内容问题。
    只接受对象：下游要按字段取值，数组或标量即使合法也用不了。
    """
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None


class QAGraph:
    def __init__(self, settings: Settings | None = None, retriever: Retriever | None = None):
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()
        self.app = self._build()

    # ---------- 节点 ----------

    def _route(self, state: QAState) -> QAState:
        started = time.perf_counter()
        result = chat(
            [
                {"role": "system", "content": ROUTE_PROMPT},
                {"role": "user", "content": state["question"]},
            ],
            self.settings,
            json_mode=True,
        )
        payload = _parse_json(result.content) or {}
        route = payload.get("route", "contract_qa")
        if route not in {"contract_qa", "out_of_scope"}:
            route = "contract_qa"  # 判不出来就放行，宁可多查一次
        timings = dict(state.get("timings", {}))
        timings["route"] = time.perf_counter() - started
        log.info("route", route=route, reason=payload.get("reason", ""))
        return {
            "route": route,
            "route_reason": str(payload.get("reason", "")),
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + result.completion_tokens,
            "timings": timings,
        }

    def _rewrite(self, state: QAState) -> QAState:
        started = time.perf_counter()
        result = chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": state["question"]},
            ],
            self.settings,
            json_mode=True,
        )
        payload = _parse_json(result.content) or {}
        queries = [q.strip() for q in payload.get("queries", []) if str(q).strip()]
        if not queries:
            queries = [state["question"]]  # 改写失败就退回原句，不阻断链路
        timings = dict(state.get("timings", {}))
        timings["rewrite"] = time.perf_counter() - started
        log.info("rewrite", queries=queries)
        return {
            "queries": queries[:MAX_QUERIES],
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + result.completion_tokens,
            "timings": timings,
        }

    def _retrieve(self, state: QAState) -> QAState:
        started = time.perf_counter()
        merged: dict[str, Hit] = {}
        for query in state.get("queries", [state["question"]]):
            for hit in self.retriever.search(query, top_k=8):
                # 同一条款被多个子查询命中时保留更高分的那次
                if hit.clause_id not in merged or hit.score > merged[hit.clause_id].score:
                    merged[hit.clause_id] = hit
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:8]
        timings = dict(state.get("timings", {}))
        timings["retrieve"] = time.perf_counter() - started
        log.info("retrieve", merged=len(merged), kept=len(hits))
        return {
            "hits": [
                {
                    "clause_id": h.clause_id,
                    "score": h.score,
                    "dense_rank": h.dense_rank,
                    "bm25_rank": h.bm25_rank,
                }
                for h in hits
            ],
            "timings": timings,
        }

    def _format_candidates(self, hits: list[Hit]) -> str:
        blocks = []
        for index, hit in enumerate(hits, start=1):
            record = hit.record
            where = record.doc_id
            if record.chapter_path:
                where += f" / {record.chapter_path[-1]}"
            if record.clause_no:
                where += f" / {record.clause_no}"
            where += f" / 第{record.page_start}-{record.page_end}页"
            blocks.append(f"[{index}] 条款ID={record.clause_id}（{where}）\n{record.text}")
        return "\n\n".join(blocks)

    def _generate(self, state: QAState) -> QAState:
        started = time.perf_counter()
        hits = self._hits_from_state(state)
        if not hits:
            return {"answer": "未检索到相关条款。", "answerable": False, "citation_ids": []}

        candidates = self._format_candidates(hits)
        messages = [
            {"role": "system", "content": GENERATE_PROMPT},
            {"role": "user", "content": f"条款片段：\n\n{candidates}\n\n问题：{state['question']}"},
        ]

        attempts = 0
        prompt_tokens = state.get("prompt_tokens", 0)
        completion_tokens = state.get("completion_tokens", 0)
        error = ""
        payload: dict | None = None

        while attempts <= GENERATE_RETRIES:
            attempts += 1
            result = chat(messages, self.settings, json_mode=True)
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            payload = _parse_json(result.content)
            if payload is not None:
                break
            error = f"输出不是合法 JSON：{result.content[:160]}"
            log.warning("generate_malformed", attempt=attempts, preview=result.content[:120])
            if attempts <= GENERATE_RETRIES:
                messages = [*messages, {"role": "user",
                                        "content": "上一条输出不是合法 JSON。"
                                                   "只输出 JSON 对象，不要任何解释文字。"}]

        timings = dict(state.get("timings", {}))
        timings["generate"] = time.perf_counter() - started

        if payload is None:
            return {
                "answer": "模型输出解析失败，已重试一次。",
                "answerable": False,
                "citation_ids": [],
                "attempts": attempts,
                "error": error,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "timings": timings,
            }

        return {
            "answer": str(payload.get("answer", "")).strip(),
            "answerable": bool(payload.get("answerable", False)),
            "citation_ids": [str(c) for c in payload.get("citations", [])],
            "attempts": attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timings": timings,
        }

    def _hits_from_state(self, state: QAState) -> list[Hit]:
        out = []
        for item in state.get("hits", []):
            record = self.retriever.records.get(item["clause_id"])
            if record is None:
                continue
            out.append(
                Hit(
                    clause_id=item["clause_id"],
                    score=item["score"],
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    record=record,
                )
            )
        return out

    # ---------- 图 ----------

    def _build(self):
        graph = StateGraph(QAState)
        graph.add_node("route", self._route)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda state: "rewrite" if state.get("route") == "contract_qa" else END,
            {"rewrite": "rewrite", END: END},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    # ---------- 对外 ----------

    def ask(self, question: str) -> GraphResult:
        state = self.app.invoke({"question": question})
        hits = self._hits_from_state(state)
        citations, violations = map_citations(state.get("citation_ids", []), hits)

        result = GraphResult(
            question=question,
            answer=state.get("answer", ""),
            answerable=bool(state.get("answerable", False)),
            citations=citations,
            hits=hits,
            queries=state.get("queries", []),
            route=state.get("route", ""),
            route_reason=state.get("route_reason", ""),
            citation_violations=violations,
            attempts=state.get("attempts", 0),
            error=state.get("error", ""),
            prompt_tokens=state.get("prompt_tokens", 0),
            completion_tokens=state.get("completion_tokens", 0),
            timings=state.get("timings", {}),
        )
        if result.route == "out_of_scope":
            result.answer = "这个问题不需要查阅合同条款，超出本工具范围。"
        return result


_graph: QAGraph | None = None


def get_graph() -> QAGraph:
    global _graph
    if _graph is None:
        _graph = QAGraph()
    return _graph
