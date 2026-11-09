"""D3 冒烟测试：固定问题集跑一遍问答，记录答案、引用与耗时。

用途有两个：验证链路通；给后续 D5/D6 的改动留一个可比对的基线。
结果写入 docs/d3-smoke.md，数字全部来自本次运行。
"""

from __future__ import annotations

import time
from pathlib import Path

from contract_qa.qa import answer_question

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "d3-smoke.md"

QUESTIONS = [
    ("事实型", "宁国市垃圾中转站项目的合同总价是多少？"),
    ("事实型", "信阳市食品安全抽检项目的预算金额是多少？"),
    ("否定型", "本项目需要缴纳投标保证金吗？"),
    ("跨条款", "逾期交货的违约金怎么算？"),
    ("不可答", "合同里有没有约定用比特币支付？"),
    ("不可答", "甲方项目经理的手机号码是多少？"),
]


def main() -> int:
    rows = []
    total_tokens = 0
    answerable_hits = 0

    for kind, question in QUESTIONS:
        started = time.perf_counter()
        result = answer_question(question)
        elapsed = time.perf_counter() - started
        total_tokens += result.tokens
        answerable_hits += int(result.answerable)
        cites = "、".join(
            f"{c.label}（第{c.page_start}-{c.page_end}页）" for c in result.citations
        ) or "—"
        rows.append(
            {
                "kind": kind,
                "question": question,
                "answer": result.answer,
                "cites": cites,
                "answerable": "是" if result.answerable else "否",
                "latency": elapsed,
                "tokens": result.tokens,
            }
        )

    lines = [
        "# D3 冒烟测试记录",
        "",
        "> 由 `scripts/smoke_ask.py` 生成，指标为本次真实运行结果，未做人工修正。",
        "> 语料：10 份可解析文档、569 条条款；"
        "检索：BM25 + bge-m3 向量，加权 RRF（BM25 1.0 / 向量 0.5）；",
        "> 生成：DeepSeek deepseek-chat，候选 8 条。",
        "",
        "| 类型 | 问题 | 回答 | 引用 | 可答 | 耗时(s) | tokens |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['question']} | {row['answer']} | {row['cites']} | "
            f"{row['answerable']} | {row['latency']:.2f} | {row['tokens']} |"
        )

    lines += [
        "",
        "## 汇总",
        "",
        f"- 问题数：{len(rows)}",
        f"- 判定可答：{answerable_hits}/{len(rows)}",
        f"- 总 tokens：{total_tokens}",
        f"- 平均耗时：{sum(r['latency'] for r in rows) / len(rows):.2f} s",
        "",
        "## 已知问题",
        "",
        "1. **向量召回噪声大**：对「投标保证金」这类术语，bge-m3 会召回「反商业贿赂承诺书」等"
        "语义相近但无用的条款。已用加权 RRF 压制（BM25 权重 1.0、向量 0.5），"
        "根治要靠 D5 的 rerank。",
        "2. **引用精度只到条级**：引用给出的是条款级页码区间，超长条款被切成片段后"
        "仍继承父条款页码，无法精确到片段所在行。",
        "3. **不可答判定靠模型自觉**：目前没有独立的校验节点，模型说不可答就不可答，"
        "需要 D6 的 verify 节点来做二次核对。",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
