"""检索与问答的单元测试：不联网。"""

from __future__ import annotations

import unittest

from contract_qa.index import tokenize
from contract_qa.qa import _to_citations
from contract_qa.retrieval import RRF_K, Hit


class TokenizeTest(unittest.TestCase):
    def test_chinese_uses_bigrams(self):
        self.assertEqual(tokenize("付款方式"), ["付款", "款方", "方式"])

    def test_single_char_kept(self):
        self.assertEqual(tokenize("合同"), ["合同"])
        self.assertEqual(tokenize("款"), ["款"])

    def test_alnum_kept_whole(self):
        self.assertIn("doc003", tokenize("DOC003"))
        self.assertIn("2025", tokenize("2025年"))

    def test_punctuation_dropped(self):
        self.assertEqual(tokenize("。，、"), [])


class FusionTest(unittest.TestCase):
    """RRF 只看排名，双路都命中的条款必须排在单路命中的前面。"""

    def _fuse(self, dense: list[str], sparse: list[str]) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for ranks in (
            {cid: i for i, cid in enumerate(dense, start=1)},
            {cid: i for i, cid in enumerate(sparse, start=1)},
        ):
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    def test_both_lists_hit_wins(self):
        result = self._fuse(["A", "B", "C"], ["B", "D"])
        self.assertEqual(result[0][0], "B")

    def test_rrf_score_matches_formula(self):
        result = dict(self._fuse(["A"], ["A"]))
        self.assertAlmostEqual(result["A"], 2.0 / (RRF_K + 1))


class CitationTest(unittest.TestCase):
    def test_unknown_citation_is_reported_not_silently_dropped(self):
        hit = Hit(
            clause_id="DOC001-C0001",
            score=0.5,
            dense_rank=1,
            bm25_rank=1,
            record=_fake_record("DOC001-C0001"),
        )
        citations, violations = _to_citations(["DOC001-C0001", "DOC999-C9999"], [hit])
        self.assertEqual(len(citations), 1)
        self.assertEqual(violations, ["DOC999-C9999"])


def _fake_record(clause_id: str):
    from contract_qa.index import ClauseRecord

    return ClauseRecord(
        clause_id=clause_id,
        doc_id="DOC001",
        doc_title="测试合同",
        clause_no="第一条",
        chapter_path=["第一章 总则"],
        page_start=1,
        page_end=2,
        text="测试条款正文",
    )


if __name__ == "__main__":
    unittest.main()
