"""检索与问答的单元测试：不联网。"""

from __future__ import annotations

import unittest

from contract_qa.index import tokenize
from contract_qa.retrieval import RRF_K


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


if __name__ == "__main__":
    unittest.main()
