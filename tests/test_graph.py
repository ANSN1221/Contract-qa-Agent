"""图链路与 JSON 容错的单元测试：不联网。"""

from __future__ import annotations

import unittest

from contract_qa.graph import _parse_json, map_citations
from contract_qa.index import ClauseRecord
from contract_qa.retrieval import Hit


def fake_hit(clause_id: str) -> Hit:
    record = ClauseRecord(
        clause_id=clause_id,
        doc_id="DOC001",
        doc_title="测试合同",
        clause_no="第一条",
        chapter_path=["第一章 总则"],
        page_start=3,
        page_end=4,
        text="测试条款正文",
    )
    return Hit(clause_id=clause_id, score=0.5, dense_rank=1, bm25_rank=2, record=record)


class ParseJsonTest(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(_parse_json('{"a": 1}'), {"a": 1})

    def test_json_wrapped_in_prose(self):
        content = '好的，结果如下：\n{"answer": "是", "citations": []}\n希望有帮助。'
        self.assertEqual(_parse_json(content)["answer"], "是")

    def test_broken_json_returns_none(self):
        self.assertIsNone(_parse_json('{"answer": "是", "citations": [}'))

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_json(""))

    def test_non_object_json_returns_none(self):
        # 数组不是我们要的结构，取不到 answer 字段，按失败处理
        self.assertIsNone(_parse_json("[1, 2, 3]"))


class MapCitationsTest(unittest.TestCase):
    def test_known_citation_mapped(self):
        citations, violations = map_citations(["DOC001-C0001"], [fake_hit("DOC001-C0001")])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["label"], "DOC001 · 第一章 总则 · 第一条")
        self.assertEqual(citations[0]["page_start"], 3)
        self.assertEqual(violations, [])

    def test_unknown_citation_is_a_violation_not_dropped(self):
        citations, violations = map_citations(
            ["DOC001-C0001", "DOC999-C9999"], [fake_hit("DOC001-C0001")]
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(violations, ["DOC999-C9999"])

    def test_empty_citations(self):
        citations, violations = map_citations([], [fake_hit("DOC001-C0001")])
        self.assertEqual((citations, violations), ([], []))


if __name__ == "__main__":
    unittest.main()
