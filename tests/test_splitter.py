"""切分器单元测试：全部用合成文本，不依赖真实 PDF。"""

from __future__ import annotations

import unittest

from contract_qa.splitter import (
    MAX_CLAUSE_CHARS,
    choose_plan,
    split_lines,
)
from contract_qa.text import build_lines, classify_heading


def make_pages(*page_texts: str) -> list[tuple[int, str]]:
    return [(index + 1, text) for index, text in enumerate(page_texts)]


def parse(*page_texts: str, doc_id: str = "T001"):
    lines, dropped = build_lines(make_pages(*page_texts))
    plan = choose_plan(lines)
    clauses, merged = split_lines(doc_id, lines, plan)
    return clauses, plan, dropped, merged


class HeadingTest(unittest.TestCase):
    def test_levels_are_ordered_from_chapter_to_item(self):
        cases = {
            "第一章 总则": 2,
            "第一节 范围": 3,
            "第五条 价款": 4,
            "一、项目概况": 5,
            "（一）甲方的义务": 6,
            "1、项目编号：X": 7,
            "1.1 采购内容": 8,
            "（1）本合同协议书": 9,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_heading(text)[0], expected)

    def test_decimal_with_thousand_separator_is_not_a_heading(self):
        self.assertEqual(classify_heading("1.5 万元整（¥15000）")[0], 0)

    def test_toc_line_is_not_a_heading(self):
        self.assertEqual(classify_heading("第一章 招标公告 ......... 2")[0], 0)

    def test_label_keeps_full_width_parentheses(self):
        self.assertEqual(classify_heading("（一）严格遵守法律法规")[1], "（一）")


class TocTest(unittest.TestCase):
    def test_toc_page_is_dropped_entirely(self):
        toc = "\n".join(
            [
                "第一章 招标公告 ............... 2",
                "第二章 投标人须知 ............. 6",
                "第三章 评标办法 ............... 24",
                "第四章 合同 ................... 32",
                "第一章",
            ]
        )
        body = "第一章 招标公告\n本项目已具备招标条件。\n第二章 投标人须知\n投标人应具备相应资质。"
        clauses, _, dropped, _ = parse(toc, body)
        texts = [c.text for c in clauses]
        self.assertFalse(any("....." in t for t in texts))
        self.assertFalse(any(t.strip() == "第一章" for t in texts))
        self.assertGreater(dropped, 0)


class ChapterPathTest(unittest.TestCase):
    def test_bare_heading_is_joined_with_its_title(self):
        body = "第一章\n招标公告\n本项目已具备招标条件，现进行公开招标。"
        clauses, _, _, _ = parse(body)
        self.assertIn("第一章 招标公告", clauses[0].text)

    def test_chapter_path_tracks_parents(self):
        body = "\n".join(
            [
                "第一章 总则",
                "一、项目概况",
                "本项目为数据中心服务器采购，共划分一个标包，交货期为合同签订后三十个日历日内。",
                "二、定义",
                "本文件中甲方指采购人，乙方指中标人，合同文件包括本协议书及其全部附件。",
                "第二章 合同条款",
                "一、付款方式",
                "合同签订后十日内支付合同总价的百分之三十作为预付款，验收合格后支付余款。",
                "二、违约责任",
                "乙方逾期交货的，每逾期一日按合同总价的万分之五向甲方支付违约金。",
            ]
        )
        clauses, plan, _, _ = parse(body)
        self.assertEqual(plan.unit_level, 5)
        paths = [" > ".join(c.chapter_path) for c in clauses if c.clause_no]
        self.assertIn("第一章 总则", paths)
        self.assertIn("第二章 合同条款", paths)


class ArticleModeTest(unittest.TestCase):
    def test_article_document_splits_by_clause(self):
        body = "\n".join(
            [
                "政府采购合同",
                *[f"第{num}条 本条款约定第{num}项义务。" for num in "一二三四五六七八九十"],
            ]
        )
        clauses, plan, _, _ = parse(body)
        self.assertEqual(plan.mode, "article")
        self.assertEqual(plan.unit_level, 4)
        numbered = [c for c in clauses if c.clause_no.startswith("第")]
        self.assertEqual(len(numbered), 10)
        self.assertEqual(numbered[0].clause_no, "第一条")


class QualityTest(unittest.TestCase):
    def test_heading_only_clause_is_merged_into_next(self):
        body = "第一章 招标公告\n一、项目基本情况\n项目编号为 ABC-2025。"
        clauses, _, _, _ = parse(body)
        self.assertEqual(len(clauses), 1)
        self.assertIn("第一章 招标公告", clauses[0].text)
        self.assertIn("项目编号为 ABC-2025", clauses[0].text)

    def test_oversized_clause_is_cut_at_punctuation(self):
        sentence = "甲方应在收到发票后三十日内支付相应款项，逾期按日万分之五计息。"
        body = "第一章 总则\n一、付款\n" + sentence * 60
        clauses, _, _, _ = parse(body)
        self.assertGreater(len(clauses), 1)
        for clause in clauses:
            self.assertLessEqual(clause.char_len, MAX_CLAUSE_CHARS)
        self.assertTrue(all(c.parent_clause_id for c in clauses if c.clause_no == ""))

    def test_clause_ids_are_unique_and_stable(self):
        body = "第一章 总则\n一、A\n内容甲。\n二、B\n内容乙。"
        first, _, _, _ = parse(body)
        second, _, _, _ = parse(body)
        ids = [c.clause_id for c in first]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [c.clause_id for c in second])


class PageMappingTest(unittest.TestCase):
    def test_page_numbers_follow_the_source_page(self):
        clauses, plan, _, _ = parse(
            "第一章 总则\n一、目的\n本合同的目的是明确甲乙双方在本次采购活动中的权利与义务，"
            "保障采购活动依法依规开展，并作为双方履行合同、解决争议的依据。",
            "二、范围\n本合同适用于本项目所需的全部货物供应、运输、安装、调试、验收及售后服务，"
            "以及招标文件中列明的其他相关工作内容，不含另行签订的服务合同。",
            "三、术语\n本文件所称“采购人”指依法开展本次采购活动的国家机关、事业单位或团体组织，"
            "所称“中标人”指经评审委员会推荐并经采购人确认的投标人。",
        )
        self.assertEqual(plan.mode, "outline")
        self.assertEqual([c.page_start for c in clauses], [1, 2, 3])

    def test_chunk_mode_keeps_page_anchor(self):
        body = "\n".join(["本项目相关说明。"] * 200)
        clauses, plan, _, _ = parse(body)
        self.assertEqual(plan.mode, "chunk")
        self.assertTrue(all(c.page_start >= 1 for c in clauses))
        self.assertTrue(all(c.chapter_path == () for c in clauses))


if __name__ == "__main__":
    unittest.main()
