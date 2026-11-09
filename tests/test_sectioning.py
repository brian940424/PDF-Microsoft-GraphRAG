import unittest

from automotive_graphrag.sectioning import (
    looks_like_section_heading,
    remove_leading_printed_page_number,
    split_into_sections,
)


class SectioningTests(unittest.TestCase):
    def test_splits_manual_headings_but_keeps_steps_and_notes_attached(self) -> None:
        sections, last_heading = split_into_sections(
            "變更螢幕鎖定方式\n1. 開啟設定\n• 也可以使用指紋\n強制重啟\n同時按住側鍵"
        )

        self.assertEqual([heading for heading, _ in sections], ["變更螢幕鎖定方式", "強制重啟"])
        self.assertIn("• 也可以使用指紋", sections[0][1])
        self.assertEqual(last_heading, "強制重啟")

    def test_page_continuation_inherits_previous_section(self) -> None:
        sections, heading = split_into_sections("步驟 3：完成註冊", "註冊指紋")

        self.assertEqual(sections, [("註冊指紋", "步驟 3：完成註冊")])
        self.assertEqual(heading, "註冊指紋")

    def test_ambiguous_short_lines_and_bullets_are_not_headings(self) -> None:
        self.assertFalse(looks_like_section_heading("• 僅適用於特定機型"))
        self.assertFalse(looks_like_section_heading("輕觸完成後即可使用。"))
        self.assertTrue(looks_like_section_heading("第 2 節 強制重啟"))

    def test_removes_an_isolated_folio_near_the_start(self) -> None:
        text = remove_leading_printed_page_number("設定\n177\n變更螢幕鎖定方式\n步驟")

        self.assertEqual(text, "設定\n\n變更螢幕鎖定方式\n步驟")

    def test_keeps_a_numbered_instruction(self) -> None:
        text = remove_leading_printed_page_number("1\n完成註冊")

        self.assertEqual(text, "1\n完成註冊")


if __name__ == "__main__":
    unittest.main()
