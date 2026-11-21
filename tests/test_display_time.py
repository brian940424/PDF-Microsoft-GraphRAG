import unittest

from automotive_graphrag.app import display_time


class DisplayTimeTests(unittest.TestCase):
    def test_utc_timestamp_is_shown_in_taiwan_time(self) -> None:
        self.assertEqual(display_time("2026-10-25T17:30:05+00:00"), "2026-10-26 01:30:05")

    def test_naive_timestamp_is_taken_as_utc(self) -> None:
        self.assertEqual(display_time("2026-10-25T01:00:00", "%Y-%m-%d %H:%M"), "2026-10-25 09:00")

    def test_other_offsets_are_converted(self) -> None:
        self.assertEqual(display_time("2026-10-25T09:00:00+09:00"), "2026-10-25 08:00:00")

    def test_empty_and_unparsable_values_are_kept(self) -> None:
        self.assertEqual(display_time(None), "")
        self.assertEqual(display_time("不是時間"), "不是時間")


if __name__ == "__main__":
    unittest.main()
