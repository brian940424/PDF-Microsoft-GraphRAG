import unittest

from automotive_graphrag.local_ranking import (
    HUB_MIN_TEXT_UNITS,
    exact_entity_titles,
    install_local_ranking,
    is_hub,
    rank_text_units,
    reset_local_ranking,
)


class ExactEntityTitleTests(unittest.TestCase):
    def test_titles_with_named_codes_or_whole_titles_in_question(self) -> None:
        titles = ["C1154 PNP SWITCH", "C1155 BR FLUID LEVEL LOW", "AV CONTROL UNIT", "C11540", "ABS", "VDC OFF SWITCH"]

        result = exact_entity_titles("出現 C1154 時 AV control unit 要怎麼檢查？", titles)

        self.assertEqual(result, ["C1154 PNP SWITCH", "AV CONTROL UNIT"])

    def test_short_single_word_titles_are_not_matched_by_text(self) -> None:
        self.assertEqual(exact_entity_titles("ABS 警告燈亮起", ["ABS", "ABS WARNING LAMP"]), [])


class RankTextUnitsTests(unittest.TestCase):
    def test_units_with_named_code_in_heading_come_first(self) -> None:
        texts = {
            "c1155": "BRC > C1155 BR FLUID LEVEL LOW > DTC Logic\n\nsteps",
            "c1154": "BRC > C1154 PNP SWITCH > DTC Logic\n\nsteps",
            "index": "BRC > DTC Index\n\n| C1154 | PNP SW |",
        }
        ordered, report = rank_text_units(
            [("ABS ACTUATOR", ["c1155", "index", "c1154"])], texts, ("C1154",),
            {"c1155": 0.9, "index": 0.8, "c1154": 0.1},
        )

        self.assertEqual(ordered, ["c1154", "index", "c1155"])
        self.assertEqual(report["named_codes"], ["C1154"])

    def test_similarity_orders_units_without_codes(self) -> None:
        texts = {name: name for name in ("a", "b", "c")}
        ordered, _ = rank_text_units([("X", ["a", "b"]), ("Y", ["c"])], texts, (), {"a": 0.1, "b": 0.5, "c": 0.9})
        self.assertEqual(ordered, ["c", "b", "a"])

    def test_hub_entity_brings_only_its_best_unit_forward(self) -> None:
        hub_units = [f"h{index}" for index in range(HUB_MIN_TEXT_UNITS)]
        texts = {unit: unit for unit in [*hub_units, "own"]}
        similarity = {unit: 0.9 - index * 0.01 for index, unit in enumerate(hub_units)} | {"own": 0.2}

        ordered, report = rank_text_units([("BOSE", hub_units), ("AUDIO UNIT", ["own", "h3"])], texts, (), similarity)

        self.assertEqual(ordered[:3], ["h0", "h3", "own"])
        self.assertEqual(report["hub_entities"], ["BOSE"])
        self.assertEqual(report["demoted_units"], HUB_MIN_TEXT_UNITS - 2)

    def test_generic_titles_are_hubs(self) -> None:
        self.assertTrue(is_hub("Diagnosis Procedure", 2))
        self.assertFalse(is_hub("C1154 PNP SWITCH", HUB_MIN_TEXT_UNITS - 1))

    def test_unknown_units_are_skipped(self) -> None:
        ordered, report = rank_text_units([("X", ["a", "missing"])], {"a": "a"})
        self.assertEqual((ordered, report["candidate_units"]), (["a"], 1))


class InstallLocalRankingTests(unittest.TestCase):
    def test_install_and_reset_restore_graphrag_methods(self) -> None:
        from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext

        original = (LocalSearchMixedContext.build_context, LocalSearchMixedContext._build_text_unit_context)
        install_local_ranking({})
        install_local_ranking({})  # a second install replaces, never stacks
        self.assertIsNot(LocalSearchMixedContext.build_context, original[0])
        reset_local_ranking()
        self.assertEqual(
            (LocalSearchMixedContext.build_context, LocalSearchMixedContext._build_text_unit_context), original,
        )


if __name__ == "__main__":
    unittest.main()
