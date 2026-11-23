import unittest

import numpy as np

from automotive_graphrag.global_sources import (
    add_source_citations,
    cited_statements,
    ground_citations,
)

VECTORS = {
    "tu-brake": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "tu-pedal": np.array([0.8, 0.6, 0.0], dtype=np.float32),
    "tu-audio": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}
TOPIC = {"brake": [1.0, 0.0, 0.0], "pedal": [0.0, 1.0, 0.0], "audio": [0.0, 0.0, 1.0]}


def fake_embed(texts):
    result = []
    for text in texts:
        vector = np.zeros(3)
        for word, topic in TOPIC.items():
            if word in text.lower():
                vector += topic
        result.append(vector.tolist())
    return result


class GlobalSourceTests(unittest.TestCase):
    def test_cited_text_is_everything_after_the_previous_citation(self) -> None:
        answer = (
            "## Brakes\n1. Bleed the brake lines.\n2. Check the pedal height. [Data: Reports (3, 4, +more)]\n"
            "Audio unit resets. [Data: Reports (9)]"
        )

        statements = cited_statements(answer)

        self.assertEqual([item[3] for item in statements], [("3", "4"), ("9",)])
        self.assertIn("Bleed the brake lines", statements[0][2])
        self.assertIn("Check the pedal height", statements[0][2])
        self.assertEqual(statements[1][2], "Audio unit resets.")

    def test_statements_pick_closest_units_of_cited_communities_only(self) -> None:
        answer = "Bleed the brake lines first. Then check the pedal height. [Data: Reports (1)] Audio. [Data: Reports (2)]"
        communities = {"1": ["tu-brake", "tu-pedal", "tu-audio"], "2": ["missing"]}

        grounded = ground_citations(answer, communities, VECTORS, fake_embed, per_statement=1)

        self.assertEqual(len(grounded), 1)
        self.assertEqual(grounded[0].report_ids, ("1",))
        # Whole text (brake + pedal) → pedal; the brake sentence adds its own best passage; audio never.
        self.assertEqual(grounded[0].text_unit_ids, ("tu-pedal", "tu-brake"))

    def test_source_ids_are_added_inside_the_citation(self) -> None:
        answer = "Bleed the brake lines. [Data: Reports (1, +more)] Done."
        grounded = ground_citations(answer, {"1": ["tu-brake", "tu-audio"]}, VECTORS, fake_embed, per_statement=1)

        result = add_source_citations(answer, grounded, {"tu-brake": "17"})

        self.assertEqual(result, "Bleed the brake lines. [Data: Reports (1, +more); Sources (17)] Done.")


if __name__ == "__main__":
    unittest.main()
