import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.semantic_query_cache import (
    hard_guard,
    read_project_mode,
    semantic_config,
    write_project_mode,
)


class SemanticQueryCachePolicyTests(unittest.TestCase):
    def test_critical_hard_negatives_are_rejected(self):
        pairs = [
            ("What is the torque 20 N·m?", "What is the torque 25 N·m?", "numeric_or_unit_mismatch"),
            ("How to diagnose P0300?", "How to diagnose P0301?", "identifier_mismatch"),
            ("Can I disconnect the battery?", "Cannot disconnect the battery?", "polarity_mismatch"),
            ("How to inspect the fuse?", "How to replace the fuse?", "action_mismatch"),
            ("What should I do after that?", "What should I do next?", "context_dependent"),
            ("Check the rattle when the engine is cold", "Check the rattle when the engine is hot", "condition_mismatch"),
            ("Compare fuse 1 and fuse 2", "Compare fuse 1 and fuse 3", "multi_part_or_comparison"),
        ]
        for left, right, expected in pairs:
            with self.subTest(left=left, right=right):
                allowed, reason = hard_guard(left, right)
                self.assertFalse(allowed)
                self.assertEqual(reason, expected)

    def test_semantic_cache_requires_explicit_global_switch_and_calibrated_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(semantic_config())
        with patch.dict(os.environ, {
            "GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE": "true",
            "GRAPHRAG_SEMANTIC_QUERY_CACHE_THRESHOLD": "0.91",
            "GRAPHRAG_SEMANTIC_QUERY_CACHE_MIN_MARGIN": "0.05",
            "GRAPHRAG_SEMANTIC_QUERY_CACHE_TOP_K": "5",
        }, clear=True):
            self.assertEqual(semantic_config(), {"threshold": 0.91, "margin": 0.05, "top_k": 5, "calibrated": True})
        with patch.dict(os.environ, {"GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE": "true"}, clear=True):
            self.assertFalse(semantic_config()["calibrated"])

    def test_project_mode_defaults_off_persists_and_is_independent_of_cache_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            self.assertEqual(read_project_mode(project), "off")
            write_project_mode(project, "shadow")
            self.assertEqual(read_project_mode(project), "shadow")
            self.assertTrue((project / "query_cache_settings.json").is_file())
            self.assertFalse((project / "cache").exists())


if __name__ == "__main__":
    unittest.main()
