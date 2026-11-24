import unittest

from automotive_graphrag.chunk_context import (
    CAPS,
    SPLIT_MIN_CHARS,
    TITLE,
    contextualize_records,
    heading_lines,
)


def _record(page: int, text: str, document: str = "AV.pdf") -> dict:
    return {
        "id": f"P-AV-p{page:04d}-b01", "chunk_id": f"P-AV-p{page:04d}-b01", "title": f"{document} - Page {page}",
        "text": text, "document_id": document, "page": page, "block_id": "b01", "section_path": "AV",
    }


TABLE_HEADER = "| Terminal | Signal name | Value |\n|---|---|---|"


def _table_rows(start: int, count: int) -> str:
    return "\n".join(
        f"| {number} (G) | Signal {number} with a reasonably long description of the condition | Battery voltage |"
        for number in range(start, start + count)
    )


class HeadingLinesTests(unittest.TestCase):
    def test_classifies_manual_headings(self) -> None:
        lines = [
            "ECU DIAGNOSIS INFORMATION", "AV CONTROL UNIT", "Reference Value", "INFOID:0000000008542262",
            "TERMINAL LAYOUT", "Terminal layout diagram.", "AWMIA1310ZZ", "NOTE:",
            "GPS (Global Positioning System)", "BRAKE MASTER CYLINDER ..........26",
            "| 25 (G) | Ground | Reverse signal |",
        ]
        found = heading_lines(lines)
        self.assertEqual(found, {0: CAPS, 1: CAPS, 2: TITLE, 4: CAPS, 8: CAPS})

    def test_short_title_directly_under_caps_heading_without_infoid(self) -> None:
        found = heading_lines(["BCM (BODY CONTROL MODULE)", "Reference Value", "NOTE:", "Body text follows here."])
        self.assertEqual(found, {0: CAPS, 1: TITLE})


class ContextualizeRecordsTests(unittest.TestCase):
    def test_continuation_table_page_inherits_component_heading(self) -> None:
        records = [
            _record(216, "ECU DIAGNOSIS INFORMATION\nAV CONTROL UNIT\nReference Value\nINFOID:1\n"
                         "TERMINAL LAYOUT\nTerminal layout diagram.\nPHYSICAL VALUES\n" + TABLE_HEADER + "\n" + _table_rows(1, 2)),
            _record(217, TABLE_HEADER + "\n" + _table_rows(25, 2)),
        ]

        result = contextualize_records(records)

        self.assertEqual(
            result[1]["text"].splitlines()[0],
            "AV > ECU DIAGNOSIS INFORMATION > AV CONTROL UNIT > Reference Value > PHYSICAL VALUES",
        )
        self.assertEqual(result[1]["id"], "P-AV-p0217-b01")
        self.assertIn("| 25 (G) |", result[1]["text"])

    def test_heading_after_body_replaces_only_the_sub_topic(self) -> None:
        records = [
            _record(210, "NAVIGATION SYSTEM FUNCTION\nDescription\nINFOID:2\nMAP-MATCHING\n" + "Map body. " * 60),
            _record(211, "GPS (Global Positioning System)\n" + "Satellite body. " * 40),
        ]

        result = contextualize_records(records)

        self.assertEqual(
            result[1]["text"].splitlines()[0],
            "AV > NAVIGATION SYSTEM FUNCTION > Description > GPS (Global Positioning System)",
        )

    def test_long_page_splits_at_headings_and_keeps_page_identity(self) -> None:
        text = (
            "NAVIGATION SYSTEM FUNCTION\nDescription\nINFOID:3\nMAP-MATCHING\n" + "Map matching body. " * 80
            + "\nGPS (Global Positioning System)\n" + "Four or more satellites. " * 80
        )
        self.assertGreater(len(text), SPLIT_MIN_CHARS)

        result = contextualize_records([_record(211, text)])

        self.assertEqual([item["id"] for item in result], ["P-AV-p0211-b01-c01", "P-AV-p0211-b01-c02"])
        self.assertTrue(all(item["chunk_id"] == "P-AV-p0211-b01" and item["page"] == 211 for item in result))
        self.assertTrue(result[1]["text"].startswith("AV > NAVIGATION SYSTEM FUNCTION > Description > GPS"))
        self.assertIn("Four or more satellites.", result[1]["text"])
        self.assertNotIn("Four or more satellites.", result[0]["text"])

    def test_long_table_splits_into_row_groups_that_repeat_the_header(self) -> None:
        rows = _table_rows(100, 30) + "\n|  |  | 0 V |"
        result = contextualize_records([
            _record(30, "BCM (BODY CONTROL MODULE)\nReference Value\nPHYSICAL VALUES\n" + TABLE_HEADER + "\n" + rows, "BCS.pdf"),
        ])

        self.assertGreater(len(result), 1)
        for item in result:
            self.assertIn("| Terminal | Signal name | Value |", item["text"])
            self.assertTrue(item["text"].startswith("AV > BCM (BODY CONTROL MODULE) > Reference Value > PHYSICAL VALUES"))
        joined = "\n".join(item["text"] for item in result)
        for number in range(100, 130):
            self.assertEqual(joined.count(f"| {number} (G) |"), 1)
        # The empty-first-cell continuation row stays with terminal 129.
        self.assertIn("| 129 (G) |", result[-1]["text"])
        self.assertTrue(result[-1]["text"].rstrip().endswith("|  |  | 0 V |"))

    def test_record_without_new_heading_context_is_unchanged(self) -> None:
        record = {**_record(5, "設定\n開啟功能。"), "section_path": "Manual > 設定"}

        result = contextualize_records([record])

        self.assertEqual(result[0]["text"], "設定\n開啟功能。")

    def test_heading_trail_does_not_leak_between_documents(self) -> None:
        result = contextualize_records([
            _record(1, "AV CONTROL UNIT\nReference Value\nINFOID:4\nBody."),
            _record(1, "Plain first page of another manual.", "BR.pdf"),
        ])

        self.assertEqual(result[1]["text"], "Plain first page of another manual.")

    def test_dtc_detection_table_restarts_trail_when_its_heading_was_lost(self) -> None:
        # BRC p.88: extraction lost "C1155 BR FLUID LEVEL LOW", so the trail kept C1154 (F-69).
        detection = (
            "DTC DETECTION LOGIC\n"
            "| DTC | Display item | Malfunction detected condition | Possible cause |\n|---|---|---|---|\n"
            "| C1155 | BR FLUID LEVEL LOW | Brake fluid level is low. | • Brake fluid level switch |\n"
            "Check the self-diagnosis results."
        )
        result = contextualize_records([
            _record(86, "C1154 PNP SWITCH\nDescription\nINFOID:6\nThe transmission range switch signal.\n"
                        "DTC DETECTION LOGIC\n| DTC | Display | Condition | Possible Cause |\n|---|---|---|---|\n"
                        "| C1154 | PNP POS SIG | Signal line is open. | • Harness |\nYES >> GO TO 2.", "BRC.pdf"),
            _record(88, detection, "BRC.pdf"),
            _record(89, "4. CHECK BRAKE FLUID LEVEL SWITCH GROUND CIRCUIT\nContinuity check.", "BRC.pdf"),
        ])

        # The code's own heading is on p.86, so that page is neither cut nor relabeled.
        self.assertEqual(result[0]["id"], "P-AV-p0086-b01")
        self.assertEqual(result[0]["text"].splitlines()[0], "AV > C1154 PNP SWITCH > Description")
        self.assertEqual(
            result[1]["text"].splitlines()[0], "AV > C1155 BR FLUID LEVEL LOW > DTC Logic > DTC DETECTION LOGIC",
        )
        self.assertTrue(result[2]["text"].startswith("AV > C1155 BR FLUID LEVEL LOW > DTC Logic"))

    def test_dtc_table_inside_a_page_cuts_it_at_the_new_code(self) -> None:
        text = (
            "C1101, C1102 WHEEL SENSOR\nDTC Logic\nINFOID:7\nNO >> Inspection End.\n"
            "DTC DETECTION LOGIC\n| DTC | Display Item | Malfunction detected condition | Possible causes |\n"
            "|---|---|---|---|\n| C1105 | RR RH SENSOR-2 | Open circuit. | • Harness |\n"
            "| C1106 | RR LH SENSOR-2 | Open circuit. | • Harness |\nCheck the self-diagnosis results."
        )

        result = contextualize_records([_record(63, text, "BRC.pdf")])

        self.assertEqual([item["id"] for item in result], ["P-AV-p0063-b01-c01", "P-AV-p0063-b01-c02"])
        self.assertIn("Inspection End.", result[0]["text"])
        self.assertTrue(result[0]["text"].startswith("AV > C1101, C1102 WHEEL SENSOR > DTC Logic"))
        self.assertTrue(result[1]["text"].startswith("AV > C1105, C1106 > DTC Logic > DTC DETECTION LOGIC"))
        self.assertNotIn("Inspection End.", result[1]["text"])

    def test_dtc_tables_without_a_display_column_do_not_switch_codes(self) -> None:
        text = (
            "ABS ACTUATOR AND ELECTRIC UNIT (CONTROL UNIT)\nReference Value\nINFOID:8\nFail-safe list.\n"
            "| DTC | Malfunction detected condition | Fail-safe condition |\n|---|---|---|\n"
            "| C1155 | When brake fluid level low signal is detected. | VDC suspended. |"
        )

        result = contextualize_records([_record(40, text, "BRC.pdf")])

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["text"].startswith("AV > ABS ACTUATOR AND ELECTRIC UNIT (CONTROL UNIT) > Reference Value"))

    def test_input_records_are_not_modified(self) -> None:
        record = _record(1, "AV CONTROL UNIT\nReference Value\nINFOID:5\nBody.")
        original = dict(record)

        contextualize_records([record])

        self.assertEqual(record, original)


if __name__ == "__main__":
    unittest.main()
