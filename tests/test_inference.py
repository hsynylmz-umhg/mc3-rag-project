"""CPU-only checks of evidence boundaries; no model or torch import needed."""

import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import Mock


_MODULE = Path(__file__).resolve().parents[1] / "starter" / "app" / "inference.py"
_SPEC = importlib.util.spec_from_file_location("mc3_inference", _MODULE)
inference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inference)


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "1": {"path": "logs/run.log", "text": "Warning: pump fault. Incident ticket ZX-291."},
            "2": {"path": "support/bugs.csv", "text": "ticket=ZX-291 | fixed_in=7.2.1 | status=closed"},
            "3": {"path": "notes.txt", "text": "Version 7.2.1 exists; unrelated to the pump."},
        }
        self.payload = {
            "answer": "7.2.1",
            "evidence": [
                {"id": 1, "quote": "Warning: pump fault. Incident ticket ZX-291."},
                {"id": "2", "quote": "ticket=ZX-291 | fixed_in=7.2.1"},
                {"id": "3", "quote": "Version 7.2.1 exists; unrelated to the pump."},
            ],
        }

    def test_two_hop_citations_survive_and_irrelevant_file_is_removed(self):
        candidate = inference._check_candidate(self.payload, self.sources)
        result = inference._checked_result(candidate, {"supported": True, "keep": [0, 1]}, self.sources)
        self.assertEqual(result["answer"], "7.2.1")
        self.assertEqual(result["citations"], ["logs/run.log", "support/bugs.csv"])

    def test_fabricated_or_noncontiguous_quote_is_rejected(self):
        for quote in ("ticket=ZX-291 | fixed_in=9.9.9", "ticket=ZX-291 ... status=closed"):
            self.payload["evidence"][1]["quote"] = quote
            self.assertIsNone(inference._check_candidate(self.payload, self.sources))

    def test_unknown_source_is_rejected(self):
        self.payload["evidence"][0]["id"] = "missing"
        self.assertIsNone(inference._check_candidate(self.payload, self.sources))

    def test_answer_must_exist_in_selected_quote(self):
        candidate = inference._check_candidate(self.payload, self.sources)
        self.assertEqual(inference._checked_result(candidate, {"supported": True, "keep": [0]}, self.sources), inference.refusal())
        self.payload["answer"] = "9.9.9"
        self.assertIsNone(inference._check_candidate(self.payload, self.sources))

    def test_numeric_and_identifier_substrings_are_not_values(self):
        self.assertFalse(inference._contains("maximum 194", "94", value=True))
        self.assertFalse(inference._contains("fixed 7.2.10", "7.2.1", value=True))
        self.assertFalse(inference._contains("part XYZ-123-A", "XYZ-123", value=True))
        self.assertTrue(inference._contains("maximum 94 °C", "94", value=True))
        self.assertTrue(inference._contains("Quarter: Q4\n  FY28", "Q4 FY28", value=True))
        self.assertTrue(inference._contains("The revision is REV-D3.", "REV-D3", value=True))
        self.assertTrue(inference._contains("The version is 7.2.1.", "7.2.1", value=True))
        self.assertFalse(inference._contains("maximum 94.5", "94", value=True))

    def test_unapproved_or_malformed_audits_refuse(self):
        candidate = inference._check_candidate(self.payload, self.sources)
        for audit in (None, {"supported": False, "keep": [1]}, {"supported": "true", "keep": [1]},
                      {"supported": True, "keep": [99]}, {"supported": True, "keep": [True]},
                      {"supported": True, "keep": [1, 1]}):
            self.assertEqual(inference._checked_result(candidate, audit, self.sources), inference.refusal())

    def test_json_requires_single_object_and_unique_keys(self):
        self.assertIsNone(inference._parse_json('{"answer":"ok","answer":"bad"}'))
        self.assertIsNone(inference._parse_json('{"answer":"ok"} trailing text'))
        self.assertIsNone(inference._parse_json('["ok"]'))
        self.assertEqual(inference._parse_json('```json\n{"answer":"ok"}\n```'), {"answer": "ok"})

    def test_citations_cannot_escape_corpus(self):
        for path in ("../secret", "/absolute", "D:/absolute", "a\\b", "a/../../b"):
            self.assertFalse(inference._safe_path(path))
        self.assertTrue(inference._safe_path("specs/valid.pdf"))

    def test_deadline_cutoff_cannot_publish_a_partial_draft(self):
        engine = inference.AnswerEngine.__new__(inference.AnswerEngine)
        engine._prepare_sources = Mock(return_value=self.sources)
        engine._generate = Mock(return_value='{"answer":"7.2.1","evidence":[')
        self.assertEqual(engine.answer("Which version?", [], time.monotonic() + 10), inference.refusal())
        self.assertEqual(engine._generate.call_count, 1)

    def test_draft_cannot_bypass_failed_verification(self):
        import json
        engine = inference.AnswerEngine.__new__(inference.AnswerEngine)
        engine._prepare_sources = Mock(return_value=self.sources)
        engine._generate = Mock(side_effect=[json.dumps(self.payload), '{"supported":false,"keep":[]}'])
        self.assertEqual(engine.answer("Which version?", [], time.monotonic() + 10), inference.refusal())
        self.assertEqual(engine._generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
