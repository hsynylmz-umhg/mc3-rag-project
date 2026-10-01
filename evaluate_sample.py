#!/usr/bin/env python3
"""Run every sample question in a FRESH container exec, check exact citation sets.

The container must already be running with GPU devices and networking disabled.
This evaluator is outside the image: expected answers never enter the pipeline.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def normalized(value):
    return re.sub(r"\s+", "", str(value)).casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", help="name of an already running MC3 container")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("sample-questions.json"))
    parser.add_argument("--report", type=Path, default=Path("sample-results.json"))
    args = parser.parse_args()
    report = []
    for question in json.loads(args.questions.read_text(encoding="utf-8"))["queries"]:
        query_id = f"query_{question['n']:02}"
        started = time.monotonic()
        record = {"id": query_id, "passed": False}
        try:
            process = subprocess.run(["docker", "exec", args.container, "python3", "/app/app.py",
                                      "--corpus", "/app/corpus", "--query-id", query_id,
                                      "--query", question["query"]],
                                     capture_output=True, text=True, timeout=30)
            elapsed = time.monotonic() - started
            record["seconds"] = round(elapsed, 3)
            if process.returncode:
                raise RuntimeError(process.stderr[-1500:])
            output = subprocess.run(["docker", "exec", args.container, "cat", f"/app/output/{query_id}_output.json"],
                                    capture_output=True, text=True, timeout=5, check=True)
            payload = json.loads(output.stdout)
            accepted = [question["expected_answer"], *question.get("answer_aliases", [])]
            answer_ok = any(normalized(payload.get("answer")) == normalized(value) for value in accepted)
            citations_ok = set(payload.get("citations", [])) == set(question["expected_citations"])
            record.update(output=payload, answer_ok=answer_ok, citations_ok=citations_ok,
                          passed=answer_ok and citations_ok and elapsed < 30)
        except Exception as exc:
            record["error"] = str(exc)
        report.append(record)
        print(f"{query_id}: {'PASS' if record['passed'] else 'FAIL'} ({record.get('seconds', 'timeout')}s)")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(item["passed"] for item in report)
    print(f"{passed}/{len(report)} exact sample results; report: {args.report}")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
