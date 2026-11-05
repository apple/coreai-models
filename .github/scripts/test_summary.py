# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Write a short Markdown summary of JUnit XML test results.

Usage: test_summary.py TITLE RESULT.xml [RESULT.xml ...]

Appends to $GITHUB_STEP_SUMMARY when set, otherwise prints. Missing files are
ignored, so the step can run after a job that stopped before writing results.
Always exits 0: the test step decides whether the job fails.
"""

import os
import sys
import xml.etree.ElementTree as ET

MAX_LISTED = 50


def outcome(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    skipped = case.find("skipped")
    if skipped is not None:
        return "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def short_message(case: ET.Element) -> str:
    for tag in ("failure", "error"):
        node = case.find(tag)
        if node is not None:
            text = node.get("message") or node.text or ""
            joined = " ".join(
                line.strip() for line in text.splitlines() if line.strip()
            )
            return joined[:200]
    return ""


def summarize(title: str, paths: list[str]) -> str:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0}
    failed: list[str] = []
    found = False
    for path in paths:
        if not os.path.exists(path):
            continue
        found = True
        root = ET.parse(path).getroot()
        for case in root.iter("testcase"):
            result = outcome(case)
            counts[result] += 1
            if result == "failed":
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                message = short_message(case)
                failed.append(f"- `{name}`" + (f": {message}" if message else ""))

    lines = [f"### {title}", ""]
    if not found:
        lines.append("No test results: the run stopped before writing them.")
        return "\n".join(lines) + "\n"
    lines += [
        "| passed | failed | skipped | xfailed |",
        "|---:|---:|---:|---:|",
        f"| {counts['passed']} | {counts['failed']} | {counts['skipped']} | {counts['xfailed']} |",
    ]
    if failed:
        lines += ["", "Failed:", ""] + failed[:MAX_LISTED]
        if len(failed) > MAX_LISTED:
            lines.append(f"- … and {len(failed) - MAX_LISTED} more")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    text = summarize(sys.argv[1], sys.argv[2:])
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
