"""The "Non-Match Entry Structure" table and the debugging snippet on
Understanding Results describe the records ``compare_with(document_non_matches=True)``
returns, scalar, nested and list-item records alike.

The table used to list one shape for every record. List-item records have no
``details`` key and carry ``reason`` (and ``similarity``) at the top level, so
code written from the table raised ``KeyError`` on the first list item.
"""

import contextlib
import io
import re
from pathlib import Path
from typing import List, Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators.levenshtein import LevenshteinComparator

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "docs"
    / "Guides"
    / "Evaluation"
    / "understanding-results.md"
).read_text(encoding="utf-8")


class Address(StructuredModel):
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)


class Line(StructuredModel):
    sku: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    qty: int = ComparableField(threshold=1.0)


class Document(StructuredModel):
    x: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    y: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    z: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    addr: Address = ComparableField()
    lines: List[Line] = ComparableField()


GROUND_TRUTH = Document(
    x="alpha",
    y=None,
    z="zeta",
    addr=Address(city="Paris"),
    lines=[Line(sku="A1", qty=1), Line(sku="B2", qty=2)],
)
PREDICTION = Document(
    x="omega",
    y="why",
    z=None,
    addr=Address(city="Rome"),
    lines=[Line(sku="A1", qty=9)],
)


def _non_matches():
    result = GROUND_TRUTH.compare_with(PREDICTION, document_non_matches=True)
    return {nm["field_path"]: nm for nm in result["non_matches"]}


def _section(heading: str) -> str:
    match = re.search(rf"{re.escape(heading)}\n(.*?)(?=\n#{{2,3}} |\n---)", PAGE, re.S)
    assert match, f"no section {heading!r}"
    return match.group(1)


def _documented_fields() -> dict:
    """Map each documented key to its description."""
    rows = re.findall(
        r"^\| `(\w+)` \|[^|]*\|(.*)\|\s*$",
        _section("### Non-Match Entry Structure"),
        re.M,
    )
    assert rows, "the Non-Match Entry Structure table is gone"
    return dict(rows)


def test_the_records_cover_scalar_nested_and_list_items():
    assert set(_non_matches()) == {"x", "y", "z", "addr.city", "lines[0]", "lines[1]"}


def test_every_key_a_record_carries_is_documented():
    documented = _documented_fields()
    for path, record in _non_matches().items():
        undocumented = set(record) - set(documented)
        assert not undocumented, f"{path} carries undocumented keys {undocumented}"


def test_details_is_documented_for_scalar_and_nested_records_only():
    records = _non_matches()
    description = _documented_fields()["details"]
    assert "scalar" in description.lower() and "nested" in description.lower()
    for path in ("x", "y", "z", "addr.city"):
        assert "reason" in records[path]["details"]
    for path in ("lines[0]", "lines[1]"):
        assert "details" not in records[path]
        assert isinstance(records[path]["reason"], str)


def test_the_debugging_snippet_prints_a_reason_for_every_false_discovery():
    block = re.search(
        r"### Debugging with Non-Matches\n.*?```python\n(.*?)```", PAGE, re.S
    )
    assert block, "the Debugging with Non-Matches snippet is gone"
    out = io.StringIO()
    namespace = {"ground_truth": GROUND_TRUTH, "prediction": PREDICTION}
    with contextlib.redirect_stdout(out):
        # Runs this repo's own documentation snippet; no external input.
        exec(block.group(1), namespace)  # nosec B102
    printed = [line for line in out.getvalue().splitlines() if "similarity=" in line]
    assert len(printed) == 3
    for line in printed:
        assert "(None)" not in line
        assert re.search(r"\(.+\)$", line)
    assert any(line.lstrip().startswith("lines[0]:") for line in printed)


@pytest.mark.parametrize("path", ["x", "lines[0]"])
def test_similarity_score_is_documented_for_false_discoveries(path):
    record = _non_matches()[path]
    assert record["non_match_type"] == "false_discovery"
    assert isinstance(record["similarity_score"], float)
    assert "false_discovery" in _documented_fields()["similarity_score"]
