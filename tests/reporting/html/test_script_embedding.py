"""Document data embedded in the report's ``<script>`` block cannot end it.

The report writes each document's values into the page as JSON inside
``<script>``. A browser ends that block at the first ``</script``, even inside
a JSON string, so a value carrying ``</script><script>...</script>`` used to
run as its own script when the report was opened. Values are model output or
third-party documents, so this needs no access to stickler or the host.

See https://github.com/awslabs/stickler/issues/431
"""

import json
import time
from html.parser import HTMLParser

import pytest

from stickler import ComparableField, StructuredModel
from stickler.reporting.html.html_reporter import (
    EvaluationHTMLReporter,
    _json_for_script,
)
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

PAYLOAD = "</script><script>alert(1)</script>"


class Invoice(StructuredModel):
    vendor: str = ComparableField()


class _ScriptBodies(HTMLParser):
    """Collect the text of every ``<script>`` block, as a browser splits them."""

    def __init__(self):
        super().__init__()
        self._inside = False
        self.bodies = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._inside = True
            self.bodies.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self._inside = False

    def handle_data(self, data):
        if self._inside:
            self.bodies[-1] += data


def _report(tmp_path, predicted_vendor, model=Invoice, gt=None, pred=None):
    jsonl = tmp_path / "results.jsonl"
    jsonl.touch()
    evaluator = BulkStructuredModelEvaluator(
        target_schema=model, individual_results_jsonl=str(jsonl)
    )
    gt = gt if gt is not None else model(vendor="Acme")
    pred = pred if pred is not None else model(vendor=predicted_vendor)
    evaluator.update(gt, pred, doc_id="doc1")
    out = tmp_path / "report.html"
    result = EvaluationHTMLReporter().generate_report(
        evaluator.compute(),
        str(out),
        model_schema=model,
        individual_results_jsonl_path=str(jsonl),
    )
    assert result.success
    parser = _ScriptBodies()
    page = out.read_text()
    parser.feed(page)
    return page, parser.bodies


def test_a_document_value_cannot_end_the_script_block(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    _, baseline = _report(plain, "Acme Corp")
    page, bodies = _report(tmp_path, PAYLOAD)

    assert len(bodies) == len(baseline)
    assert not any(body.strip() == "alert(1)" for body in bodies)
    assert PAYLOAD not in page


def test_a_field_name_cannot_end_the_script_block(tmp_path):
    """A JSON Schema property name reaches the page through the field
    thresholds, the second blob in the same block, so it is guarded too."""
    name = "a" + PAYLOAD
    model = StructuredModel.from_json_schema(
        {"type": "object", "properties": {"vendor": {"type": "string"}, name: {"type": "string"}}}
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    _, baseline = _report(plain, "Acme Corp")
    page, bodies = _report(
        tmp_path,
        None,
        model=model,
        gt=model(**{"vendor": "Acme", name: "x"}),
        pred=model(**{"vendor": "Acme", name: "y"}),
    )

    assert len(bodies) == len(baseline)
    assert not any(body.strip() == "alert(1)" for body in bodies)
    assert PAYLOAD not in page


def test_the_value_still_reaches_the_page_data_unchanged(tmp_path):
    page, _ = _report(tmp_path, PAYLOAD)

    assert json.dumps(PAYLOAD).translate({ord("<"): "\\u003c", ord(">"): "\\u003e"}) in page


@pytest.mark.parametrize(
    "value",
    [PAYLOAD, "a & b", "x\u2028y\u2029z", {"k</script>": ["<!--", "-->"]}, None],
)
def test_json_for_script_round_trips_and_leaves_no_markup(value):
    out = _json_for_script(value)

    assert json.loads(out) == value
    for ch in "<>&\u2028\u2029":
        assert ch not in out


# The escape table `_json_for_script` used while it went through `str.translate`.
# `str.translate` with multi-character replacements is ~10x slower on large
# payloads, so the helper now chains `str.replace`; the output must not change.
_TRANSLATE_REFERENCE = str.maketrans(
    {
        "<": "\\u003c",
        ">": "\\u003e",
        "&": "\\u0026",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
    }
)


@pytest.mark.parametrize(
    "value",
    [
        PAYLOAD,
        "a < b > c & d",
        "x\u2028y\u2029z",
        "caf\u00e9 \u65e5\u672c \U0001f600 </script>",
        {"k</script>": ["<!--", "-->", "&amp;", 1, 2.5, None, True]},
        [{"reason": f"score >= {i / 10} & < {i}"} for i in range(50)],
        "plain ascii with nothing to escape",
        "",
        None,
        0,
    ],
)
def test_json_for_script_output_matches_the_translate_version(value):
    assert _json_for_script(value) == json.dumps(value).translate(_TRANSLATE_REFERENCE)


def test_json_for_script_costs_about_as_much_as_json_dumps():
    """A report with `<`, `>` or `&` in its reasons must not fall off the fast path (#457).

    With `str.translate` this took ~10x longer than `json.dumps` on a large blob;
    chained `str.replace` is within ~2x. The bound is loose enough for a noisy runner.
    """
    docs = [
        {"doc_id": f"d{i}", "reason": f"score >= 0.{i % 10} & value < {i}", "text": "x" * 200}
        for i in range(10000)
    ]

    def best_of(fn, runs=5):
        best = float("inf")
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best

    plain = best_of(lambda: json.dumps(docs))
    escaped = best_of(lambda: _json_for_script(docs))

    assert escaped < plain * 6
