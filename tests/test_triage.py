import json
from types import SimpleNamespace

import pandas as pd
import pytest

from triage.baseline import BaselineClassifier
from triage.data import SUBSYSTEMS, generate, time_split
from triage.decide import decide
from triage.evaluate import score
from triage.llm import ClaudeClassifier
from triage.similar import SimilarCases


@pytest.fixture(scope="module")
def split():
    return time_split(generate(n=600))


@pytest.fixture(scope="module")
def model(split):
    train, _ = split
    return BaselineClassifier().fit(train.text, train.subsystem, train.severity)


def test_generator_is_deterministic_and_covers_every_subsystem():
    a, b = generate(n=200, seed=3), generate(n=200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.subsystem) == set(SUBSYSTEMS)


def test_time_split_never_trains_on_the_future(split):
    train, test = split
    assert train.reported_on.max() <= test.reported_on.min()
    assert len(test) == 150


def test_baseline_handles_typos(model):
    p = model.predict(["brk presure dropping on coch C3"])[0]
    assert p["subsystem"] == "brakes"


def test_safety_critical_subsystem_is_never_low(model):
    d = decide("brake pipe pressure dropping", {"subsystem": "brakes", "subsystem_conf": 0.9,
                                                "severity": "low", "severity_conf": 0.9, "provider": "x"})
    assert d["severity"] == "high" and d["queue"] == "safety_desk"


def test_red_flag_words_escalate_even_for_comfort_systems():
    d = decide("smoke from roof unit", {"subsystem": "hvac", "subsystem_conf": 0.95,
                                       "severity": "low", "severity_conf": 0.95, "provider": "x"})
    assert d["severity"] == "high"
    assert "red-flag" in d["reasons"][0]


def test_rules_never_downgrade():
    d = decide("door chime not sounding", {"subsystem": "doors", "subsystem_conf": 0.95,
                                          "severity": "high", "severity_conf": 0.95, "provider": "x"})
    assert d["severity"] == "high"


def test_low_confidence_goes_to_a_human():
    d = decide("odd noise", {"subsystem": "doors", "subsystem_conf": 0.3,
                            "severity": "low", "severity_conf": 0.9, "provider": "x"})
    assert d["needs_human_review"] is True


def test_similar_cases_returns_the_fix(split):
    train, _ = split
    hits = SimilarCases(train).top("panto not raising at stn", k=2)
    assert hits and hits[0]["subsystem"] == "pantograph" and hits[0]["fix"]


def test_score_reports_high_severity_recall(split, model):
    _, test = split
    m = score(test, model.predict(test.text))
    assert m["high_severity_recall_with_rules"] >= m["high_severity_recall_model"]
    assert m["high_severity_recall_with_rules"] == 1.0
    assert 0 <= m["auto_routed_share"] <= 1


def test_unseen_phrasings_are_harder_than_templated_test(split, model):
    _, test = split
    unseen = pd.read_csv("data/unseen_phrasings.csv")
    templated = score(test, model.predict(test.text))["subsystem_accuracy"]
    fresh = score(unseen, model.predict(unseen.text))["subsystem_accuracy"]
    assert fresh < templated - 0.2   # documents the generalisation gap rather than hiding it


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._response, self._error = response, error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


def _resp(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_claude_gets_similar_cases_as_examples(split, model):
    train, _ = split
    client = FakeClient(_resp(json.dumps({"subsystem": "pantograph", "severity": "high", "confidence": 0.8})))
    p = ClaudeClassifier(model, SimilarCases(train), client=client).predict_one("sparks seen from roof equipment")
    assert p["provider"] == "claude" and p["subsystem"] == "pantograph"
    assert "<similar_past_reports>" in client.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("client,why", [
    (FakeClient(error=TimeoutError()), "error:TimeoutError"),
    (FakeClient(_resp("", stop="refusal")), "refusal"),
    (FakeClient(_resp("{not json")), "unparseable"),
])
def test_claude_failures_fall_back_to_baseline(model, client, why):
    p = ClaudeClassifier(model, None, client=client).predict_one("door 3 not closing fully")
    assert p["provider"] == f"baseline (claude {why})" and p["subsystem"] == "doors"
