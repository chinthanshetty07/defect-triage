"""GenAI classifier: Claude with structured outputs, given the most similar past cases as examples.

Off by default (the evaluation runs offline). On any API error, refusal or malformed output it returns the
baseline's answer, marked as such, so a model outage degrades quality rather than stopping triage.
"""
from __future__ import annotations

import json

from .data import SUBSYSTEMS

SEVERITIES = ["low", "medium", "high"]
SCHEMA = {
    "type": "object",
    "properties": {
        "subsystem": {"type": "string", "enum": SUBSYSTEMS},
        "severity": {"type": "string", "enum": SEVERITIES},
        "confidence": {"type": "number"},
    },
    "required": ["subsystem", "severity", "confidence"],
    "additionalProperties": False,
}
SYSTEM = (
    "You triage rolling-stock defect reports written by train maintainers (abbreviations and typos are common: "
    "BP brake pipe, BC brake cylinder, TMS train management system, ATP/TPWS train protection, VCB vacuum circuit "
    "breaker, OHE overhead line, panto pantograph, TM traction motor, PIS passenger information).\n"
    "Pick the subsystem whose fault the report describes. If two systems are mentioned, pick the one with the "
    "more safety-relevant symptom. Severity: high = could affect safe running (braking, train protection, running "
    "gear, fire/smoke, doors open in service); medium = service-affecting; low = comfort or cosmetic. "
    "Lower your confidence when unsure."
)


class ClaudeClassifier:
    model = "claude-opus-5-5"

    def __init__(self, fallback, similar=None, client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.fallback, self.similar = client, fallback, similar

    def predict_one(self, text: str) -> dict:
        examples = self.similar.top(text, k=4) if self.similar else []
        shots = "\n".join(f"- \"{e['text']}\" -> {e['subsystem']}, {e['severity']}" for e in examples)
        prompt = (f"<similar_past_reports>\n{shots}\n</similar_past_reports>\n\n" if shots else "") + \
                 f"<report>\n{text}\n</report>"
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=4000, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            return self._fallback(text, f"error:{type(exc).__name__}")
        if r.stop_reason == "refusal":
            return self._fallback(text, "refusal")
        raw = next((b.text for b in r.content if b.type == "text"), "")
        try:
            d = json.loads(raw)
            conf = min(max(float(d["confidence"]), 0.0), 1.0)
            return {"subsystem": d["subsystem"], "subsystem_conf": conf, "severity": d["severity"],
                    "severity_conf": conf, "provider": "claude"}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return self._fallback(text, "unparseable")

    def predict(self, texts) -> list[dict]:
        return [self.predict_one(t) for t in texts]

    def _fallback(self, text: str, why: str) -> dict:
        p = self.fallback.predict([text])[0]
        p["provider"] = f"baseline (claude {why})"
        return p
