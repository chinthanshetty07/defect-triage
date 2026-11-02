"""Turns a prediction into an action. The asymmetry is deliberate: a missed brake fault costs far more than a
technician looking at a door that was fine, so the rules only ever escalate, never downgrade."""
from __future__ import annotations

import re

from .data import SAFETY_CRITICAL

_RED_FLAGS = re.compile(r"smoke|fire|burn|smell|crack|arcing|derail|brake|brk|\bbp\b|\bbc\b|atp|tpws|balise|"
                        r"stuck open|hunting|vibration", re.I)


def decide(text: str, pred: dict, min_conf: float = 0.6) -> dict:
    reasons = []
    severity = pred["severity"]
    if pred["subsystem"] in SAFETY_CRITICAL and severity != "high":
        severity, _ = "high", reasons.append(f"{pred['subsystem']} is safety-critical")
    if _RED_FLAGS.search(text) and severity != "high":
        severity, _ = "high", reasons.append("report contains a safety red-flag term")
    needs_human = pred["subsystem_conf"] < min_conf or pred["severity_conf"] < min_conf
    if needs_human:
        reasons.append("low model confidence")
    queue = "safety_desk" if severity == "high" else pred["subsystem"]
    return {**pred, "severity": severity, "queue": queue, "needs_human_review": needs_human, "reasons": reasons}
