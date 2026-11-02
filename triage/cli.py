"""Triage one report from the command line: python -m triage.cli "BC presure low on coch C3 urgent" """
from __future__ import annotations

import json
import sys

from .baseline import BaselineClassifier
from .data import generate
from .decide import decide
from .similar import SimilarCases


def main(text: str) -> dict:
    history = generate()
    model = BaselineClassifier().fit(history.text, history.subsystem, history.severity)
    result = decide(text, model.predict([text])[0])
    result["similar_past_cases"] = SimilarCases(history).top(text)
    return result


if __name__ == "__main__":
    print(json.dumps(main(" ".join(sys.argv[1:]) or "brk not releasing on coach B4"), indent=2, default=str))
