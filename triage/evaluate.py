"""Train on older reports, evaluate on newer ones, and report what matters operationally.

    python -m triage.evaluate              # offline: baseline only
    python -m triage.evaluate --llm 40     # also Claude on the 40 unseen-phrasing reports (needs ANTHROPIC_API_KEY; costs money)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .baseline import BaselineClassifier
from .data import generate, time_split
from .decide import decide
from .similar import SimilarCases


def score(test: pd.DataFrame, preds: list[dict], min_conf: float = 0.6) -> dict:
    decided = [decide(t, p, min_conf) for t, p in zip(test["text"], preds)]
    df = test.assign(pred_sub=[p["subsystem"] for p in preds], pred_sev=[p["severity"] for p in preds],
                     final_sev=[d["severity"] for d in decided], human=[d["needs_human_review"] for d in decided])
    high = df["severity"] == "high"
    auto = ~df["human"]
    confusions = (df[df.subsystem != df.pred_sub].groupby(["subsystem", "pred_sub"]).size()
                  .sort_values(ascending=False).head(5))
    return {
        "n_test": int(len(df)),
        "subsystem_accuracy": round(accuracy_score(df.subsystem, df.pred_sub), 3),
        "subsystem_macro_f1": round(f1_score(df.subsystem, df.pred_sub, average="macro"), 3),
        "severity_accuracy": round(accuracy_score(df.severity, df.pred_sev), 3),
        # The number that matters most: of the genuinely high-severity reports, how many were flagged high?
        "high_severity_recall_model": round(recall_score(high, df.pred_sev == "high"), 3),
        "high_severity_recall_with_rules": round(recall_score(high, df.final_sev == "high"), 3),
        "high_severity_flag_rate": round(float((df.final_sev == "high").mean()), 3),
        "auto_routed_share": round(float(auto.mean()), 3),
        "auto_routed_subsystem_accuracy": round(accuracy_score(df.subsystem[auto], df.pred_sub[auto]), 3) if auto.any() else None,
        "top_confusions": {f"{a} -> {b}": int(n) for (a, b), n in confusions.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="also evaluate Claude on this many unseen-phrasing reports")
    args = ap.parse_args()

    train, test = time_split(generate())
    base = BaselineClassifier().fit(train.text, train.subsystem, train.severity)
    unseen = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "unseen_phrasings.csv")
    results = {
        "baseline_templated_test": score(test, base.predict(test.text)),
        # Hand-written reports in wording the generator never produced: the honest measure of generalisation.
        "baseline_unseen_phrasings": score(unseen, base.predict(unseen.text)),
    }

    if args.llm:
        from .llm import ClaudeClassifier

        llm = ClaudeClassifier(base, SimilarCases(train))
        sample = unseen.head(args.llm)
        preds = llm.predict(sample.text)
        results["claude_unseen_phrasings"] = score(sample, preds)
        results["claude_unseen_phrasings"]["fell_back"] = sum(p["provider"] != "claude" for p in preds)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    np.random.seed(0)
    main()
