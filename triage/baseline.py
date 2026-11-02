"""Classical NLP baseline: TF-IDF on words and character n-grams (robust to typos like "brk" and "presure")
feeding a logistic regression. Fast, explainable, and the bar any LLM approach has to beat."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"))])


class BaselineClassifier:
    def __init__(self) -> None:
        self.subsystem = make_pipeline()
        self.severity = make_pipeline()

    def fit(self, texts, subsystems, severities) -> "BaselineClassifier":
        self.subsystem.fit(texts, subsystems)
        self.severity.fit(texts, severities)
        return self

    def predict(self, texts) -> list[dict]:
        sp, vp = self.subsystem.predict_proba(texts), self.severity.predict_proba(texts)
        s_cls, v_cls = self.subsystem.classes_, self.severity.classes_
        out = []
        for a, b in zip(sp, vp):
            i, j = int(np.argmax(a)), int(np.argmax(b))
            out.append({"subsystem": s_cls[i], "subsystem_conf": float(a[i]),
                        "severity": v_cls[j], "severity_conf": float(b[j]), "provider": "baseline"})
        return out
