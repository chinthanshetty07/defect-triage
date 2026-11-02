"""Finds the most similar past reports and how they were fixed, so the technician starts from history and the
LLM gets relevant examples (retrieval-augmented few-shot)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class SimilarCases:
    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history.reset_index(drop=True)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.history["text"])

    def top(self, text: str, k: int = 3, min_score: float = 0.2) -> list[dict]:
        q = self.vec.transform([text])
        scores = (self.matrix @ q.T).toarray().ravel()   # rows are L2-normalised, so this is cosine similarity
        idx = np.argsort(-scores)[:k]
        return [{"id": self.history.at[i, "id"], "text": self.history.at[i, "text"],
                 "subsystem": self.history.at[i, "subsystem"], "severity": self.history.at[i, "severity"],
                 "fix": self.history.at[i, "fix"], "score": round(float(scores[i]), 3)}
                for i in idx if scores[i] >= min_score]
