"""The Sensitive category classifier — the Gate's one lexical Free signal.

A Bernoulli Naive Bayes over word unigrams and bigrams, fit on the committed
eval questions: the 400 generated ordinary cases as the negative class, the 150
human-reviewed sensitive cases as the positive one. Closed-form — Laplace-
smoothed counts, no gradient descent, no seed — so the fit is byte-for-byte
reproducible and `tests/gate/test_sensitive.py` re-fits it and compares.

Why learned rather than a hand-authored keyword list: decision 31 settles the
Gate's *combination* by fitting it to the labelled set rather than hand-weighting,
and a hand-tuned lexicon for this signal would be the same move the project
rejects everywhere else. The committed artifact is still fully inspectable —
`gate/sensitive_classifier.json` is a readable `{term: log-odds}` dictionary, and
a large positive weight on `wire transfer` or `refund the` is exactly what a
reviewer would expect.

**Its failure mode is lexical, and independent of the two retrieval signals**
(`nivara_ai.gate.signals`). It reads only the words of the question, so it is
false-low on a sensitive ask phrased with no money/fraud/identity vocabulary and
false-high on an ordinary question that mentions a charge or a password in
passing — neither of which correlates with a retrieval score or a rerank margin,
which are computed from embeddings over the Corpus. That independence is the
whole point of combining three signals rather than trusting one.

This module has no import of `nivara_ai.corpus`: the classifier learns what a
*question* sounds like, never what a Document says, and
`tests/gate/test_sensitive.py` parses this source to keep that true — the same
structural guarantee `nivara_ai.eval.generate` holds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions

#: The committed classifier — a readable `{term: weight}` dictionary plus the
#: bias and the hash of the training questions it was fit against. Regenerated
#: by `scripts/gate_calibration.py`, never hand-edited. Package data (like
#: `turn/system_prompt.md`), because the Gate scores every Turn's question
#: against it on the request path and it ships inside the deployed image.
CLASSIFIER_PATH = Path(__file__).with_name("sensitive_classifier.json")

#: A term must appear in at least this many training questions to earn a weight
#: — below it the count is too thin for the smoothed ratio to mean anything, and
#: the vocabulary fills with one-off typos and proper nouns.
MIN_DOCUMENT_FREQUENCY = 3

_TOKEN = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    """The unigrams and bigrams of `text`, deduplicated — this is a Bernoulli
    model, so a term is present or absent, never counted."""

    tokens = _TOKEN.findall(text.lower())
    grams = set(tokens)
    grams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return grams


@dataclass(frozen=True)
class SensitiveClassifier:
    """`score(question)` in [0, 1]: the modelled probability that a question is
    about money movement, a disputed or fraudulent charge, or identity and
    account recovery — the topics the Gate routes to a person (decision 30).

    `weights[t]` is the Bernoulli log-odds contribution of term `t` being
    present; `bias` folds in the class prior and every term's absent-case
    contribution, so `score` is `sigmoid(bias + sum of present-term weights)`.
    """

    weights: dict[str, float]
    bias: float
    training_questions_sha: str

    def score(self, question: str) -> float:
        logit = self.bias + sum(
            self.weights[t] for t in _terms(question) if t in self.weights
        )
        return 1.0 / (1.0 + math.exp(-logit))

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "bias": round(self.bias, 6),
            "training_questions_sha": self.training_questions_sha,
            "weights": {t: round(w, 6) for t, w in sorted(self.weights.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> SensitiveClassifier:
        return cls(
            weights=dict(data["weights"]),
            bias=data["bias"],
            training_questions_sha=data["training_questions_sha"],
        )

    def save(self, path: Path = CLASSIFIER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class _LabelledQuestion:
    text: str
    sensitive: bool


def training_questions() -> list[_LabelledQuestion]:
    """The labelled set this classifier learns from: ordinary eval questions
    (negative) and the human-reviewed sensitive slice (positive). Sorted by
    text so the training-set hash is stable regardless of file order."""

    rows = [_LabelledQuestion(q.text, False) for q in load_questions()]
    rows += [_LabelledQuestion(q.text, True) for q in load_reviewed_sensitive_questions()]
    return sorted(rows, key=lambda r: (r.text, r.sensitive))


def _training_sha(rows: list[_LabelledQuestion]) -> str:
    payload = "\n".join(f"{int(r.sensitive)}\t{r.text}" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fit_sensitive_classifier(
    rows: list[_LabelledQuestion] | None = None,
) -> SensitiveClassifier:
    """Fit the Bernoulli NB in closed form. Deterministic — no randomness, no
    iteration count, so a re-fit reproduces the committed file exactly."""

    rows = rows if rows is not None else training_questions()
    positives = [r for r in rows if r.sensitive]
    negatives = [r for r in rows if not r.sensitive]
    n_pos, n_neg = len(positives), len(negatives)

    pos_df: dict[str, int] = {}
    neg_df: dict[str, int] = {}
    for row in rows:
        target = pos_df if row.sensitive else neg_df
        for term in _terms(row.text):
            target[term] = target.get(term, 0) + 1

    vocab = sorted(
        t
        for t in {*pos_df, *neg_df}
        if pos_df.get(t, 0) + neg_df.get(t, 0) >= MIN_DOCUMENT_FREQUENCY
    )

    weights: dict[str, float] = {}
    # bias = log(prior ratio) + sum over the whole vocabulary of the absent-term
    # log-odds; each present term then swaps its absent contribution for its
    # present one via `weights[t]`.
    bias = math.log(n_pos / n_neg)
    for term in vocab:
        p_pos = (pos_df.get(term, 0) + 1) / (n_pos + 2)
        p_neg = (neg_df.get(term, 0) + 1) / (n_neg + 2)
        present = math.log(p_pos) - math.log(p_neg)
        absent = math.log(1 - p_pos) - math.log(1 - p_neg)
        weights[term] = present - absent
        bias += absent

    return SensitiveClassifier(
        weights=weights, bias=bias, training_questions_sha=_training_sha(rows)
    )


def load_sensitive_classifier(path: Path = CLASSIFIER_PATH) -> SensitiveClassifier:
    """Read the committed classifier. The request path calls this; only
    `scripts/gate_calibration.py` fits and writes a fresh one."""

    return SensitiveClassifier.from_dict(json.loads(path.read_text()))
