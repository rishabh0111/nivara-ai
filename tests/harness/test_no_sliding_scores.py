"""Decision 38: cosine similarity, ROUGE, BERTScore and every generic
text-similarity score are excluded — a number that slides is not a verdict.

This scans the harness package's own source (the same way
`tests/eval/test_generate.py` parses the eval generator to prove it never
imports the Corpus) so the exclusion is a property of the code rather than a
line in a README.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "src" / "nivara_ai" / "harness"

#: Substrings that would mean a sliding text-overlap metric had crept in. Lower
#: case; the scan is case-insensitive.
FORBIDDEN = (
    "cosine",
    "rouge",
    "bertscore",
    "bert_score",
    "bleu",
    "sacrebleu",
    "meteor",
    "levenshtein",
    "jaccard",
    "n-gram",
    "ngram",
    "edit_distance",
    "edit distance",
    "semantic_similarity",
    "embedding_similarity",
    "text_similarity",
)


@pytest.mark.parametrize("source", sorted(_HARNESS.rglob("*.py")), ids=lambda p: p.name)
def test_no_text_similarity_token_appears_in_the_harness_source(source: Path):
    text = source.read_text().lower()
    hits = [token for token in FORBIDDEN if token in text]
    assert not hits, f"{source.name} names a text-similarity metric: {hits}"


def test_the_scan_actually_covers_the_package():
    assert len(list(_HARNESS.rglob("*.py"))) >= 6
