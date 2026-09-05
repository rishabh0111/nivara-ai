"""`embed_query(late_interaction=False)` must not load the late-interaction
model at all, not just discard its output.

Found live: the deployed Retriever always runs with `rerank=False` (the
measured default — see `retriever.py`'s module docstring), so nothing on
that path ever reads the late-interaction vector, yet `LocalEmbedder` loaded
and ran that ~126 MB encoder on every query anyway (`eval/retrieval_ablation.md`,
"Encoder footprint"). On a 512 MB instance that is dead weight worth not
paying for — this pins the fix with a fake encoder rather than a real model,
so it runs without a network call or a download.
"""

from __future__ import annotations

import pytest

from nivara_ai.retrieval.embedding import LocalEmbedder


@pytest.fixture(autouse=True)
def _reset_encoder_state(monkeypatch):
    from nivara_ai.retrieval import embedding

    monkeypatch.setattr(embedding, "_dense_encoders", {})
    monkeypatch.setattr(embedding, "_sparse_encoder_instance", None)
    monkeypatch.setattr(embedding, "_late_interaction_encoder_instance", None)


class _FakeDense:
    def query_embed(self, texts):
        return [[0.1, 0.2]]


class _FakeSparse:
    def query_embed(self, texts):
        from nivara_ai.retrieval.embedding import SparseVector

        return [SparseVector(indices=[1], values=[1.0])]


def _stub_dense_and_sparse(monkeypatch):
    monkeypatch.setattr("fastembed.TextEmbedding", lambda model: _FakeDense())
    monkeypatch.setattr("fastembed.SparseTextEmbedding", lambda model: _FakeSparse())


class TestLateInteractionIsSkippedWhenNotNeeded:
    def test_the_late_interaction_encoder_is_never_constructed(self, monkeypatch):
        _stub_dense_and_sparse(monkeypatch)

        def fail(model):
            raise AssertionError("late-interaction encoder must not be constructed")

        monkeypatch.setattr("fastembed.LateInteractionTextEmbedding", fail)

        result = LocalEmbedder().embed_query("how do I export my billing history?", late_interaction=False)
        assert result.late_interaction == []

    def test_defaults_to_computing_it_when_not_told_otherwise(self, monkeypatch):
        _stub_dense_and_sparse(monkeypatch)
        built = []

        class _FakeLate:
            def __init__(self):
                built.append(True)

            def query_embed(self, texts):
                return [[[0.5, 0.6]]]

        monkeypatch.setattr("fastembed.LateInteractionTextEmbedding", lambda model: _FakeLate())

        result = LocalEmbedder().embed_query("how do I export my billing history?")
        assert built == [True]
        assert result.late_interaction == [[0.5, 0.6]]
