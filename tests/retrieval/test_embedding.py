"""The local encoders: no provider quota, and the same numbers every run
(decision 26).

These pin actual embedding output. `filterwarnings` in `pyproject.toml`
silences fastembed's "model updated on HuggingFace" notice on the grounds
that the committed model id still returns the committed vectors — this file
is what makes that claim testable rather than asserted, so a silent
upstream change fails CI here instead of quietly shifting recall.

"Deterministic" holds two different claims, checked two different ways.
Same process, same machine, called twice — `TestTheEncodersAreDeterministic`
— is exact: this is what the retrieval pipeline, the ablation and the
threshold sweep actually rely on. Across machines, quantised ONNX inference
on CPU is not bit-reproducible: different SIMD instruction sets round the
int8 arithmetic differently, so the same committed model file returns
measurably different floats on a different CPU than the one these values
were pinned from. `TestTheCommittedModelStillReturnsTheCommittedVectors`
checks the dense encoder against a tolerance wide enough to absorb that,
not exact equality — it is still there to catch a genuinely different
model or a corrupted download, just not a different CPU.

Needs the model files (cached after first fetch), not a running Qdrant.
"""

import pytest

from nivara_ai.retrieval import DENSE_DIM, LATE_INTERACTION_DIM, LocalEmbedder

try:
    _EMBEDDER = LocalEmbedder()
    _PINNED = _EMBEDDER.embed_query("how do I change the billing contact?")
except Exception as exc:  # offline / no model cache is routine here, not a failure
    _EMBEDDER = None
    _SKIP_REASON = f"local encoders unavailable: {exc}"

pytestmark = pytest.mark.skipif(_EMBEDDER is None, reason=locals().get("_SKIP_REASON", ""))


class TestTheEncodersAreDeterministic:
    def test_the_same_query_dense_vector_is_reproduced_exactly(self):
        again = _EMBEDDER.embed_query("how do I change the billing contact?")
        assert again.dense == _PINNED.dense

    def test_the_same_query_sparse_vector_is_reproduced_exactly(self):
        again = _EMBEDDER.embed_query("how do I change the billing contact?")
        assert again.sparse == _PINNED.sparse

    def test_the_same_query_late_interaction_multivector_is_reproduced_exactly(self):
        again = _EMBEDDER.embed_query("how do I change the billing contact?")
        assert again.late_interaction == _PINNED.late_interaction


class TestTheCommittedModelStillReturnsTheCommittedVectors:
    def test_the_dense_encoder_is_the_pinned_768_dim_quantised_model(self):
        # abs=0.3: wide enough to absorb the CPU-dependent rounding a
        # quantised ONNX model shows across machines (observed drift here
        # has been up to ~0.21 per component), tight enough that a genuinely
        # different model or a corrupted download still fails this.
        assert len(_PINNED.dense) == DENSE_DIM == 768
        assert _PINNED.dense[:5] == pytest.approx(
            [
                0.65517,
                -0.27933,
                -2.94938,
                -0.53566,
                1.21775,
            ],
            abs=0.3,
        )

    def test_the_sparse_encoder_produces_the_pinned_bm25_terms(self):
        # BM25 term frequencies only — the IDF weighting is applied
        # server-side by Qdrant's modifier, so every value here is 1.0.
        assert _PINNED.sparse.indices == [2025758272, 617246313, 541354036]
        assert _PINNED.sparse.values == [1.0, 1.0, 1.0]

    def test_the_late_interaction_encoder_is_the_pinned_96_dim_multivector(self):
        # One 96-dim row per query token; the small ColBERT model pads a
        # query to 32 tokens, so the shape is fixed run to run.
        assert LATE_INTERACTION_DIM == 96
        assert {len(row) for row in _PINNED.late_interaction} == {96}
        assert len(_PINNED.late_interaction) == 32
        assert [round(x, 5) for x in _PINNED.late_interaction[0][:5]] == [
            -0.07656,
            0.05174,
            -0.04216,
            0.00259,
            -0.07164,
        ]
