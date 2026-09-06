# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

# The encoders, baked in rather than fetched on first request.
#
# fastembed downloads a model the first time it is constructed, into
# `FASTEMBED_CACHE_PATH` (default: a directory under /tmp). On a free instance
# that filesystem is new on every start and the instance spins down after
# fifteen idle minutes, so "first request" is most requests — each one paying a
# ~23-file download from the HF Hub, unauthenticated and rate-limited, while
# holding `_encoder_lock` (see `retrieval/embedding.py`). Found live: a stalled
# download there leaves every queued Turn waiting behind it on an unbounded
# acquire, and the Visitor watches "working" heartbeats forever.
#
# Downloaded here instead, at build time, into a path that ships in the image.
# The model names come from the module that commits them, so this cannot warm
# a different encoder than the one the request path loads.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed

RUN python -c "\
from nivara_ai.retrieval.embedding import DENSE_MODEL, SPARSE_MODEL, LATE_INTERACTION_MODEL; \
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding; \
TextEmbedding(DENSE_MODEL); \
SparseTextEmbedding(SPARSE_MODEL); \
LateInteractionTextEmbedding(LATE_INTERACTION_MODEL)"

EXPOSE 8000

CMD ["uvicorn", "nivara_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
