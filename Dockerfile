# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "nivara_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
