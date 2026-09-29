# PRISM - reproducible image for the scoring API and the batch pipeline.
# Multi-stage so the runtime layer carries no build toolchain.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.11-slim AS runtime

# libgomp is required at runtime by LightGBM and XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash prism

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    OMP_NUM_THREADS=4

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=prism:prism prism ./prism
COPY --chown=prism:prism configs ./configs
COPY --chown=prism:prism pyproject.toml README.md SPEC.md ./
RUN mkdir -p artifacts/data artifacts/models artifacts/reports artifacts/mlruns docs/figures \
    && chown -R prism:prism /app

USER prism
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: serve the API. Override to run the batch pipeline instead, e.g.
#   docker run prism-retention python -m prism.pipelines.run_all --config configs/fast.yaml
CMD ["uvicorn", "prism.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
