# ── Single-Space: CityFlow + Visualizer Dashboard ──────────────────────────
#
# Two-stage build:
#   1. builder  - compiles the vendored CityFlow Python extension
#   2. runtime  - installs API + visualizer dependencies, serves the PIXI.js
#                 dashboard via server/visualizer_app.py
#
# Runtime env vars:
#   DATA_DIR          generated CityFlow dataset root (default: /app/data/generated)
#   REPLAY_ROOT       on-disk replay cache            (default: /app/results/replays)
#   CHECKPOINT_PATH   DQN checkpoint                  (default: /app/artifacts/dqn_shared/best_validation.pt)
# ---------------------------------------------------------------------------

# ── Stage 1: Build CityFlow ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY third_party/CityFlow ./CityFlow
RUN rm -rf ./CityFlow/build
RUN pip install --no-cache-dir ./CityFlow


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# CityFlow compiled extension
COPY --from=builder /usr/local/lib/python3.12/site-packages/cityflow* \
                    /usr/local/lib/python3.12/site-packages/

# Python dependencies
# openenv_app/requirements.txt: fastapi, uvicorn, torch, openenv-core, etc.
# server/requirements.txt: joblib, python-multipart (extras not in openenv_app)
COPY openenv_app/requirements.txt ./openenv_app/requirements.txt
COPY server/requirements.txt      ./server/requirements.txt
RUN pip install --no-cache-dir \
      -r openenv_app/requirements.txt \
      -r server/requirements.txt

# Application source
COPY agents/         ./agents/
COPY district_llm/   ./district_llm/
COPY env/            ./env/
COPY openenv_app/    ./openenv_app/
COPY server/         ./server/
COPY training/       ./training/

# PIXI.js frontend
COPY third_party/CityFlow/frontend/ ./third_party/CityFlow/frontend/

# City data and splits
COPY data/splits/              ./data/splits/
COPY data/generated/city_0002/ ./data/generated/city_0002/

# Artifacts
COPY artifacts/dqn_shared/best_validation.pt \
     ./artifacts/dqn_shared/best_validation.pt
COPY artifacts/district_llm_adapter_v3/main_run/adapter/ \
     ./artifacts/district_llm_adapter_v3/main_run/adapter/

RUN mkdir -p /app/results/replays

ENV DATA_DIR=/app/data/generated
ENV REPLAY_ROOT=/app/results/replays
ENV CHECKPOINT_PATH=/app/artifacts/dqn_shared/best_validation.pt
ENV DISTRICT_LLM_ADAPTER_PATH=/app/artifacts/district_llm_adapter_v3/main_run/adapter

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.visualizer_app:app --host 0.0.0.0 --port ${PORT:-7860}"]
