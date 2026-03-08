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


FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages/cityflow* \
                    /usr/local/lib/python3.12/site-packages/

COPY openenv_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py      ./__init__.py
COPY client.py        ./client.py
COPY models.py        ./models.py
COPY agents/          ./agents/
COPY district_llm/    ./district_llm/
COPY env/             ./env/
COPY openenv_app/     ./openenv_app/
COPY server/          ./server/
COPY training/        ./training/
COPY data/splits/     ./data/splits/
COPY artifacts/dqn_shared/best_validation.pt ./artifacts/dqn_shared/best_validation.pt
COPY artifacts/district_llm_adapter_v3/main_run/adapter/ ./artifacts/district_llm_adapter_v3/main_run/adapter/

RUN mkdir -p /app/data/generated /app/data/splits

ENV DATA_DIR=/app/data/generated
ENV SPLITS_DIR=/app/data/splits
ENV CHECKPOINT_PATH=/app/artifacts/dqn_shared/best_validation.pt
ENV DISTRICT_LLM_ADAPTER_PATH=/app/artifacts/district_llm_adapter_v3/main_run/adapter

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
