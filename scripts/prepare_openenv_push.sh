#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/.openenv_push}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/third_party" "${OUT_DIR}/data" "${OUT_DIR}/artifacts/dqn_shared" \
  "${OUT_DIR}/artifacts/district_llm_adapter_v3/main_run"

cp -a "${ROOT_DIR}/Dockerfile" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/README.md" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/openenv_app" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/server" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/agents" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/district_llm" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/env" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/training" "${OUT_DIR}/"
cp -a "${ROOT_DIR}/third_party/CityFlow" "${OUT_DIR}/third_party/"
cp -a "${ROOT_DIR}/data/splits" "${OUT_DIR}/data/"
mkdir -p "${OUT_DIR}/data/generated"
cp -a "${ROOT_DIR}/data/generated/city_0002" "${OUT_DIR}/data/generated/"
cp -a "${ROOT_DIR}/artifacts/dqn_shared/best_validation.pt" "${OUT_DIR}/artifacts/dqn_shared/"
cp -a "${ROOT_DIR}/artifacts/district_llm_adapter_v3/main_run/adapter" \
  "${OUT_DIR}/artifacts/district_llm_adapter_v3/main_run/"

rm -rf "${OUT_DIR}/third_party/CityFlow/build"
find "${OUT_DIR}" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "${OUT_DIR}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

echo "Prepared lean OpenEnv push directory at ${OUT_DIR}"
du -sh "${OUT_DIR}"
