#!/usr/bin/env bash
# Push the visualizer Space to HuggingFace using upload-large-folder.
#
# Usage:
#   bash scripts/push_openenv.sh [stage_dir] [hf_repo_id]
#
# Requires: huggingface-hub (pip install "huggingface-hub[cli]>=0.24.0")
#   Log in once with: hf login
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${1:-${ROOT_DIR}/.openenv_push}"
REPO_ID="${2:-tokev/traffic-visualizer}"

"${ROOT_DIR}/scripts/prepare_openenv_push.sh" "${STAGE_DIR}"

PY_SCRIPT="$(mktemp /tmp/hf_push_XXXXXX.py)"
# Convert MINGW64 /c/Users/... path to Windows C:\Users\... for Python
WIN_STAGE_DIR="$(echo "${STAGE_DIR}" | sed 's|^/\([a-zA-Z]\)/|\1:/|' | tr '/' '\\')"
cat > "${PY_SCRIPT}" <<PYEOF
from huggingface_hub import HfApi

repo_id = "${REPO_ID}"
folder_path = r"${WIN_STAGE_DIR}"

# Subclass HfApi so that upload_large_folder's internal create_repo call
# always passes space_sdk="docker" for Space repos (the standalone function
# doesn't expose this parameter, causing a validator error).
class DockerSpaceApi(HfApi):
    def create_repo(self, repo_id, repo_type=None, space_sdk=None, **kwargs):
        if repo_type == "space" and space_sdk is None:
            space_sdk = "docker"
        return super().create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk=space_sdk, **kwargs)

api = DockerSpaceApi()
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
print(f"Space ready: {repo_id}")

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=folder_path,
)
print("Upload complete.")
PYEOF

echo ""
echo "Uploading to HuggingFace Space: ${REPO_ID} ..."
conda run -n traffic-llm python "${PY_SCRIPT}"
rm -f "${PY_SCRIPT}"

echo ""
echo "Done. Visit: https://huggingface.co/spaces/${REPO_ID}"
