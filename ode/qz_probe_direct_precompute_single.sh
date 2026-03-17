#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
INPUT_DIR="$ROOT/ode/data_distilled"
OUTPUT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/precompute_probe_single"
PYTHON_BIN="$ROOT/.venv/bin/python"
LIMIT="${LIMIT:-20}"
TOTAL_INPUT_FILES="${TOTAL_INPUT_FILES:-12000}"

cd "$ROOT"

export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected uv environment python at $PYTHON_BIN" >&2
  exit 1
fi

echo "[probe] start: $(date -Iseconds)"
echo "[probe] root: $ROOT"
echo "[probe] input_dir: $INPUT_DIR"
echo "[probe] output_root: $OUTPUT_ROOT"
echo "[probe] limit: $LIMIT"
echo "[probe] total_input_files_for_projection: $TOTAL_INPUT_FILES"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[probe] gpu inventory:"
  nvidia-smi -L || true
fi

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export LIMIT TOTAL_INPUT_FILES

"$PYTHON_BIN" ode/convert_ode_pt_to_precomputed.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --stage stage2 \
  --export-mode ode_regression \
  --trajectory-step all_non_last \
  --limit "$LIMIT"

"$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

output_root = Path("/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/precompute_probe_single")
precomputed_root = output_root / ".precomputed"
limit = int(os.environ["LIMIT"])
total_input_files = int(os.environ["TOTAL_INPUT_FILES"])

def count_and_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*.pt") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return len(files), total_bytes

latents_count, latents_bytes = count_and_size(precomputed_root / "latents")
audio_count, audio_bytes = count_and_size(precomputed_root / "audio_latents")
conditions_count, conditions_bytes = count_and_size(precomputed_root / "conditions")
manifest_path = output_root / "conversion_manifest.json"
manifest_bytes = manifest_path.stat().st_size if manifest_path.exists() else 0

total_bytes = latents_bytes + audio_bytes + conditions_bytes + manifest_bytes
projected_total_bytes = math.ceil(total_bytes * total_input_files / max(limit, 1))

summary = {
    "probe_input_files": limit,
    "projected_total_input_files": total_input_files,
    "counts": {
        "latents": latents_count,
        "audio_latents": audio_count,
        "conditions": conditions_count,
    },
    "sizes_bytes": {
        "latents": latents_bytes,
        "audio_latents": audio_bytes,
        "conditions": conditions_bytes,
        "manifest": manifest_bytes,
        "total_probe": total_bytes,
        "projected_total": projected_total_bytes,
    },
    "sizes_gib": {
        "total_probe": round(total_bytes / (1024 ** 3), 4),
        "projected_total": round(projected_total_bytes / (1024 ** 3), 2),
    },
}

print("[probe] summary:")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[probe] disk usage after run:"
du -sh "$OUTPUT_ROOT"
echo "[probe] done: $(date -Iseconds)"
