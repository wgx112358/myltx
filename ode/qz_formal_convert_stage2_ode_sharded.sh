#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
INPUT_DIR="$ROOT/ode/data_distilled"
FINAL_OUTPUT_DIR="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/precompute"
WORK_ROOT="$ROOT/ode/qz_convert_stage2_ode_shards"
DEFAULT_SHARD_COUNT=8
PYTHON_BIN="$ROOT/.venv/bin/python"

cd "$ROOT"

export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected uv environment python at $PYTHON_BIN" >&2
  exit 1
fi

detect_shard_count() {
  if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
    echo "$SHARD_COUNT_OVERRIDE"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [[ "$gpu_count" -gt 0 ]]; then
      echo "$gpu_count"
      return
    fi
  fi

  echo "$DEFAULT_SHARD_COUNT"
}

SHARD_COUNT="$(detect_shard_count)"
if ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]] || [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "Invalid shard count: $SHARD_COUNT" >&2
  exit 1
fi

echo "Using shard count: $SHARD_COUNT"

rm -rf "$FINAL_OUTPUT_DIR" "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
mkdir -p "$FINAL_OUTPUT_DIR/.precomputed/latents" \
         "$FINAL_OUTPUT_DIR/.precomputed/audio_latents" \
         "$FINAL_OUTPUT_DIR/.precomputed/conditions"

export SHARD_COUNT
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path("/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx")
input_dir = root / "ode/data_distilled"
work_root = root / "ode/qz_convert_stage2_ode_shards"
shard_count = int(os.environ["SHARD_COUNT"])

files = sorted(input_dir.glob("*.pt"))
if not files:
    raise SystemExit("No .pt files found for sharded conversion.")

for shard_id in range(shard_count):
    shard_input = work_root / f"inputs/shard_{shard_id:02d}"
    shard_input.mkdir(parents=True, exist_ok=True)

for index, src in enumerate(files):
    shard_id = index % shard_count
    shard_input = work_root / f"inputs/shard_{shard_id:02d}"
    dst = shard_input / src.name
    dst.symlink_to(src.resolve())

print(f"Prepared {len(files)} files across {shard_count} shards.")
PY

pids=()
for shard_id in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_input="$WORK_ROOT/inputs/shard_$(printf '%02d' "$shard_id")"
  shard_log="$WORK_ROOT/shard_$(printf '%02d' "$shard_id").log"
  shard_manifest="conversion_manifest_shard_$(printf '%02d' "$shard_id").json"

  echo "[launch] shard=$shard_id input=$shard_input output=$FINAL_OUTPUT_DIR manifest=$shard_manifest log=$shard_log"

  CUDA_VISIBLE_DEVICES="$shard_id" "$PYTHON_BIN" ode/convert_ode_pt_to_precomputed.py \
    --input-dir "$shard_input" \
    --output-dir "$FINAL_OUTPUT_DIR" \
    --stage stage2 \
    --export-mode ode_regression \
    --trajectory-step all_non_last \
    --manifest-name "$shard_manifest" \
    >"$shard_log" 2>&1 &

  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard conversion failed." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path("/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx")
final_root = Path("/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/precompute")
manifest_path = final_root / "conversion_manifest.json"
shard_count = int(os.environ["SHARD_COUNT"])

latents = sum(1 for _ in (final_root / ".precomputed/latents").glob("*.pt"))
audio = sum(1 for _ in (final_root / ".precomputed/audio_latents").glob("*.pt"))
conditions = sum(1 for _ in (final_root / ".precomputed/conditions").glob("*.pt"))

manifest = {
    "input_dir": str(root / "ode/data_distilled"),
    "output_root": str(final_root),
    "export_mode": "ode_regression",
    "stage": "stage2",
    "trajectory_step": "all_non_last",
    "sharded_parallel_conversion": True,
    "direct_write_to_final_output": True,
    "shard_count": shard_count,
    "counts": {
        "latents": latents,
        "audio_latents": audio,
        "conditions": conditions,
    },
}

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

find "$FINAL_OUTPUT_DIR" -maxdepth 1 -type f -name 'conversion_manifest_shard_*.json' -delete

echo "Sharded conversion finished successfully."
