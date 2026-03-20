#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
INPUT_DIR="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/ode_data_raw_v1"
OUTPUT_ROOT="/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/precompute_v1"
WORK_ROOT="$ROOT/ode/qz_convert_raw_v1_to_precompute_v1_shards"
DEFAULT_SHARD_COUNT=8
PYTHON_BIN="$ROOT/.venv/bin/python"
CONFIG_PATH="$ROOT/ode/configs/gen_ode_data_distilled.yaml"

cd "$ROOT"

export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
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

echo "[convert-raw-v1] start: $(date -Iseconds)"
echo "[convert-raw-v1] root: $ROOT"
echo "[convert-raw-v1] input_dir: $INPUT_DIR"
echo "[convert-raw-v1] output_root: $OUTPUT_ROOT"
echo "[convert-raw-v1] work_root: $WORK_ROOT"
echo "[convert-raw-v1] shard_count: $SHARD_COUNT"
echo "[convert-raw-v1] config: $CONFIG_PATH"

rm -rf "$OUTPUT_ROOT" "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
mkdir -p "$OUTPUT_ROOT/.precomputed/latents" \
         "$OUTPUT_ROOT/.precomputed/audio_latents" \
         "$OUTPUT_ROOT/.precomputed/conditions"

export INPUT_DIR WORK_ROOT SHARD_COUNT
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

input_dir = Path(os.environ["INPUT_DIR"])
work_root = Path(os.environ["WORK_ROOT"])
shard_count = int(os.environ["SHARD_COUNT"])

files = sorted(input_dir.glob("*.pt"))
if not files:
    raise SystemExit(f"No .pt files found under {input_dir}")

for shard_id in range(shard_count):
    shard_input = work_root / f"inputs/shard_{shard_id:02d}"
    shard_input.mkdir(parents=True, exist_ok=True)

for index, src in enumerate(files):
    shard_id = index % shard_count
    shard_input = work_root / f"inputs/shard_{shard_id:02d}"
    dst = shard_input / src.name
    dst.symlink_to(src.resolve())

print(f"Prepared {len(files)} files across {shard_count} shard directories.")
PY

pids=()
active_shards=0
for shard_id in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_input="$WORK_ROOT/inputs/shard_$(printf '%02d' "$shard_id")"
  shard_log="$WORK_ROOT/shard_$(printf '%02d' "$shard_id").log"
  shard_manifest="conversion_manifest_shard_$(printf '%02d' "$shard_id").json"
  shard_files=$(find "$shard_input" -maxdepth 1 -type l -name '*.pt' | wc -l)
  if [[ "$shard_files" -eq 0 ]]; then
    echo "[skip] shard=$shard_id has no files"
    continue
  fi

  echo "[launch] shard=$shard_id files=$shard_files log=$shard_log"
  CUDA_VISIBLE_DEVICES="$shard_id" "$PYTHON_BIN" ode/convert_ode_pt_to_precomputed.py \
    --config "$CONFIG_PATH" \
    --input-dir "$shard_input" \
    --output-dir "$OUTPUT_ROOT" \
    --stage stage2 \
    --standard-trajectory-step last \
    --manifest-name "$shard_manifest" \
    >"$shard_log" 2>&1 &

  pids+=("$!")
  active_shards=$((active_shards + 1))
done

if [[ "$active_shards" -eq 0 ]]; then
  echo "No active shards launched." >&2
  exit 1
fi

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

export OUTPUT_ROOT INPUT_DIR ACTIVE_SHARDS="$active_shards"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

input_dir = Path(os.environ["INPUT_DIR"])
output_root = Path(os.environ["OUTPUT_ROOT"])
precomputed_root = output_root / ".precomputed"
manifest_path = output_root / "conversion_manifest.json"

def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.pt"))

manifest = {
    "input_dir": str(input_dir),
    "output_root": str(output_root),
    "precomputed_root": str(precomputed_root),
    "stage": "stage2",
    "trajectory_step": "last",
    "export_mode": "standard",
    "sharded_parallel_conversion": True,
    "active_shards": int(os.environ["ACTIVE_SHARDS"]),
    "counts": {
        "latents": count_files(precomputed_root / "latents"),
        "audio_latents": count_files(precomputed_root / "audio_latents"),
        "conditions": count_files(precomputed_root / "conditions"),
    },
}

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

find "$OUTPUT_ROOT" -maxdepth 1 -type f -name 'conversion_manifest_shard_*.json' -delete

echo "[convert-raw-v1] done: $(date -Iseconds)"
