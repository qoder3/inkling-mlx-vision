#!/usr/bin/env zsh
set -euo pipefail

# Portable launcher for inkling-mlx-serve (override defaults via env vars or CLI flags).
MODEL_VARIANT="${MODEL_VARIANT:-8.9GB}"
MODEL="${INKLING_MODEL:-shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-${MODEL_VARIANT}}"
DRAFTER="${INKLING_DRAFTER:-shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter}"
VISION="${INKLING_VISION:-./vision_tower.safetensors}"
HOST="${INKLING_HOST:-127.0.0.1}"
PORT="${INKLING_PORT:-8080}"
CONTEXT="${INKLING_CONTEXT:-131072}"
MAX_TOKENS="${INKLING_MAX_TOKENS:-16384}"
PREFILL_CHUNK="${INKLING_PREFILL_CHUNK:-512}"
if [[ "${MODEL_VARIANT}" == "12.8GB" || "${MODEL}" == *"12.8GB"* ]]; then
  PROMPT_CACHE="${INKLING_PROMPT_CACHE:-2}"
else
  PROMPT_CACHE="${INKLING_PROMPT_CACHE:-3}"
fi
MODEL_ID="${INKLING_MODEL_ID:-inkling-2-mini}"

RESTART=0
ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --restart)
      RESTART=1
      shift
      ;;
    --port)
      PORT="${2:-$PORT}"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

EXISTING_PIDS=($(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true))
if (( ${#EXISTING_PIDS[@]} > 0 )); then
  if (( RESTART == 1 )); then
    echo "Stopping existing process(es) on port ${PORT}: ${EXISTING_PIDS[*]}" >&2
    kill "${EXISTING_PIDS[@]}" 2>/dev/null || true
    for _ in {1..30}; do
      sleep 0.1
      if ! lsof -tiTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        break
      fi
    done
  else
    echo "Port ${PORT} is already in use by PID(s): ${EXISTING_PIDS[*]}" >&2
    echo "Pass --restart to replace it, or stop it first: kill ${EXISTING_PIDS[*]}" >&2
    exit 1
  fi
fi

export LAMQ_GLOBAL_KV="${LAMQ_GLOBAL_KV:-fp8}"
export PYTHONUNBUFFERED=1

VISION_FLAGS=()
if [[ -f "${VISION}" ]]; then
  VISION_FLAGS=(--vision "${VISION}")
fi

DRAFTER_FLAGS=()
if [[ -n "${DRAFTER}" ]]; then
  DRAFTER_FLAGS=(--drafter "${DRAFTER}")
fi

exec inkling-mlx-serve \
  "${MODEL}" \
  "${DRAFTER_FLAGS[@]}" \
  "${VISION_FLAGS[@]}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-id "${MODEL_ID}" \
  --context "${CONTEXT}" \
  --max-tokens "${MAX_TOKENS}" \
  --prompt-cache "${PROMPT_CACHE}" \
  --prefill-chunk "${PREFILL_CHUNK}" \
  --reasoning-effort medium \
  --temp 0.6 \
  --top-p 0.95 \
  "${ARGS[@]}"
