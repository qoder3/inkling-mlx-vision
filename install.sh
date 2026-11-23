#!/usr/bin/env bash
set -euo pipefail

# One-command installer for inkling-mlx-vision on Apple Silicon macOS.
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/qoder3/inkling-mlx-vision/main/install.sh | bash
#
# With options:
#   curl -fsSL https://raw.githubusercontent.com/qoder3/inkling-mlx-vision/main/install.sh | bash -s -- --variant 12.8GB --start

REPO_URL="${INKLING_REPO_URL:-https://github.com/qoder3/inkling-mlx-vision.git}"
LOOKATMYQUANT_URL="${LOOKATMYQUANT_URL:-git+https://github.com/ShinMegamiBoson/lookatmyquant.git}"

INSTALL_DIR=""
VARIANT=""
PORT="8080"
KV_MODE=""
SKIP_VISION=0
RUN_TESTS=0
START_AFTER=0
DRY_RUN=0

# Colors (disabled when stdout is not a TTY)
if [[ -t 1 ]]; then
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'
  C_CYAN=$'\033[36m'
  C_RESET=$'\033[0m'
else
  C_BOLD="" C_DIM="" C_GREEN="" C_YELLOW="" C_RED="" C_CYAN="" C_RESET=""
fi

info()  { printf "%s==>%s %s\n" "${C_CYAN}${C_BOLD}" "${C_RESET}" "$*"; }
ok()    { printf "%s ✓%s %s\n"  "${C_GREEN}${C_BOLD}" "${C_RESET}" "$*"; }
warn()  { printf "%s !%s %s\n"  "${C_YELLOW}${C_BOLD}" "${C_RESET}" "$*" >&2; }
die()   { printf "%s ✗%s %s\n"  "${C_RED}${C_BOLD}" "${C_RESET}" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
inkling-mlx-vision installer (Apple Silicon macOS)

Usage:
  ./install.sh [options]
  curl -fsSL https://raw.githubusercontent.com/qoder3/inkling-mlx-vision/main/install.sh | bash -s -- [options]

Options:
  --dir PATH                Install/checkout directory (default: current repo or ~/inkling-mlx-vision)
  --variant 5.4GB|8.9GB|12.8GB
                            Model quant variant (default: auto-selected from Unified Memory)
  --port PORT               Server port saved to .inkling.env (default: 8080)
  --kv fp8|q4               Global KV cache quantization (default: fp8, or q4 on <=16 GB Macs)
  --no-vision               Skip extracting vision_tower.safetensors (~637 MB)
  --run-tests               Install pytest and run the offline test suite after setup
  --start                   Start the server immediately after installation
  --dry-run                 Print the detected hardware and planned steps without modifying anything
  -h, --help                Show this help message
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dir)        [[ $# -ge 2 ]] || die "--dir requires a path"; INSTALL_DIR="$2"; shift 2 ;;
    --dir=*)      INSTALL_DIR="${1#*=}"; shift ;;
    --variant)    [[ $# -ge 2 ]] || die "--variant requires a value"; VARIANT="$2"; shift 2 ;;
    --variant=*)  VARIANT="${1#*=}"; shift ;;
    --port)       [[ $# -ge 2 ]] || die "--port requires a port"; PORT="$2"; shift 2 ;;
    --port=*)     PORT="${1#*=}"; shift ;;
    --kv)         [[ $# -ge 2 ]] || die "--kv requires fp8 or q4"; KV_MODE="$2"; shift 2 ;;
    --kv=*)       KV_MODE="${1#*=}"; shift ;;
    --no-vision)  SKIP_VISION=1; shift ;;
    --run-tests)  RUN_TESTS=1; shift ;;
    --start)      START_AFTER=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "Unknown option: $1 (run with --help)" ;;
  esac
done

# 1. Hardware & OS preflight
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
if [[ "${OS_NAME}" != "Darwin" || "${ARCH}" != "arm64" ]]; then
  die "inkling-mlx-vision requires macOS on Apple Silicon (M1/M2/M3/M4). Detected: ${OS_NAME} ${ARCH}."
fi

CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")"
MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
MEM_GB=$(( MEM_BYTES / 1073741824 ))

# Auto-select quant variant and KV mode from Unified Memory when not explicitly specified
if [[ -z "${VARIANT}" ]]; then
  if (( MEM_GB >= 32 )); then
    VARIANT="12.8GB"
  else
    VARIANT="8.9GB"
  fi
fi
case "${VARIANT}" in
  5.4GB|8.9GB|12.8GB) ;;
  *) die "Unsupported --variant '${VARIANT}'. Expected one of: 5.4GB, 8.9GB, 12.8GB." ;;
esac

if [[ -z "${KV_MODE}" ]]; then
  if (( MEM_GB > 0 && MEM_GB <= 16 )); then
    KV_MODE="q4"
  else
    KV_MODE="fp8"
  fi
fi
case "${KV_MODE}" in
  fp8|q4) ;;
  *) die "Unsupported --kv '${KV_MODE}'. Expected fp8 or q4." ;;
esac

# Detect whether we are already inside a checkout of inkling-mlx-vision
if [[ -z "${INSTALL_DIR}" ]]; then
  if [[ -f "./pyproject.toml" ]] && grep -q 'name = "inkling-mlx-vision"' ./pyproject.toml 2>/dev/null; then
    INSTALL_DIR="$(pwd)"
  else
    INSTALL_DIR="${INKLING_HOME:-${HOME}/inkling-mlx-vision}"
  fi
fi

info "Detected ${C_BOLD}${CHIP}${C_RESET} with ${C_BOLD}${MEM_GB} GB${C_RESET} Unified Memory"
ok   "Install directory : ${INSTALL_DIR}"
ok   "Model variant     : Inkling-2-Mini-LookAtMyQuant-${VARIANT} (Global KV: ${KV_MODE}, port: ${PORT})"
if (( SKIP_VISION == 1 )); then
  ok "Vision tower      : skipped (--no-vision)"
else
  ok "Vision tower      : vision_tower.safetensors (~637 MB HTTP-Range extract)"
fi

if (( DRY_RUN == 1 )); then
  info "Dry-run mode enabled; no files were modified."
  exit 0
fi

# 2. Check git & install uv if missing
if ! command -v git >/dev/null 2>&1; then
  die "git is required. Install Xcode Command Line Tools first: xcode-select --install"
fi

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (fast Python package & environment manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv installed but not found in PATH. Open a new terminal and re-run."
fi
ok "uv ready: $(uv --version)"

# 3. Clone or update repository
if [[ -f "${INSTALL_DIR}/pyproject.toml" ]] && grep -q 'name = "inkling-mlx-vision"' "${INSTALL_DIR}/pyproject.toml" 2>/dev/null; then
  ok "Using existing repository at ${INSTALL_DIR}"
else
  if [[ -e "${INSTALL_DIR}" ]] && [[ -n "$(ls -A "${INSTALL_DIR}" 2>/dev/null || true)" ]]; then
    die "Directory '${INSTALL_DIR}' already exists and is not an inkling-mlx-vision checkout. Pass --dir PATH."
  fi
  info "Cloning inkling-mlx-vision into ${INSTALL_DIR}..."
  git clone --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"

# 4. Create Python 3.12 virtualenv & install dependencies
if [[ ! -x ".venv/bin/python" ]]; then
  info "Creating Python 3.12 virtual environment (.venv)..."
  uv venv --python 3.12 .venv
else
  ok "Reusing existing .venv ($(.venv/bin/python --version))"
fi

info "Installing lookatmyquant and inkling-mlx-vision into .venv..."
uv pip install --python .venv/bin/python "${LOOKATMYQUANT_URL}"
if (( RUN_TESTS == 1 )); then
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  uv pip install --python .venv/bin/python -e .
fi
ok "Packages installed"

# 5. Extract vision_tower.safetensors (~637 MB via HTTP Range, skips 16.9 GB LLM download)
if (( SKIP_VISION == 0 )); then
  if [[ -f "${INSTALL_DIR}/vision_tower.safetensors" ]]; then
    ok "vision_tower.safetensors already exists ($(du -h "${INSTALL_DIR}/vision_tower.safetensors" | awk '{print $1}'))"
  else
    info "Extracting vision_tower.safetensors (~637 MB from Hugging Face via HTTP Range)..."
    .venv/bin/inkling-extract-vision -o "${INSTALL_DIR}/vision_tower.safetensors"
    ok "Extracted vision_tower.safetensors"
  fi
fi

# 6. Write .inkling.env defaults (non-overwriting so explicit env vars always win)
cat > "${INSTALL_DIR}/.inkling.env" <<EOF
# Auto-generated by install.sh — edit freely or override via environment variables.
: "\${MODEL_VARIANT:=${VARIANT}}"
: "\${INKLING_PORT:=${PORT}}"
: "\${LAMQ_GLOBAL_KV:=${KV_MODE}}"
EOF
chmod +x "${INSTALL_DIR}/scripts/start_server.sh"
ok "Saved defaults to ${INSTALL_DIR}/.inkling.env"

# 7. Install convenient launcher shim in ~/.local/bin/inkling-server (best-effort)
SHIM_DIR="${HOME}/.local/bin"
if mkdir -p "${SHIM_DIR}" 2>/dev/null; then
  cat > "${SHIM_DIR}/inkling-server" <<EOF
#!/usr/bin/env bash
exec "${INSTALL_DIR}/scripts/start_server.sh" "\$@"
EOF
  chmod +x "${SHIM_DIR}/inkling-server"
  ok "Installed launcher command: ${SHIM_DIR}/inkling-server"
fi

# 8. Optional offline test suite
if (( RUN_TESTS == 1 )); then
  info "Running offline test suite (pytest)..."
  .venv/bin/pytest
  ok "All offline tests passed"
fi

printf "\n%sInstallation complete!%s\n\n" "${C_GREEN}${C_BOLD}" "${C_RESET}"
printf "Start the server:\n"
printf "  %s%s/scripts/start_server.sh%s\n" "${C_BOLD}" "${INSTALL_DIR}" "${C_RESET}"
if [[ -x "${SHIM_DIR}/inkling-server" ]]; then
  printf "  %s# or (if ~/.local/bin is in PATH):%s\n" "${C_DIM}" "${C_RESET}"
  printf "  %sinkling-server%s\n" "${C_BOLD}" "${C_RESET}"
fi
printf "\nSwitch model variant anytime:\n"
printf "  %sMODEL_VARIANT=12.8GB %s/scripts/start_server.sh --restart%s\n" "${C_DIM}" "${INSTALL_DIR}" "${C_RESET}"
printf "\nOpenAI-compatible endpoint (first start downloads the %s model from Hugging Face):\n" "${VARIANT}"
printf "  %shttp://127.0.0.1:%s/v1%s  (model id: %sinkling-2-mini%s)\n\n" "${C_CYAN}" "${PORT}" "${C_RESET}" "${C_BOLD}" "${C_RESET}"

if (( START_AFTER == 1 )); then
  info "Starting inkling-mlx-serve..."
  exec "${INSTALL_DIR}/scripts/start_server.sh" --restart
fi
