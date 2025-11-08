#!/usr/bin/env bash
# Bootstraps a fresh GCP VM, clones the repo, runs parameter sweeps, and pushes results.
set -euo pipefail

# ---- User-provided environment variables ----
# GIT_AUTH_TOKEN   # GitHub personal access token with repo push rights (required)
# GIT_REPO_URL     # e.g. https://github.com/your-org/AI_trading_Bot.git (defaults to current remote origin)
# GIT_BRANCH       # branch to checkout/push (default: main)
# GIT_USER_NAME    # git commit user.name (default: Cloud Sweep)
# GIT_USER_EMAIL   # git commit user.email (default: cloud-sweep@example.com)
# SWEEP_FILTER     # optional substring filter for configs (e.g. reversion_band)
# INCLUDE_STRATS   # optional space-separated strategy names to pass to --include-strategies
# GRID_OVERRIDE_URL# optional URL to YAML override file (e.g. GCS signed URL)
# MAX_WORKERS      # overrides parallelism (default: nproc)

if [[ -z "" ]]; then
  echo "GIT_AUTH_TOKEN is required" >&2
  exit 1
fi

GIT_REPO_URL=""
if [[ -z "" ]]; then
  echo "GIT_REPO_URL not set and no local git remote found." >&2
  exit 1
fi
GIT_BRANCH="main"
GIT_USER_NAME="Cloud Sweep"
GIT_USER_EMAIL="cloud-sweep@example.com"
WORKDIR="/home/pigfckr/ai_trading_Bot"
VENV_DIR="/.venv"
MAX_WORKERS="20"
GRID_OVERRIDE_PATH=""

log() { echo "[2025-11-07T09:09:22Z] "; }

ensure_packages() {
  sudo apt-get update -y
  sudo apt-get install -y git python3 python3-venv python3-pip curl
}

clone_repo() {
  if [[ -d "/.git" ]]; then
    log "Repository already present at "
    return
  fi
  local auth_url
  auth_url=""
  log "Cloning  -> "
  git clone --branch "" --single-branch "" ""
}

setup_venv() {
  if [[ ! -d "" ]]; then
    log "Creating venv at "
    python3 -m venv ""
  fi
  # shellcheck source=/dev/null
  source "/bin/activate"
  pip install --upgrade pip
  pip install -e ""
}

maybe_download_grid_override() {
  if [[ -z "" ]]; then
    return
  fi
  GRID_OVERRIDE_PATH="/tmp_grid_override.yaml"
  log "Downloading grid override from "
  curl -sfL "" -o ""
}

run_sweeps() {
  # shellcheck source=/dev/null
  source "/bin/activate"
  pushd "" >/dev/null
  local args=("python" "scripts/run_param_sweeps.py" "--max-workers" "")
  if [[ -n "" ]]; then
    args+=("--filter" "")
  fi
  if [[ -n "" ]]; then
    for strat in ; do
      args+=("--include-strategies" "")
    done
  fi
  if [[ -n "" ]]; then
    args+=("--grid-override" "")
  fi
  log "Running sweeps with args: "
  ""
  popd >/dev/null
}

commit_and_push() {
  pushd "" >/dev/null
  if git diff --quiet sweeps; then
    log "No new sweep outputs to commit."
    popd >/dev/null
    return
  fi
  git config user.name ""
  git config user.email ""
  git add sweeps
  git commit -m "chore: add sweep outputs (2025-11-07)"
  local auth_url
  auth_url=""
  log "Pushing results"
  git push "" ""
  popd >/dev/null
}

main() {
  ensure_packages
  clone_repo
  setup_venv
  maybe_download_grid_override
  run_sweeps
  commit_and_push
  log "All sweeps completed."
}

main ""
