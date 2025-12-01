#!/usr/bin/env bash
# Helper script to turn a fresh Linux VM (e.g. OVH Ubuntu 22.04) into
# a running diachronic corpus agent.
#
# What it does:
#   - Installs base packages (python3-venv, python3-pip, git) via apt if available
#   - Creates/updates a Python virtualenv in .venv
#   - Installs Python dependencies
#   - If 111.db exists and corpus_platform.db is missing, creates a symlink
#   - Runs check_results.py for a quick sanity check
#   - Runs the full professional cycle once (run_professional_cycle.py)
#
# Usage on the VM (from the project root):
#   chmod +x deploy_on_linux_vm.sh
#   ./deploy_on_linux_vm.sh
#
# You can re-run this script safely; it is mostly idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

step() {
  echo
  echo "[deploy] $1"
  echo "------------------------------------------------------------"
}

step "1/5 Install base packages (may ask for sudo password)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip git
else
  echo "apt-get not found; please install Python 3, venv and pip manually." >&2
fi

step "2/5 Create or reuse Python virtual environment (.venv)"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt || echo "[deploy] requirements.txt install had some errors; will continue."
fi

# Extra libraries commonly used in this project
pip install fastapi uvicorn stanza lxml jinja2 || echo "[deploy] Extra library install had some errors; will continue."

step "3/5 Ensure database alias (corpus_platform.db -> 111.db if needed)"
if [ -f "111.db" ] && [ ! -e "corpus_platform.db" ]; then
  ln -s 111.db corpus_platform.db
  echo "[deploy] Created symlink corpus_platform.db -> 111.db"
else
  echo "[deploy] Using existing corpus_platform.db (or no DB present yet)."
fi

step "4/5 Quick corpus status check (check_results.py)"
if [ -f "check_results.py" ]; then
  python check_results.py || echo "[deploy] check_results.py failed (see above); continuing anyway."
else
  echo "[deploy] check_results.py not found; skipping status check."
fi

step "5/5 Run full professional research cycle (overnight agent)"
if [ -f "run_professional_cycle.py" ]; then
  echo "[deploy] Starting run_professional_cycle.py ..."
  python run_professional_cycle.py
  echo "[deploy] run_professional_cycle.py finished."
else
  echo "[deploy] run_professional_cycle.py not found; nothing to run."
fi

echo
echo "[deploy] All steps completed. For continuous runs, you can schedule this script via cron or systemd timer on the VM."
