#!/usr/bin/env bash
# Long-running, resumable wrapper for download_tushare_lake.py.
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
root="${ROOT:-data/_tushare_lake/${timestamp}}"
log="${root}/download.log"
sentinel="${root}/download.done"
exitcode="${root}/download.exitcode"
timeout_seconds="${TIMEOUT_SECONDS:-86400}"
mkdir -p "$root"

printf 'command: timeout %s pixi run python tools/download_tushare_lake.py --root %q %s\n' \
  "$timeout_seconds" "$root" "$*"
printf 'log: %s\noutput: %s\nsentinel: %s\n' "$log" "$root" "$sentinel"

set +e
timeout "$timeout_seconds" pixi run python tools/download_tushare_lake.py \
  --root "$root" "$@" >"$log" 2>&1
code=$?
set -e

printf '%s\n' "$code" >"$exitcode"
touch "$sentinel"
printf 'finished: exit=%s log=%s sentinel=%s\n' "$code" "$log" "$sentinel"
exit "$code"
