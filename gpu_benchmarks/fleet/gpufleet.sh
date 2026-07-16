#!/bin/sh
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${GPUFLEET_PYTHON:-}" ]; then
  "$GPUFLEET_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
    || { echo "GPUFLEET_PYTHON must be Python 3.11 or newer" >&2; exit 2; }
  exec "$GPUFLEET_PYTHON" -m gpufleet "$@"
fi

python=""
for candidate in python3.13 python3.12 python3.11 python3; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
    >/dev/null 2>&1 || continue
  python="$candidate"
  break
done

if [ -z "$python" ]; then
  echo "gpufleet requires Python 3.11 or newer" >&2
  exit 2
fi
exec "$python" -m gpufleet "$@"
