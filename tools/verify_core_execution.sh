#!/usr/bin/env bash
# Verify compiler-enumerated core contracts against a completed libtest run.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <backend-package> <test-log>" >&2
    exit 2
fi
core_package="$1"
core_test_log="$2"
cargo rustdoc -p poulpy-core --lib --features enable-core -- \
    --document-private-items --document-hidden-items -Z unstable-options --output-format json
core_target_dir="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
core_target_prefix="${CARGO_BUILD_TARGET:+$CARGO_BUILD_TARGET/}"
python3 tools/check_core_contracts.py \
    --rustdoc-json "$core_target_dir/${core_target_prefix}doc/poulpy_core.json" \
    --run-package "$core_package" "$core_test_log"
