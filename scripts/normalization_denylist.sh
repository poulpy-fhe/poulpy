#!/usr/bin/env bash
# Normalization-migration deny-list ratchet (docs/spec/normalization_typestate.md, PR 0).
#
# Each pattern below is a bypass of the coefficient-typestate design that the migration
# must shrink, never grow. The recorded number is the baseline occurrence count at the
# time PR 0 landed. CI fails if a pattern's count EXCEEDS its baseline (a new bypass was
# introduced). When migration work removes sites, lower the baseline here in the same PR
# so the ratchet locks in the progress.
#
# Counting rule: fixed-string grep over poulpy-*/src and poulpy-*/examples, one count per
# matching line (definitions included, so the numbers are stable and trivially
# reproducible by hand).

set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

check() {
    local pattern="$1" baseline="$2" why="$3"
    local count
    count=$(grep -rn --include='*.rs' -F "$pattern" poulpy-*/src poulpy-*/examples 2>/dev/null | wc -l)
    if [ "$count" -gt "$baseline" ]; then
        echo "DENY-LIST: '$pattern' has $count occurrences (baseline $baseline): $why"
        grep -rn --include='*.rs' -F "$pattern" poulpy-*/src poulpy-*/examples 2>/dev/null | tail -5
        fail=1
    elif [ "$count" -lt "$baseline" ]; then
        echo "ratchet: '$pattern' dropped to $count (baseline $baseline) — lower the baseline in scripts/normalization_denylist.sh"
    else
        echo "ok: '$pattern' = $count"
    fi
}

check "into_unnormalized"    184 "free relabel to Unnormalized; migrate to Unwritten scratch roots (spec §6.4)"
check ".normalize("           23 "receiver normalization; migrate to module out-of-place normalize (spec PR 3/5/6)"
check "set_normalized("        9 "unsafe OEP relabel; backend fused kernels only (spec §9.2)"
check "set_unnormalized("      6 "OEP relabel; backend crates only (spec §9.2)"
check "from_data_like"         5 "state-forwarding raw constructor; to be closed (spec PR 2)"
check "map_data_mut"           3 "state-forwarding raw mapping; to be closed (spec PR 2)"
check "as_scalar_znx_mut"      1 "safe raw mutation of typed storage; to be deleted (spec PR 2)"
check "take_unnormalized_"     3 "state-forging scratch take; migrate to Unwritten takes (spec §8.1)"
check "relabel_unchecked"      7 "crate-private relabel primitive; callers must stay the normalize family + OEP"
check "from_data_with_state"   8 "crate-private stateful raw constructor; callers must stay in poulpy-hal"

if [ "$fail" -ne 0 ]; then
    echo "normalization deny-list ratchet FAILED"
    exit 1
fi
echo "normalization deny-list ratchet passed"
