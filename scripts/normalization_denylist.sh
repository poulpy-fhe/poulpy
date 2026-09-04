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
    count=$(grep -rn --include='*.rs' -F "$pattern" poulpy-*/src poulpy-*/examples 2>/dev/null | wc -l || true)
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

check "into_unnormalized"    143 "owned-root relabel; migrate to Unwritten scratch roots (spec §6.4)"
check ".normalize("           23 "receiver normalization; migrate to module out-of-place normalize (spec PR 3/5/6)"
check "set_normalized("        9 "unsafe OEP relabel; backend fused kernels only (spec §9.2)"
check "set_unnormalized("      9 "OEP relabel; backend crates only (spec §9.2)"
check "from_data_like"         5 "state-forwarding reborrow; oep-only since PR 2 (backend kernel plumbing)"
check "map_data_mut"           3 "state-forwarding reborrow; oep-only since PR 2 (backend kernel plumbing)"
check "as_scalar_znx_mut"      0 "safe raw mutation of typed storage; deleted in PR 2 — must not return"
check "take_unnormalized_"     3 "state-forging scratch take; migrate to Unwritten takes (spec §8.1)"
check "relabel_unchecked"     10 "crate-private relabel primitive; callers must stay the normalize family, OEP, and the containment bridges"
check "from_data_with_state"  12 "crate-private stateful raw constructor; callers must stay in poulpy-hal (oep reborrows, kernel capability, from_data_unnormalized)"
check "borrowed_carry_view"   42 "TRANSITIONAL borrowed-view relabel bridges; removed by the PR 5 scratch-transaction migration — must only shrink"
check "transfer_data_mut"     10 "storage plumbing access (transfer/view construction); revisited by PR 7"
check "kernel_words_mut"      85 "sealed kernel capability uses (backend kernels + harness wrapper); audited per spec §9.1"
check "harness_words_mut"     55 "test-harness state-erased writes; production code must never call it"
check "weaken_backend_ref"     5 "safe shared-view weakening; sound, tracked for the PR 4 canonicality audit"
check "from_data_unnormalized" 4 "weakest-label raw ingestion; sound, tracked for the PR 7 Raw migration"

if [ "$fail" -ne 0 ]; then
    echo "normalization deny-list ratchet FAILED"
    exit 1
fi
echo "normalization deny-list ratchet passed"
