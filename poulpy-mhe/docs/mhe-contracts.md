# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds the operations every PAT shape shares, `api::public_key` the
collective public key protocol. Both are re-exported by `api` and the crate
root. Every operation that takes caller scratch has a matching `_tmp_bytes`
query in the same trait.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `PatAggregate` | `PatAggregateImpl` | `reference::PatAggregateReference` |
| `PatNormalize` | `PatNormalizeImpl` | `reference::PatNormalizeReference` |
| `PatFinalize` | `PatFinalizeImpl` | `reference::PatFinalizeReference` |
| `GLWEPublicKeyShare` | `GLWEPublicKeyShareImpl` | `reference::GLWEPublicKeyShareReference`; finalization is a derived default |

## Canonical flag

Aggregation adds limbs without normalizing and clears the PAT's canonical
flag. Normalization and finalization restore canonical digits; `write_to`
refuses a PAT whose flag is clear. Headroom for chains of additions follows
the [radix failure estimates](../../docs/base2k-failure-probability.md).

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!` or `impl_mhe_public_key_reference!` alone when
replacing the other one. The reference traits stay callable from an override.

## Workspace

Size scratch with the queries: `pat_normalize_tmp_bytes`,
`pat_finalize_tmp_bytes`, `glwe_public_key_share_tmp_bytes` and
`glwe_public_key_finalize_tmp_bytes`. A replacement that needs more workspace
replaces the matching query.
