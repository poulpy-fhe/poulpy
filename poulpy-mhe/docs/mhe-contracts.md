# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds the operations every PAT shape shares, `api::public_key` the
collective public key protocol, `api::evaluation_key` the collective
switching and automorphism key protocols, `api::keyswitch` the collective key
switching protocols and `api::tensor_key` the collective tensor key protocol.
All are re-exported by `api` and the crate root.
Every operation that takes caller scratch has a matching `_tmp_bytes` query in
the same trait.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `PatAggregate` | `PatAggregateImpl` | `reference::PatAggregateReference` |
| `PatNormalize` | `PatNormalizeImpl` | `reference::PatNormalizeReference` |
| `PatFinalize` | `PatFinalizeImpl` | `reference::PatFinalizeReference` |
| `GLWEPublicKeyShare` | `GLWEPublicKeyShareImpl` | `reference::GLWEPublicKeyShareReference`; finalization is a derived default |
| `GLWESwitchingKeyShare` | `GLWESwitchingKeyShareImpl` | `reference::GLWESwitchingKeyShareReference`; aggregation, normalization and finalization are derived defaults |
| `GLWEAutomorphismKeyShare` | `GLWEAutomorphismKeyShareImpl` | `reference::GLWEAutomorphismKeyShareReference`; aggregation, normalization and finalization are derived defaults |
| `GLWEKeyswitchShare` | `GLWEKeyswitchShareImpl` | `reference::GLWEKeyswitchShareReference` |
| `GLWEPublicKeyswitchShare` | `GLWEPublicKeyswitchShareImpl` | `reference::GLWEPublicKeyswitchShareReference` |
| `GLWETensorKeyShare` | `GLWETensorKeyShareImpl` | `reference::GLWETensorKeyShareReference`; aggregation, normalization and finalization are the `GGLWEPat` operations |

## Canonical flag

Aggregation adds limbs without normalizing and clears the PAT's canonical
flag. Normalization and finalization restore canonical digits; `write_to`
refuses a PAT whose flag is clear. Headroom for chains of additions follows
the [radix failure estimates](../../docs/base2k-failure-probability.md).

## Key metadata

Switching key shares carry the input and output degrees, automorphism key
shares the Galois element, as core's compressed keys do. Aggregation asserts
that both shares carry the same metadata, and finalization copies it into the
key.

## Key switching shares

Key switching shares are core `GLWE`s: aggregate them with core
`glwe_add_assign` and normalize them with core `glwe_normalize_assign`, which
track core's canonical flag. A share carries the party's smudging noise,
drawn with the `flood` noise parameters, so that the aggregate reveals
nothing about the parties' secrets beyond the switched ciphertext, provided
the `flood` sigma is large compared with the input ciphertext's noise.
`GLWEPublicKeyswitchShare` needs a public key at least as precise as the
share.

## Tensor key shares

A tensor key share is a `GGLWEPat` laid out as the tensor key: every entry is
an encryption of zero under the collective public key with a component of the
party's secret added to its masks. Aggregate, normalize and finalize the
shares with the `GGLWEPat` operations of `PatAggregate`, `PatNormalize` and
`PatFinalize`; the finalized key is a core `GLWETensorKey` of the ideal
secret. The public key must be at least as precise as the share.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, `impl_mhe_public_key_reference!`,
`impl_mhe_evaluation_key_reference!`, `impl_mhe_keyswitch_reference!` or
`impl_mhe_tensor_key_reference!` alone when replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: `pat_normalize_tmp_bytes`,
`pat_finalize_tmp_bytes`, `glwe_public_key_share_tmp_bytes`,
`glwe_public_key_finalize_tmp_bytes`, and the share, normalize and finalize
queries of `GLWESwitchingKeyShare` and `GLWEAutomorphismKeyShare`, and the
share and finalize queries of `GLWEKeyswitchShare` and
`GLWEPublicKeyswitchShare`, and the share query of `GLWETensorKeyShare`. A
replacement that needs more workspace replaces the matching query.
