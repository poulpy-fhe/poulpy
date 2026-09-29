# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds one trait per PAT type with its operations, `api::public_key`
the collective public key protocol and `api::evaluation_key` the collective
switching and automorphism key protocols. All are re-exported by `api` and the crate root.
Every operation that takes caller scratch has a matching `_tmp_bytes` query in
the same trait.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `GLWEPatCompressedOps` | `GLWEPatCompressedImpl` | `reference::GLWEPatCompressedReference` |
| `GGLWEPatCompressedOps` | `GGLWEPatCompressedImpl` | `reference::GGLWEPatCompressedReference` |
| `GGLWEPatOps` | `GGLWEPatImpl` | `reference::GGLWEPatReference` |
| `GLWEPublicKeyShare` | `GLWEPublicKeyShareImpl` | `reference::GLWEPublicKeyShareReference`; finalization is a derived default |
| `GLWESwitchingKeyShare` | `GLWESwitchingKeyShareImpl` | `reference::GLWESwitchingKeyShareReference`; aggregation, normalization and finalization are derived defaults |
| `GLWEAutomorphismKeyShare` | `GLWEAutomorphismKeyShareImpl` | `reference::GLWEAutomorphismKeyShareReference`; aggregation, normalization and finalization are derived defaults |

## Normalization

Aggregation adds limbs without normalizing; normalization and finalization
produce canonical digits. Headroom for chains of additions follows the
[radix failure estimates](../../docs/base2k-failure-probability.md).

## Key metadata

Switching key shares carry the input and output degrees, automorphism key
shares the Galois element, as core's compressed keys do. Aggregation asserts
that both shares carry the same metadata, and finalization copies it into the
key.

## Randomness and seeds

A public mask seed is common to all parties contributing to one result.
Use a fresh seed for every result, separated by protocol, session and key
identity (including the Galois element). Derive separate seeds from the CRS
for a key set. In particular, switching keys for different input secrets and
one output secret must have different seeds: subtracting same-mask bodies
reveals the gadget-scaled input-secret difference plus small error. Each of
the `rank` public key entries is shared under its own seed; finalization
rejects entries that share one, since common masks would make public-key
encryption rank 1 in its ephemerals.

Private `source_xe` streams must be independently
seeded for each party and purpose, kept secret and consumed without replay.
Never initialize a private stream from a public mask seed. An advancing error
stream can supply successive fresh samples.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, which covers the three PAT types,
`impl_mhe_public_key_reference!` or `impl_mhe_evaluation_key_reference!` alone
when replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_normalize_tmp_bytes` and
`*_finalize_tmp_bytes`, `glwe_public_key_share_tmp_bytes`,
`glwe_public_key_finalize_tmp_bytes`, and the share, normalize and finalize
queries of `GLWESwitchingKeyShare` and `GLWEAutomorphismKeyShare`. A
replacement that needs more workspace replaces the matching query.
