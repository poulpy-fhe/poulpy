# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds one trait per PAT type with its operations, `api::public_key`
the collective public key protocol. Both are re-exported by `api` and the crate
root. Every operation that takes caller scratch has a matching `_tmp_bytes`
query in the same trait.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `GLWEPatCompressedOps` | `GLWEPatCompressedImpl` | `reference::GLWEPatCompressedReference` |
| `GGLWEPatCompressedOps` | `GGLWEPatCompressedImpl` | `reference::GGLWEPatCompressedReference` |
| `GGLWEPatOps` | `GGLWEPatImpl` | `reference::GGLWEPatReference` |
| `GLWEPublicKeyShare` | `GLWEPublicKeyShareImpl` | `reference::GLWEPublicKeyShareReference`; finalization is a derived default |

## Normalization

Aggregation adds limbs without normalizing; normalization and finalization
produce canonical digits. Headroom for chains of additions follows the
[radix failure estimates](../../docs/base2k-failure-probability.md).

## Randomness and seeds

A public mask seed is common to all parties contributing to one result.
Use a fresh seed for every result, separated by protocol, session and key
identity. Derive separate seeds from the CRS
for a key set. Each of
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
`impl_mhe_pat_reference!`, which covers the three PAT types, or
`impl_mhe_public_key_reference!` alone when replacing the other one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_normalize_tmp_bytes` and
`*_finalize_tmp_bytes`, `glwe_public_key_share_tmp_bytes` and
`glwe_public_key_finalize_tmp_bytes`. A replacement that needs more workspace
replaces the matching query.
