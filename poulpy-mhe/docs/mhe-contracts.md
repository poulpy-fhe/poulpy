# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds one trait per PAT type with its operations, `api::public_key`
the collective public key protocol, `api::evaluation_key` the collective
switching and automorphism key protocols, `api::tensor_key` the collective
tensor key protocol and `api::ggsw` the collective GGSW protocol. A protocol
trait, named
`*MHEProtocol`, holds `mhe_*_share_gen`, `mhe_*_share_aggregate` and
`mhe_*_share_finalize` on the protocol's share type; the prefix keeps them apart
from core's operations. All are re-exported by `api` and the crate root. Every
operation that takes caller scratch has a matching `_tmp_bytes` query in the
same trait.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `GLWEPatCompressedOps` | `GLWEPatCompressedImpl` | `reference::GLWEPatCompressedReference` |
| `GGLWEPatCompressedOps` | `GGLWEPatCompressedImpl` | `reference::GGLWEPatCompressedReference` |
| `GGLWEPatOps` | `GGLWEPatImpl` | `reference::GGLWEPatReference` |
| `GLWEPublicKeyMHEProtocol` | `GLWEPublicKeyMHEProtocolImpl` | `reference::GLWEPublicKeyMHEProtocolReference`; aggregation and finalization are derived defaults |
| `GLWESwitchingKeyMHEProtocol` | `GLWESwitchingKeyMHEProtocolImpl` | `reference::GLWESwitchingKeyMHEProtocolReference`; aggregation and finalization are derived defaults |
| `GLWEAutomorphismKeyMHEProtocol` | `GLWEAutomorphismKeyMHEProtocolImpl` | `reference::GLWEAutomorphismKeyMHEProtocolReference`; aggregation and finalization are derived defaults |
| `GLWETensorKeyMHEProtocol` | `GLWETensorKeyMHEProtocolImpl` | `reference::GLWETensorKeyMHEProtocolReference`; aggregation and finalization are derived defaults over `GGLWEPatImpl` |
| `GGSWMHEProtocol` | `GGSWMHEProtocolImpl` | `reference::GGSWMHEProtocolReference`; aggregation is a derived default over `GGLWEPatCompressedImpl` |

## Normalization

Aggregation adds limbs without normalizing; finalization, the only
normalization, produces canonical digits. Headroom for chains of additions
follows the
[radix failure estimates](../../docs/base2k-failure-probability.md).

## Share metadata

Public key shares carry the distribution of the secret they were generated
with, which the finalized key draws its ephemerals from, as core's key takes
its secret's. Switching key shares carry the input and output degrees,
automorphism key shares the Galois element, as core's compressed keys do.
Aggregation asserts that both shares carry the same metadata, and finalization
copies it into the key.

## Randomness and seeds

A public mask seed is common to all parties contributing to one result.
Use a fresh seed for every result, separated by protocol, session and key
identity (including the Galois element). Derive separate seeds from the CRS
for a key set. In particular, switching keys for different input secrets and
one output secret must have different seeds: subtracting same-mask bodies
reveals the gadget-scaled input-secret difference plus small error. Public
key generation derives a distinct seed for each of the `rank` entries from the
common seed; finalization rejects entries that share one, since common masks
would make public-key encryption rank 1 in its ephemerals. GGSW generation
derives its sub-seeds and intentionally shares each circular column's mask
between its two halves; its finalized ephemeral key may be reused under the
GGSW conditions below.

Private `source_xe` and `source_xu` streams must be independently
seeded for each party and purpose, kept secret and consumed without replay.
Never initialize a private stream from a public mask seed. An advancing error
stream can supply successive fresh samples.

## Tensor key shares

A tensor key share is a `GGLWEPat` laid out as core's `GLWETensorKey`: every
entry is an encryption of zero under the collective public key with a component
of the party's secret added to its masks. Its masks are sums, so finalization
normalizes every column. The public key must be at least as precise as the
share.

## Collective GGSW

A GGSW share holds seeded GGLWE PATs: column 0 transcribes a seeded encryption
of the party's message under its secret, and every column `j >= 1` two seeded
halves over common masks, an encryption of the message under the party's
ephemeral secret (rank 1) and an encryption of zero under component `j` of its
secret. Finalization takes the ephemeral key, the collective switching key
from the sum of the ephemeral secrets to the ideal secret built with
`GLWESwitchingKeyMHEProtocol`, prepared: an entry of column `j` is the key
switch of the negated second half plus the first half in mask column `j`, and
decrypts to the message times component `j` of the ideal secret. Column 0 has
seeded masks, so only its bodies are normalized. One ephemeral key serves
every GGSW of a key set. Every GGSW needs its own seed and the ephemeral key a
seed distinct from all of them: shares over the same masks and the same
ephemeral secret reveal the difference of their messages. The ephemeral secret
must be freshly sampled, independent of the party's secret, and kept as
private as it: with `u_i = s_i`, the two halves of a column reveal the message.
The ephemeral key's gadget (`dnum * dsize * base2k`) must cover the GGSW
precision; one guard digit (`k_aux >= base2k + log2 n`) keeps its noise far
below the circular term.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, which covers every PAT type,
`impl_mhe_public_key_reference!`, `impl_mhe_evaluation_key_reference!`,
`impl_mhe_tensor_key_reference!` or `impl_mhe_ggsw_reference!` alone when
replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_finalize_tmp_bytes`, and each
protocol's `mhe_*_share_gen_tmp_bytes` and `mhe_*_share_finalize_tmp_bytes`. A
replacement that needs more workspace replaces the matching query.
