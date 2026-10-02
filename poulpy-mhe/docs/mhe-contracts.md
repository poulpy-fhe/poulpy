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

A share carries the metadata of the core object it finalizes into, as listed
on each protocol trait. Aggregation asserts that both shares carry the same
metadata, and finalization copies it into the result.

## Randomness and seeds

A public mask seed is common to all parties contributing to one result. Use a
fresh seed for every result, separated by protocol, session and key identity;
choosing distinct seeds is the caller's responsibility. Derive separate seeds
from the CRS for a key set. Each protocol trait states what its seed derives
and what reusing it reveals. Choosing the secret distribution, which public-key
encryption also draws its ephemerals from, is the caller's responsibility too.

Private `source_xe` and `source_xu` streams must be independently seeded for
each party and purpose, kept secret and consumed without replay. Never
initialize a private stream from a public mask seed. An advancing error stream
can supply successive fresh samples.

## Threat model

The protocols are secure against passive (semi-honest) adversaries: when every
party follows the protocol, any coalition of all parties but one learns nothing
about the ideal secret beyond the finalized outputs. Nothing is proven or
authenticated:

- Aggregation adds whatever shares it receives. A malicious party can bias the
  result or cancel the other shares, for instance by sending the target minus
  the others' sum, which makes a collective key one it alone holds.
- Inputs are trusted. Keys and ciphertexts passed to a protocol must be the
  honestly finalized collective objects of the session; a share encrypted
  under another key reveals its secret part to that key's holder.
- `read_from` checks a share's layout, not its provenance.
- Share generation zeroes the scratch it was given before returning, as
  core's secret-handling operations do.

Active security requires commitments or proofs on the shares, outside this
crate.

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
