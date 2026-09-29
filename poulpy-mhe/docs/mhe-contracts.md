# Multiparty operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend opts
into the reference implementation with the `impl_mhe_*_reference!` macros or
implements a contract itself. The reference composes `poulpy-core` and
`poulpy-hal` operations; same-layer compositions are crate-private derived
defaults in `oep::derived`.

## Public API organization

`api::pat` holds one trait per PAT type with its operations, `api::public_key`
the collective public key protocol, `api::evaluation_key` the collective
switching and automorphism key protocols, `api::keyswitch` the collective key
switching protocols, `api::tensor_key` the collective tensor key protocol,
`api::ggsw` the collective GGSW protocol, `api::sharing` the
encryption-to-shares and shares-to-encryption protocols and `api::refresh` the
collective refresh protocol. A protocol
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
| `GLWEPrivateKeyswitchMHEProtocol` | `GLWEPrivateKeyswitchMHEProtocolImpl` | `reference::GLWEPrivateKeyswitchMHEProtocolReference` |
| `GLWEPublicKeyswitchMHEProtocol` | `GLWEPublicKeyswitchMHEProtocolImpl` | `reference::GLWEPublicKeyswitchMHEProtocolReference` |
| `GLWEEncToShareMHEProtocol` | `GLWEEncToShareMHEProtocolImpl` | `reference::GLWEEncToShareMHEProtocolReference` |
| `GLWEShareToEncMHEProtocol` | `GLWEShareToEncMHEProtocolImpl` | `reference::GLWEShareToEncMHEProtocolReference`; aggregation and finalization are derived defaults over `GLWEPatCompressedImpl` |
| `GLWERefreshMHEProtocol` | `GLWERefreshMHEProtocolImpl` | `reference::GLWERefreshMHEProtocolReference` |

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

Private `source_xe`, `source_xu`, `source_xm` and `source_smudge` streams must
be independently seeded for each party and purpose, kept secret and consumed
without replay. Never initialize a private stream from a public mask seed. An
advancing error stream can supply successive fresh samples.

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
  core's secret-handling operations do. Encryption-to-shares finalization
  also zeroes its scratch after normalizing the private additive share.

Active security requires commitments or proofs on the shares, outside this
crate.

## Smudging

A protocol whose share is a function of the parties' secrets adds a flood to
every share: a `SmudgingNoise`, either a discrete Gaussian with an explicit
cutoff or a uniform distribution on consecutive integers. The flood is sampled
on the precision grid of the value it hides, which each protocol trait names,
so that it reaches its bottom bit; on a coarser grid the low bits would be
exact linear equations in the secrets.

Each party provisions its own flood, so that its shares hide its secret
whichever other parties are corrupt. For a statistical margin `lambda`, the
flood must dominate the input encryption, evaluation and rounding errors by
`2^lambda`, in units of the sampling grid, over the whole transcript:

- For a fixed integer discrepancy vector `e`, the untruncated Gaussian shifts
  by at most `||e||_2 / (2 sigma)` in statistical distance, the uniform
  distribution by at most `||e||_1 / 2^bits`.
- A cutoff omits at most `tau = 2 exp(-cutoff^2 / 2)` per coefficient, so
  comparing two worlds of `M` conditioned samples costs at most `2 M tau`.
  Choose it for the whole transcript, separately from the shift budget.
- Random input-error tails and repeated or adaptive calls need their own
  bounds.

Correctness needs the sum of every party's flood bound, `cutoff * 2^log_sigma`
or `2^(bits-1)`, the input and conversion errors and any fresh encryption noise
to fit the decoding margin. The protocols check the flood against the sampling
precision before drawing randomness. They cannot infer the input error or
certify a security level, and neither can statistical tests.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, which covers every PAT type,
`impl_mhe_public_key_reference!`, `impl_mhe_evaluation_key_reference!`,
`impl_mhe_tensor_key_reference!`, `impl_mhe_ggsw_reference!`, `impl_mhe_keyswitch_reference!`,
`impl_mhe_sharing_reference!` or `impl_mhe_refresh_reference!` alone when
replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_finalize_tmp_bytes`, and each
protocol's `mhe_*_share_gen_tmp_bytes` and `mhe_*_share_finalize_tmp_bytes`. A
replacement that needs more workspace replaces the matching query.
