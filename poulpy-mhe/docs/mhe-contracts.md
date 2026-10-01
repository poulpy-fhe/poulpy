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
switching protocols, `api::tensor_key` the collective tensor key protocol and
`api::ggsw` the collective GGSW protocol. A protocol
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
| `GLWEKeyswitchMHEProtocol` | `GLWEKeyswitchMHEProtocolImpl` | `reference::GLWEKeyswitchMHEProtocolReference` |
| `GLWEPublicKeyswitchMHEProtocol` | `GLWEPublicKeyswitchMHEProtocolImpl` | `reference::GLWEPublicKeyswitchMHEProtocolReference` |

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
every GGSW of a key set, so a party reuses its ephemeral secret for its key
share and every GGSW share. Every GGSW needs its own seed and the ephemeral key a
seed distinct from all of them: shares over the same masks and the same
ephemeral secret reveal the difference of their messages. The ephemeral secret
must be freshly sampled, independent of the party's secret, and kept as
private as it: with `u_i = s_i`, the two halves of a column reveal the message.
The ephemeral key's gadget (`dnum * dsize * base2k`) must cover the GGSW
precision; one guard digit (`k_aux >= base2k + log2 n`) keeps its noise far
below the circular term.

## Key switching shares

A key switching share wraps one core `GLWE`: rank 0 for a switch to a secret
key, the output public key's rank for a switch to a public key. It carries the
party's smudging noise, drawn with the `flood` noise parameters, so that the
aggregate reveals nothing about the parties' secrets beyond the switched
ciphertext when the smudging contract below is met. Aggregation adds the
shares; finalization adds the ciphertext body and normalizes.
`GLWEPublicKeyswitchMHEProtocol` needs a public key at least as precise as the
share.

## Smudging parameters

Secret and public key switching
require `flood: &impl SmudgingInfos`.
`SmudgingNoise` provides two full-width distributions:

- `SmudgingNoise::gaussian(k, log_sigma, cutoff)` samples an integer discrete
  Gaussian with mass proportional to `exp(-z^2 / (2 sigma^2))`, where
  `sigma = 2^log_sigma`, conditioned on `|z| <= cutoff * sigma`.
- `SmudgingNoise::uniform(k, bits)` samples exactly uniformly from
  `[-2^(bits-1), 2^(bits-1)-1]`. This interval has mean `-1/2`.

Both add `z * 2^-k`. The CPU sampler generates a complete integer per
coefficient and decomposes it across balanced limbs, preserving all low bits.
The Gaussian implementation uses the exact integer rejection algorithm of
[Canonne, Kamath and Steinke, section 5](https://arxiv.org/pdf/2004.00010#page=30).
It has variable runtime and allocates scalar big integers. There is no
floating-point sampling, machine-word truncation or fixed retry limit.
The sampler's exact-distribution claim assumes independent uniform source bits;
using private seeded streams adds the source's computational security assumption.
An observable sampling-time side channel is outside this distribution claim.

Set `k` to the sampling destination's precision to hide errors on its full
integer grid. A smaller `k` leaves a coarser sampling lattice and may expose
low error bits. Sampling precision is `res.k` for secret-key switching,
and `ct.k` for public-key switching.
The API checks this precision, distribution bounds and coefficient headroom
before sampling. It cannot infer the input error or certify statistical hiding.

For a statistical hiding margin `lambda`, each party must provision its own
flood; the model allows every other party to be corrupt. The Gaussian scale
must dominate the input encryption, evaluation and rounding noise by at least
`2^lambda`, in common units, with dimensions and the full transcript accounted
for. More explicitly, for a fixed integer discrepancy vector `e`, the
untruncated product Gaussian has shift distance at most `||e||_2 / (2 sigma)`.
The uniform distribution has shift distance at most `||e||_1 / 2^bits`.
Thus, for `M` shifted coordinates bounded by `H`, uniform width satisfying
`2^bits >= 2^lambda * M * H` budgets the shift term. Random input-error tails
and repeated or adaptive calls require their own bounds in the security proof.

A Gaussian cutoff omits probability at most `tau = 2 exp(-cutoff^2 / 2)` per
coefficient. Conditioning `M` samples changes their distribution by at most
`M * tau`; comparing two conditioned worlds can cost `2 * M * tau`.
Choose the cutoff for the complete transcript, separately from its shift
budget. Statistical tests cannot
certify a `2^-128` tail or shift bound.

Correctness needs the sum of all parties' absolute flood bounds, input and
conversion errors, and fresh encryption noise to fit the decoding margin.
The integer bounds are `cutoff * 2^log_sigma` and `2^(bits-1)`, respectively.
Increasing ciphertext precision alone does not create decoding margin when
the encoded integer message and error scales remain unchanged. Lazy
aggregation still requires its ordinary coefficient headroom.

`NoiseInfos` and `EncryptionLayout` continue to describe ordinary encryption; they no longer
serve as flood descriptors. A former power-of-two sigma `2^s` with bound
`c * 2^s` migrates to `SmudgingNoise::gaussian(k, s, c)`; the new sampler uses
an exact conditional discrete Gaussian rather than a rounded real Gaussian.

Small-sigma tests check forwarding and noise presence. These
functional checks complement the distribution argument and parameter bounds.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, which covers every PAT type,
`impl_mhe_public_key_reference!`, `impl_mhe_evaluation_key_reference!`,
`impl_mhe_tensor_key_reference!`, `impl_mhe_ggsw_reference!` or
`impl_mhe_keyswitch_reference!` alone when replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_finalize_tmp_bytes`, and each
protocol's `mhe_*_share_gen_tmp_bytes` and `mhe_*_share_finalize_tmp_bytes`. A
replacement that needs more workspace replaces the matching query.
