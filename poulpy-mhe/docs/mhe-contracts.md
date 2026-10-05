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
`api::ggsw` the collective GGSW protocol and `api::sharing` the
encryption-to-shares and shares-to-encryption protocols. `ckks` holds the
CKKS-specific protocols, layered the same way: `ckks::api::refresh` the
collective CKKS refresh protocol. A protocol
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
| `ckks::CKKSRefreshMHEProtocol` | `ckks::oep::CKKSRefreshMHEProtocolImpl` | `ckks::reference::CKKSRefreshMHEProtocolReference` |

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

Collective key switching and generic encryption-to-shares add a flood to every
share: a `SmudgingNoise`, either a discrete Gaussian with an explicit
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

## CKKS refresh

`CKKSRefreshMHEProtocol` uses private integer masks `M_i` and ordinary noise.
Its encryption-to-shares part is `d_i = <a, s_i> - M_i + e_i` modulo the input
modulus `q = 2^k`; its shares-to-encryption part is
`r_i = -<A, s_i> + M_i + e'_i` modulo the output modulus `Q >= q`. The fresh
common seed determines `A`. Every `e_i` is sampled automatically with sigma
3.2 and bound `6 * 3.2` on the input precision grid. Each `e'_i` uses the
caller-selected encryption distribution on the output precision grid.
Successive fresh child seeds from the private `source_xe` supply independent
errors for the two parts; `source_xm` remains independent of `source_xe`.
The errors are added only to the public parts, leaving `M_i` unchanged.

The public opening is `t = m + e + sum(e_i) - sum(M_i)`, where `e` is the
input error. The masks cancel during re-encryption, giving a ciphertext of
`m + e + sum(e_i) + sum(e'_i)`. Ordinary noise must remain: combining the two
public parts modulo `q` cancels `M_i` and leaves
`<a - A, s_i> + e_i + e'_i`, which would be an exact secret-key equation if
both errors were removed. Encryption parameters must provide the intended
RLWE/GLWE security; sigma 3.2 alone does not establish a security level.

The masks statistically hide the public opening without an additional flood.
Let `B` bound the coefficients of `m + e + sum(e_i)`, including
all input encryption, evaluation and rounding errors. With `n` coefficients
and independent masks uniform over `log_bound` bits, one honest party's mask
hides the opening within statistical distance `n * B / 2^log_bound`. Require
`log_bound >= log2(B) + log2(n) + lambda` for a per-call margin `lambda`, and
budget repeated calls and error tails over the whole transcript. The
integer-preserving raise also requires no wrap; the sufficient bound
`B + parties * 2^log_bound < 2^(k - 1)` applies coefficientwise. The caller
must establish these bounds; the protocol knows neither `B` nor the party count.

In the passive model, the whole transcript can be simulated from the input and
the actual output ciphertexts and the corrupt parties' state. With only one
honest party, its mask cancels from the output and remains independent of it.
Sample a statistically indistinguishable masked opening, then derive that
party's decryption share from the input and its re-encryption share from the
output. This accounts for the correlation between the two public parts and
does not require an additional smudging flood. The construction follows the
ordinary-noise masked refresh in
[POSEIDON, Protocol 4 and Appendix B](https://www.dpss.inesc-id.pt/~ler/docencia/atpds2021/papers/poseidon.pdf).

This guarantee is privacy beyond the actual encrypted output. Refresh preserves
the input error and does not sanitize it for disclosure. A later release of an
approximate decryption, or a requirement to hide the prior error from the output
recipient, needs separately sized flooding or another suitable protection.

## Replacing an operation

An override must compute the same result as the reference, including its
seed and layout checks, and pass parity against a validated backend; the
parity suite arrives with the first override.
`impl_mhe_reference_full!` selects every family; select
`impl_mhe_pat_reference!`, which covers every PAT type,
`impl_mhe_public_key_reference!`, `impl_mhe_evaluation_key_reference!`,
`impl_mhe_tensor_key_reference!`, `impl_mhe_ggsw_reference!`, `impl_mhe_keyswitch_reference!`,
`impl_mhe_sharing_reference!` or `impl_mhe_ckks_refresh_reference!` alone when
replacing another one. The
reference traits stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_finalize_tmp_bytes`, and each
protocol's `mhe_*_share_gen_tmp_bytes` and `mhe_*_share_finalize_tmp_bytes`. A
replacement that needs more workspace replaces the matching query.
