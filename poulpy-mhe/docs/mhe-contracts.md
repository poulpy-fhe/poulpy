# Multiparty operation contracts

Public `api` traits follow the [layering contract](../../docs/guides/layering-contract.md). The references compose
`poulpy-core` and `poulpy-hal` operations.

## Public API organization

The `api` traits cover the PAT types and their operations, the collective
public key, switching and automorphism key, key switching, tensor key and GGSW
protocols, and the encryption-to-shares and shares-to-encryption protocols, all
at the single path `poulpy_mhe::api` (also re-exported at the crate root). A protocol
trait, named
`*MHEProtocol`, holds `mhe_*_share_gen`, `mhe_*_share_aggregate` and
`mhe_*_share_finalize` on the protocol's share type; the prefix keeps them apart
from core's operations. Every
operation that takes caller scratch has a matching `_tmp_bytes` query in the
same trait, and panics before any draw or write when given less.

## Operation map

| API family | OEP contract | Reference |
|---|---|---|
| `MHEGLWEPatCompressedOps` | `MHEGLWEPatCompressedImpl` | `reference::MHEGLWEPatCompressedReference` |
| `MHEGGLWEPatCompressedOps` | `MHEGGLWEPatCompressedImpl` | `reference::MHEGGLWEPatCompressedReference` |
| `MHEGGLWEPatOps` | `MHEGGLWEPatImpl` | `reference::MHEGGLWEPatReference` |
| `MHEGLWEPublicKeyProtocol` | `MHEGLWEPublicKeyProtocolImpl` | `reference::MHEGLWEPublicKeyProtocolReference`; aggregation and finalization are derived defaults |
| `MHEGLWESwitchingKeyProtocol` | `MHEGLWESwitchingKeyProtocolImpl` | `reference::MHEGLWESwitchingKeyProtocolReference`; aggregation and finalization are derived defaults |
| `MHEGLWEAutomorphismKeyProtocol` | `MHEGLWEAutomorphismKeyProtocolImpl` | `reference::MHEGLWEAutomorphismKeyProtocolReference`; aggregation and finalization are derived defaults |
| `MHEGLWETensorKeyProtocol` | `MHEGLWETensorKeyProtocolImpl` | `reference::MHEGLWETensorKeyProtocolReference`; aggregation and finalization are derived defaults over `MHEGGLWEPatImpl` |
| `MHEGGSWProtocol` | `MHEGGSWProtocolImpl` | `reference::MHEGGSWProtocolReference`; aggregation is a derived default over `MHEGGLWEPatCompressedImpl` |
| `MHEGLWEPrivateKeyswitchProtocol` | `MHEGLWEPrivateKeyswitchProtocolImpl` | `reference::MHEGLWEPrivateKeyswitchProtocolReference` |
| `MHEGLWEPublicKeyswitchProtocol` | `MHEGLWEPublicKeyswitchProtocolImpl` | `reference::MHEGLWEPublicKeyswitchProtocolReference` |
| `MHEGLWEEncToShareProtocol` | `MHEGLWEEncToShareProtocolImpl` | `reference::MHEGLWEEncToShareProtocolReference` |
| `MHEGLWEShareToEncProtocol` | `MHEGLWEShareToEncProtocolImpl` | `reference::MHEGLWEShareToEncProtocolReference`; aggregation and finalization are derived defaults over `MHEGLWEPatCompressedImpl` |

## Normalization

Generation normalizes, so fresh shares are canonical. Aggregation adds limbs
without normalizing; finalization normalizes the aggregate into canonical
digits. Headroom for chains of additions
follows the
[radix failure estimates](../../docs/guides/base2k-failure-probability.md).

## Share metadata

Outputs follow the core noise metadata rule (`poulpy_core::oep`): only the
estimate left on an output is specified. Shares, aggregates and finalized keys
record the estimates below. Invalid provenance is rejected before mutation, and
delegates only forward calls.

A share carries `ComponentNoise`: secret provenance and `rank + 1` effective
fresh component variances, ordered as body followed by masks. Each mask variance
is stored before secret weighting. The secret distribution records its base law
and number of independently summed parties. Each `FreshNoiseEstimate` records a
effective second-moment estimate and its precision: an estimate `V` at precision `k`
has torus variance `V * 2^(-2k)`. Rescaling the estimate does not change the stored
secret count. The estimate is not an assertion that the resulting distribution
is Gaussian. For the standard negacyclic ring, `phase_noise(n)` computes
`V_body + n * E[S²] * sum(V_masks)`. Conjugate-invariant public-key encryption
uses `weighted_phase_noise(n, 2*n)`, the average over coefficients.
Rank-zero shares have one body component even when their secret provenance names
a nonzero secret distribution.

Seeded share aggregation checks compatible base laws and adds independent
component variances after bringing them to the same precision, while also adding secret
party counts; an untagged share leaves the aggregate untagged. Public-key ciphertext and tensor-key shares retain the destination
secret count. Their raw variances include public-key error multiplied by new
ephemerals, fresh mask error, and body noise. Secret weighting is applied only
when computing the phase estimate.
Public-key encryption draws its fresh errors one limb past the share's
precision and normalizes once to it; `noise()` records the resulting component
estimates on the share's grid. A deliberate flood replaces the body error at the
share's precision.
For a centered base secret law the independent ephemeral contributions give
additive variances. For a noncentered law, reusing a public key correlates its
error across shares; each component uses the conservative squared sum of
standard deviations. Compatibility checks compare secret provenance, not the
changing fresh variance. The public key's ephemeral sampling law must equal its
recorded base secret law; changing that tag independently is rejected before
share generation mutates output or consumes randomness.

A GGSW's first column carries ordinary aggregated body noise. Other columns
contain `E_s * U` in the body and `E_u` in each mask, plus each component's
key-switching and rounding error. Their variance model uses the ephemeral
second moment, including nonzero means, and bounds each gadget digit of
width `B = k_digit` by `2^(B-1) * (1 - 2^-B)/(1 - 2^-b)` when `B` is a whole
number of limbs, whose digits are sums of limbs balanced at `b`, and by
`2^(B-1)` otherwise, where digits are balanced at `B` itself. The stored estimate takes each component's maximum over all gadget
columns. The secret's second moment enters when deriving phase noise. Key coverage is checked before this
construction, so no gadget truncation residue is omitted.

Private key-switch and encryption-to-shares transcripts record their selected
flood, rather than ordinary encryption noise. A Gaussian uses its sigma-squared
parameter, which bounds the conditioned discrete draw's variance. A `bits`-wide
uniform flood has variance `(2^(2*bits)-1)/12`; its mean is `-1/2`, and that bias
is not included in the centered variance. Public key-switching replaces the
ordinary body error with the flood. These transcripts describe newly generated
share-construction error at its original grid. Key-switch finalization clears
the resulting ciphertext's `noise()` to `None`: the output also
contains the input ciphertext's existing error and any precision-conversion
error, whose composition is not tracked.

When private key switching generates a share narrower than the received mask,
subtracting the two cropped inner products and normalizing also introduces
conversion error. Its effective model adds one unit of variance at the share's
precision to the flood variance. Public key-switch shares add the same term
when narrower than the received mask and the public key has the share precision;
a wider key already includes core's conversion estimate. Equal or wider shares
add no conversion term.

As in core's noise models, precision reduction adds a modeled half-ulp variance
per ciphertext component. For noncentered secrets, core also adds the squared
tie-rounding bias bound described in the core contracts. The components are folded against the secret only
when deriving phase noise. Adding that rounding term
independently is an approximation, not a proof that it is uncorrelated with
other errors. Unknown secret moments produce an infinite estimate while keeping
secret provenance. Homomorphic operations clear fresh component noise from their
outputs. Copies, preparation, compression and serialization preserve it, and
fresh key-generation or encryption factories record it after their internal
arithmetic. Callers still choose protocol parameters using the complete error
and statistical budget.

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
every share: a `Noise`, either a discrete Gaussian with `sigma >= 1`
truncated at six `sigma` (`Noise::CUTOFF_FACTOR`) or a uniform distribution on
consecutive integers. The flood is sampled on the precision grid of the value it hides,
which each protocol trait names, so that it reaches its bottom bit; on a
coarser grid the low bits would be exact linear equations in the secrets.

Each party provisions its own flood, so that its shares hide its secret
whichever other parties are corrupt. For a statistical margin `lambda`, the
flood must dominate the input encryption, evaluation and rounding errors by
`2^lambda`, in units of the sampling grid, over the whole transcript:

- For a fixed integer discrepancy vector `e`, the untruncated Gaussian shifts
  by at most `||e||_2 / (2 sigma)` in statistical distance, the uniform
  distribution by at most `||e||_1 / 2^bits`.
- Truncating at `B = floor(6 sigma)` adds little. Comparing the truncated
  worlds directly over `M` coefficients gives at most
  `(||e||_2 / (2 sigma) + ||e||_1 p_edge) / (1 - M tau)`, where
  `tau <= 2 exp(-18)` is the untruncated mass beyond `B` and `p_edge`, the
  largest probability within `|e_i|` of `±B`, is about
  `exp(-18) / (sigma sqrt(2 pi))` when `|e_i|` is small next to `sigma`: about
  `1e-8` times the shift term per coefficient.
- Random input-error tails and repeated or adaptive calls need their own
  bounds.

Correctness needs the sum of every party's flood bound, `floor(6 sigma)` or
`2^(bits-1)`, the input and conversion errors and any fresh encryption noise
to fit the decoding margin. The protocols check the flood against the sampling
precision before drawing randomness. They cannot infer the input error or
certify a security level, and neither can statistical tests.

## Replacing an operation

An override reproduces the reference, its seed and layout checks included,
as the [layering contract](../../docs/guides/layering-contract.md) states.
`mhe_parity_test_suite!(mod <name>, backend_ref = R, backend_test = T)`
checks a backend bit for bit against a validated one: each protocol's shares,
aggregates and finalized output and each PAT type's aggregation and
finalization, with their layouts, noise estimates, flags, seeds and the draws
left in every private stream. Fixtures are built on `R` and transferred to
`T`, each call runs on its own query's scratch, and a controlled-sampling
comparison backend copies `T`'s draws when their samplers differ. Every backend
type registering mhe runs it; the comparison backends also run
`mhe_backend_test_suite!`.
`impl_mhe_reference_full!` selects every family. Each OEP trait has its own
opt-in, `impl_<family>_reference_full!` with `<family>` the trait in snake case
without `Impl`; a backend replacing one family selects the others alone:
`impl_mhe_glwe_pat_compressed_reference_full!`, `impl_mhe_gglwe_pat_compressed_reference_full!`,
`impl_mhe_gglwe_pat_reference_full!`, `impl_mhe_glwe_public_key_protocol_reference_full!`,
`impl_mhe_glwe_switching_key_protocol_reference_full!`,
`impl_mhe_glwe_automorphism_key_protocol_reference_full!`,
`impl_mhe_glwe_tensor_key_protocol_reference_full!`, `impl_mhe_ggsw_protocol_reference_full!`,
`impl_mhe_glwe_private_keyswitch_protocol_reference_full!`,
`impl_mhe_glwe_public_keyswitch_protocol_reference_full!`,
`impl_mhe_glwe_enc_to_share_protocol_reference_full!` and
`impl_mhe_glwe_share_to_enc_protocol_reference_full!`. The reference traits
stay callable from an override.

## Workspace

Size scratch with the queries: each PAT type's `*_finalize_tmp_bytes`, and each
protocol's `mhe_*_share_gen_tmp_bytes` and `mhe_*_share_finalize_tmp_bytes`. A
replacement that needs more workspace replaces the matching query.
