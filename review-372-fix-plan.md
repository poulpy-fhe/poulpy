# Fix plan: review of `jp/372-unified-noise` (issue #372)

- Reviewed range: `3d3ee09d2..7624f65c6` (merge base with `origin/main` to HEAD at review time), 2026-10-06.
- Source: adversarial multi-agent review (73 raw findings, 65 confirmed), merged into the items below.
- IDs match the review report: H high, M medium, L low, A unresolved assumption, T test gap, D docs. X0 is a decision raised while preparing this plan.

## How to use this file

1. Work through the items in order. The order minimizes rework: decisions first, then the wire format, the noise model, the metadata lifecycle, the sampler, and docs with the CHANGELOG last.
2. Each item is self-contained: location, problem, fix, sub-steps, verification and dependencies.
3. Line numbers refer to `7624f65c6` and drift as patches land. Locate code by the symbol named in each item.
4. Tick an item (`- [x]`) once it is resolved. A resolution is either a patch (fill `Patched in:` with the commit) or a recorded decision not to patch (`Notes: won't fix, <reason>`). Tick sub-steps independently.
5. Items marked **decide** need a design choice before coding; the options are listed in the item.

## Conventions for patches

- Build and test: one cargo process at a time, `CARGO_BUILD_JOBS` at most 2, tests with `--release`.
- Git: no commit without an explicit request; commit messages are a single header line.
- Code: delegates forward only (see X0); no ring branches (per-ring impls where the math differs); library code never calls `Module::new`; `use` imports instead of qualified paths.
- Prose: no em-dash separator; keep comments, doc comments and CHANGELOG minimal; the CHANGELOG describes the net diff against the base and is written once at the end (item 29).

## Index

| # | ID | Severity | Title |
|---|---|---|---|
| 00 | X0 | decide | Where metadata is stamped: delegates or references |
| 01 | H1 | high | Lossless secret-law codec (public key round trip panics) |
| 02 | H2 | high | Keys encrypted to ENCAPSULATED secrets cannot be serialized |
| 03 | L13 | low | Compact LWE metadata (size, double build) |
| 04 | L3 | low | Reader error paths and `LWECompressed` counts |
| 05 | T4, T5 | medium | Serialization tests that test something |
| 06 | M2 | medium | Ring-correct product weight (conjugate-invariant) |
| 07 | M1 | medium | Binary secrets: rounding bias in public-key metadata |
| 08 | A1 | assumption | Inherited mask terms with noncentered laws |
| 09 | A2 | assumption | Fixed-weight laws after flattening |
| 10 | L8 | low | MHE public key-switch share conversion error |
| 11 | L9 | low | MHE GGSW switching term with dsize >= 2 |
| 12 | T6, T7, T8 | medium | Measure noise against metadata; CI and reduced-precision coverage |
| 13 | M3 | medium | `ckks_copy` level drop keeps a stale tag |
| 14 | L4 | low | Stale fresh tags after data changes |
| 15 | L5 | low | Dropped or misshaped tags |
| 16 | L6 | low | Validate before stamping in public-key encryption |
| 17 | L1 | low | MHE GGSW finalize panics on an untagged key |
| 18 | note | decide | Narrowing `glwe_copy` versus `glwe_normalize` |
| 19 | M4 | medium | Carried ENCRYPTION fallback writes every limb (perf) |
| 20 | L2 | low | `cumulative_table` certification blow-up |
| 21 | L11 | low | Feature gating of `noise.rs` tests |
| 22 | T1, T2, T3, T9 | medium | Sampler conformance tests and golden pins |
| 23 | A3 | assumption | Controlled-sampling stream independence |
| 24 | L10 | low | `noise()` name collision |
| 25 | L7 | low | Document the provenance check on retagged keys |
| 26 | L12 | low | MHE aggregate docs |
| 27 | D3 | low | OEP contracts and metadata obligations |
| 28 | D2 | low | Stale and inaccurate docs |
| 29 | D1 | medium | CHANGELOG net diff (last) |

## Phase 0: decide first

- [x] **00 · X0 · decide · Where metadata is stamped: delegates or references**
  - **Where:** metadata logic added to delegate bodies on this branch: about 196 `set_noise` calls in 41 files under `poulpy-core/src/delegates/`, `poulpy-ckks/src/delegates/`, `poulpy-bin-fhe/src/delegates/` and `poulpy-mhe/src/delegates/`. Examples: `glwe_add_into` in `poulpy-core/src/delegates/operations.rs` (`BE::glwe_add_into(..); res.set_noise(None);`), `glwe_encrypt_pk` in `poulpy-core/src/delegates/encryption.rs` (plan computed before `BE::*`, stamped after).
  - **Problem:** the project convention is that a delegate body is a single `BE::*` call. The branch makes delegates compute and stamp metadata so that backend overrides cannot skip it. Items 03, 13, 14, 16 and 27 touch exactly this logic, so the placement must be settled first. This was not part of the review report.
  - **Options:**
    - (a) Keep stamping in delegates and record it as a deliberate exception (metadata is post-processing every override must receive). Fixes below then go in delegates.
    - (b) Conform: move stamping into reference and derived bodies, restore single-call delegates, and state the metadata obligation in every OEP `# Safety` section (item 27). Out-of-tree overrides become responsible for the tags.
  - **Steps:**
    - [x] decision recorded in Notes
  - **Depends on:** none · **Patched in:** this commit · **Notes:** Decision: option (b). Delegates forward only; reference and derived implementations own metadata and OEP contracts state the obligation.

## Phase 1: wire format and serialization

- [x] **01 · H1 · high · Lossless secret-law codec (public key round trip panics)**
  - **Where:** `Distribution::write_to`, `read_from`, `pack_f64`, `unpack_f64` (`poulpy-core/src/dist.rs:130-195`); `ComponentNoise::write_optional`, `read_optional` (`poulpy-core/src/component_noise.rs:300-386`); `GLWEPublicKey::read_from` and `write_to` (`poulpy-core/src/layouts/glwe_public_key.rs:303-347`); `GLWEPublicKeyCompressed::read_from` (`poulpy-core/src/layouts/compressed/glwe_public_key.rs:190-240`). Panic sites: `public_key_phase_plan` (`poulpy-core/src/fresh_noise_model.rs:198-204`) and `assert_public_key_distribution` (`poulpy-mhe/src/reference/mod.rs:75-90`).
  - **Problem:** a public-key stream stores the secret law twice. `Distribution::write_to` drops the 8 low mantissa bits of a probability; PNM3 keeps `p.to_bits()`. After `read_from`, `dist` differs bitwise from the provenance base, and `public_key_phase_plan` panics ("ephemeral distribution differs from its secret provenance") in every `glwe_encrypt_pk*`, `ggsw_encrypt_pk` and MHE public-key share. Affected: any p with a nonzero low byte, including 2/3 (uniform ternary), 1/3, 0.1, 0.3, 0.7, 0.9. Dyadic p (0.5, 0.25) survive, and tests use only 0.5.
  - **Evidence:** `(2/3).to_bits() = 0x3fe5555555555555`. The old codec writes `1<<56 | bits>>8` and reads back `(word & 0x00ff_ffff_ffff_ffff) << 8 = 0x3fe5555555555500`. `Distribution::eq` compares bits.
  - **Fix (recommended):** make `Distribution::write_to/read_from` lossless: a u8 tag plus the full u64 payload, the layout PNM3 already uses (`component_noise.rs:301-316, 345-366`), including its `[0, 1]` probability validation. Have `write_optional/read_optional` call the same encoder. The public-key, compressed-key and BRK formats already break on this branch. Do not canonicalize PNM3 instead: `component_noise.rs:564-570` pins lossless provenance for 0.3, and truncating would make local and received metadata disagree in `same_secret` and `aggregate_metadata`.
  - **Fallback:** in both public-key readers, when the metadata base is not `NONE`, require the legacy word to equal the base's legacy encoding (else `InvalidData`) and set `dist = base`.
  - **Steps:**
    - [x] code: one lossless codec shared by `Distribution` and PNM3
    - [x] test: core, `TernaryProb(2/3)`: generate, write and read (plain and compressed, then decompress), prepare, `glwe_encrypt_pk` and `glwe_encrypt_zero_pk`; assert no panic and `pk2 == pk`
    - [x] test: MHE share round trip with 2/3 as a separate case (the loop at `poulpy-mhe/src/test_suite/public_key.rs:149` hardcodes E[s^2] = 0.5 in its expectations)
    - [x] docs: codec docs in `dist.rs`
  - **Verify:** new tests pass. Golden digests that hash public keys change only by the codec; re-pin them in item 22.
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **02 · H2 · high · Keys encrypted to ENCAPSULATED secrets cannot be serialized**
  - **Where:** the `ENCAPSULATED` arm of `ComponentNoise::write_optional` (`poulpy-core/src/component_noise.rs:310-315`), reached from `GGLWE::write_to` (`poulpy-core/src/layouts/gglwe.rs:814`). Provenance is recorded from `sk_out.dist` by the `glwe_switching_key_encrypt_sk` delegate (`poulpy-core/src/delegates/encryption.rs:530-536`). Producers: CKKS `generate_keys` (`poulpy-ckks/src/layouts/bootstrapping_keys.rs:300-306`, every preset), core `glwe_ci_trace_key_encrypt_sk` (`poulpy-core/src/reference/ci_conversion.rs:134-135` with `poulpy-core/src/layouts/glwe_secret.rs:458`), SHIP keyset dense-to-sparse (`poulpy-ckks/src/layouts/ship/keyset.rs:538-551`), PaCo (`poulpy-ckks/src/paco/secret.rs:428`), ring-switch outbound key.
  - **Problem:** `write_to` returns `InvalidData` ("secret distribution has no wire representation") after `GLWESwitchingKey::write_to` already emitted the 8 degree bytes. These keys serialized at the merge base, and the CKKS docs (`bootstrapping_keys.rs:14-21, 82-86`, `docs/bootstrapping.md:271`) call the unprepared key set the form to serialize.
  - **Fix (recommended):** in `ComponentNoise::from_secret_at` (`component_noise.rs:147-155`), record `Distribution::NONE` when the base is `ENCAPSULATED(_)`. `base_moments` returns `None` for both, so no estimate changes, and every producer passes through this constructor. Keep the `write_optional` error only as a guard for hand-built metadata.
  - **Alternative:** write `ENCAPSULATED` with the `NONE` wire tag `(6, 0)` in `write_optional` (a round trip then turns the label into `NONE`).
  - **Steps:**
    - [x] code: normalize in `from_secret_at`
    - [x] code: validate serializability before writing the degree headers in `GLWESwitchingKey::write_to`, `GLWESwitchingKeyCompressed::write_to` and the automorphism-key writers
    - [x] test: replace `encapsulated_distribution_rejection_does_not_write` with round trips (write, read, compare equal) of a `GLWECITraceKey` and of a switching key encrypted to an `ENCAPSULATED` `sk_out`
    - [x] test: write and read the key set produced by CKKS `generate_keys` for one preset
  - **Verify:** both the dense-to-sparse and sparse-to-dense keys write; new tests pass. The CHANGELOG sentence (line 69) is handled in item 29.
  - **Depends on:** 01 (same files) · **Patched in:** this commit · **Notes:**

- [x] **03 · L13 · low (perf, format) · Compact LWE metadata**
  - **Where:** `ComponentNoise::from_secret_at` (`poulpy-core/src/component_noise.rs:147-155`), `write_optional` and `read_optional` (`317-386`); `lwe_encrypt_sk` delegate (`poulpy-core/src/delegates/encryption.rs:139-147`) and reference (`poulpy-core/src/reference/encryption/lwe.rs:98-103`).
  - **Problem:** a fresh LWE tag stores n+1 terms: 16(n+1) bytes in memory (about twice a 1-limb payload) and 8(n+1)+33 bytes in PNM3 (about the payload size). The tag is built twice per encryption (delegate and reference), with 4 allocations.
  - **Fix:** keep n+1 logical terms. PNM3 writes `count` and a stored prefix ending at the last nonzero term; the reader pads zeros at the shared precision after checking `count` against the destination shape (pass the expected count into `read_optional`, so an untrusted count cannot force an allocation). Build the tag once (placement per X0). Optional: build the `Arc` in one pass.
  - **Decide:** whether to change the format at all on top of the intended n+1 semantics.
  - **Steps:**
    - [x] code: stored-prefix encoding in PNM3
    - [x] code: single build per encryption
    - [x] test: LWE round trip with Some metadata (n+1 terms); a `u64::MAX` count is rejected without allocating
  - **Verify:** a fresh LWE's metadata block shrinks from 33 + 8(n+1) to about 49 bytes; round trips stay exact.
  - **Depends on:** 00, 01, 02; finish before re-pinning digests (item 22) · **Patched in:** this commit · **Notes:** Decision: compact stored-prefix PNM3 while retaining all logical LWE components. Fresh body-only metadata uses 49 bytes.

- [x] **04 · L3 · low · Reader error paths and `LWECompressed` counts**
  - **Where:** `read_from` in `poulpy-core/src/layouts/glwe.rs:331-339`, `lwe.rs:433-442`, `gglwe.rs:799-808`, `ggsw.rs:616-626`, `compressed/glwe.rs:288-297`, `compressed/gglwe.rs:332-347`, `compressed/ggsw.rs:370-385`, `compressed/glwe_public_key.rs` (before `data.read_from`, about line 219), `compressed/lwe.rs:114-120`; `decompress_lwe` (`compressed/lwe.rs:139-147`).
  - **Problem:**
    - Fields (`base2k`, `data`, rank, seed, ...) are overwritten before the metadata count is validated. On `Err` the old tag stays attached to new data, possibly with the wrong component count.
    - `LWECompressed::read_from` assigns the noise first and cannot validate its count (the format stores no dimension). `decompress_lwe` then panics through `expect` on a mismatch: a crash on malformed input.
  - **Fix:**
    - Set `self.noise = None` right after `read_optional` and assign the parsed tag last, or peek the header and validate before mutating, as `GLWEPublicKey::read_from` does.
    - **Decide** for `LWECompressed`: drop its noise field (nothing produces Some), or store the LWE dimension on the wire and validate like the siblings. Minimal alternative: `decompress_lwe` drops a mismatched tag instead of panicking.
  - **Steps:**
    - [x] code: clear-then-assign in every reader
    - [x] code: `LWECompressed` per the decision
    - [x] test: a failing read leaves `noise() == None` and `write_to` works afterwards; tagged rank-2 receiver with a rank-1 payload; `LWECompressed` count mismatch
  - **Depends on:** 01, 02, 03 · **Patched in:** this commit · **Notes:** Decision: LWECompressed carries no metadata because its format has no secret dimension; reject tagged streams before allocation. All readers validate destination component counts before committing headers, including absent metadata, so repeated malformed reads cannot enlarge the trusted allocation bound. Compressed readers also preserve seed and entry shape.

- [x] **05 · T4, T5 · medium (tests) · Serialization tests that test something**
  - **Where:** `poulpy-core/src/test_suite/serialization.rs:32-168` (round trips) and `128-147` (malformed public-key stream); `poulpy-core/src/test_suite/noise/conversion.rs:42-67` (malformed LWE stream); `poulpy-bin-fhe/src/blind_rotation/test_suite/serialization.rs:16-57`.
  - **Problem:** the two malformed-stream tests are vacuous: their streams lack the PNM3 prefix, so `read_optional` rejects them before the shape guards they target. Round-trip suites only cover `noise: None` for LWE, GGLWE, GGSW, GLWEPublicKey, the compressed types, key wrappers and BRK. Some round trips exist only for GLWE and the compressed public key with p = 0.5, which hides items 01 and 02.
  - **Fix:** prepend `ComponentNoise::write_optional(None, &mut stream)` to crafted streams and assert the message (`"invalid public key"`, `"LWE body and mask sizes must match"`); remove the stale `max_size` parameter of `write_vec_znx_bytes`. Give each round-trip fixture distinct Some metadata with the right count (RANK+1, N_LWE+1); public key with `TernaryProb(0.3)`; set metadata on BRK GGSWs.
  - **Steps:**
    - [x] vacuous tests reach their shape guards
    - [x] Some-metadata round trips for every container
    - [x] a public-key stream with a wrong component count is rejected
  - **Depends on:** 01, 02, 03, 04 · **Patched in:** this commit · **Notes:**

## Phase 2: noise model

- [x] **06 · M2 · medium · Ring-correct product weight (conjugate-invariant)**
  - **Where:** `public_key_encryption_plan` and `public_key_phase_plan` (`poulpy-core/src/fresh_noise_model.rs:183-313`: `n` at 205, `rank_n` at 212, `phase_noise(n)` at 219 and 259, components at 283-299); `ComponentNoise::phase_noise` and `weighted_phase_noise` (`poulpy-core/src/component_noise.rs:182-214`); docs `poulpy-core/docs/core-contracts.md:138-163, 192-198`, `poulpy-mhe/docs/mhe-contracts.md:57`.
  - **Problem:** on CI modules (n stored coefficients of an element of Z[X]/(X^2n+1)) a product coefficient has weight 4n-3 at index 0 and 2n-2 elsewhere (mean about 2n). The planner and `phase_noise` use n. Recorded public-key phase noise is 2x low on average and 4x at coefficient 0: at n=256 with ternary 0.5 it records 257 sigma^2 against a true 513 sigma^2 (mean) and 1022 sigma^2 (c=0). With k_pk > k, the selector can drop a key limb it needs (target exceeded 1.73x).
  - **Fix:** a per-ring product weight through per-ring impls, not a branch: for example a core trait implemented for `Standard` (n) and `ConjugateInvariant` (4n as the bound, 2n for a mean), or arithmetic on `CYCLOTOMIC_ORDER_FACTOR`. Use `rank * w` wherever `rank_n` feeds a fold (inherited, `secret_fold`, rounding, truncation including the L1 branch, `ephemeral_fold`), and reconstruct with `weighted_phase_noise(n, w)`. Keep `base_moments` on the stored n. Document `phase_noise(n)` as negacyclic.
  - **Alternative:** CI reports no model (public-key encryption records `None` and uses the full key precision), documented.
  - **Steps:**
    - [x] code
    - [x] docs
    - [x] test: in item 12
  - **Depends on:** none; do before 07, which reuses the weight · **Patched in:** this commit · **Notes:**

- [x] **07 · M1 · medium · Binary secrets: rounding bias in public-key metadata**
  - **Where:** `public_key_phase_plan`: `rounding` (`poulpy-core/src/fresh_noise_model.rs:241`) and `component_rounding` (`284`); docs `poulpy-core/docs/core-contracts.md:151, 176-185, 199-200`, `poulpy-mhe/docs/mhe-contracts.md:107-111`.
  - **Problem:** normalization rounds ties up, so with D = work - k dropped bits each residue has mean 2^-(D+1) ulp. A noncentered secret (BinaryProb, BinaryFixed, BinaryBlock, any party count) adds these coherently: E[err_c] = 2^-(D+1) (1 + r mu_S (2c+2-n)). The model only adds the zero-mean Q = (1 + r n E[S^2]) / 4. Ternary laws are unaffected.
  - **Evidence:** the MHE tag fixture (n=64, r=2, 3 parties, `BinaryProb(0.5)`, k_pk=57, k=56, D=1) records 1573; the mean alone at c=63 is 0.25 (1 + 2 x 64 x 1.5) = 48.25, so its square is 2328; the true second moment is about 3835 (Monte Carlo 3859). At n=2048 with a 1-bit gap the estimate is 4.9x low on average and 13x at c=n-1.
  - **Fix:** for a noncentered law with work > k, let M = 2^-(D+1) (1 + r w mu_S), with w from item 06, and set `component_rounding = 0.25 + M^2 / (1 + secret_fold)`; the phase then gains M^2. Optionally include M^2 in the `fits` target, or prefer a work precision with M^2 <= Q when the key has the limbs. Alternative: an unbiased tie rule in this normalization (a HAL change).
  - **Steps:**
    - [x] code
    - [x] update expectations in `fresh_noise_model.rs` tests (about lines 531-532) and `poulpy-mhe/src/test_suite/public_key.rs` (about 244, 294, 337)
    - [x] docs
    - [x] test: in item 12
  - **Depends on:** 06 · **Patched in:** this commit · **Notes:**

- [x] **08 · A1 · assumption · Inherited mask terms with noncentered laws**
  - **Where:** `public_key_phase_plan` (`poulpy-core/src/fresh_noise_model.rs:218-221, 285-291`).
  - **Problem:** inherited components r n q_u V_j, recombined with n q_S, assume uncorrelated mask errors. With a binary u they share u's mean, so Cov(e_x, e_x') is proportional to mu_S^2 (n - 2|x - x'|). This holds for every key generated in the repo (public-key masks are recorded as 0) and fails for keys whose metadata carries nonzero masks (`set_noise`, `with_components`, PNM3).
  - **Decide:**
    - Minimal: when the law is noncentered and any mask variance is nonzero, set the inherited estimate to infinity (falls back to full precision with an unbounded estimate).
    - Precise: add the cross term r mu_u^2 mu_S^2 n(n-1)(n-2)/3 M to `inherited` and the matching per-component term (exact for BinaryProb, an upper bound for BinaryFixed; BinaryBlock keeps the fallback).
    - In either case, state the exactness condition of I in `core-contracts.md` (near line 149).
  - **Steps:**
    - [x] code
    - [x] test: BinaryProb with nonzero mask metadata
  - **Depends on:** 06, 07 · **Patched in:** this commit · **Notes:** Decision: noncentered inherited mask errors use full key precision and an unbounded estimate.

- [x] **09 · A2 · assumption · Fixed-weight laws after flattening**
  - **Where:** `ComponentNoise::lwe_phase_noise` (`poulpy-core/src/component_noise.rs:192-196`), `base_moments` (`poulpy-core/src/fresh_noise_model.rs:10-28`), `lwe_secret_from_glwe_secret` (`poulpy-core/src/layouts/glwe_secret.rs:351-355`); convention at `poulpy-core/src/dist.rs:73-75`.
  - **Problem:** a rank-r GLWE secret tagged `TernaryFixed(h)` or `BinaryFixed(h)` and flattened to r n coefficients keeps h nonzeros per column, but `lwe_phase_noise` uses h/(r n): the weight is r times too small (and the BinaryFixed mean too). This only matters once LWE masks carry noise.
  - **Decide:** carry the sampling block length in the fixed-weight tag (or in `SecretDistribution`) and use it in `base_moments`; or give `lwe_phase_noise` a `weight_block` argument; or document the restriction. Do not rescale the tag when flattening (it contradicts `dist.rs:73-75` and the test at `poulpy-core/src/test_suite/noise/conversion.rs:87`).
  - **Steps:**
    - [x] code or docs per the decision
    - [x] test: a flattened rank-2 fixed-weight secret
  - **Depends on:** none · **Patched in:** this commit · **Notes:** Decision: add lwe_phase_noise_with_block for the original sampling dimension after flattening; preserve distribution tags.

- [x] **10 · L8 · low · MHE public key-switch share conversion error**
  - **Where:** `mhe_glwe_public_keyswitch_share_gen_reference` (`poulpy-mhe/src/reference/keyswitch.rs:304-309`; compare the private key switch at `133-143`).
  - **Problem:** with share k < mask k and `pk_out.k() == share k`, the plaintext is truncated to the key prefix and normalized to k, but core records no rounding (work == k), so the body leaves out the conversion error the private key switch charges.
  - **Fix:** after `glwe_encrypt_pk_smudged`, add the same conversion term to the body when `res.k() < mask.k()`, only when `pk_out.k() == res.k()` (to avoid stacking on core's 0.25). Extend `poulpy-mhe/docs/mhe-contracts.md:102-105`.
  - **Steps:**
    - [x] code
    - [x] test: share k = K - BASE2K and pk k = share k; assert the body term
  - **Depends on:** 07 · **Patched in:** this commit · **Notes:**

- [x] **11 · L9 · low · MHE GGSW switching term with dsize >= 2**
  - **Where:** `fresh_ggsw_metadata` (`poulpy-mhe/src/reference/ggsw.rs:330-353`); `poulpy-mhe/docs/mhe-contracts.md:86-87`.
  - **Problem:** a combined digit of dsize balanced limbs reaches 2^(b-1) (2^B - 1) / (2^b - 1) > 2^(B-1), with B = dsize b, so the switching term is underestimated by up to f^2 < 4.
  - **Fix:** multiply the switching variance by f^2 with f = (1 - 2^-B) / (1 - 2^-b) (f = 1 at dsize 1). Mirror it in `poulpy-mhe/src/test_suite/ggsw.rs:205-212` and fix the docs.
  - **Steps:**
    - [x] code
    - [x] docs
    - [x] test: GGSW finalize with an ephemeral key of `Dsize(2)`
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **12 · T6, T7, T8 · medium (tests) · Measure noise against metadata; CI and reduced-precision coverage**
  - **Where:** `test_glwe_encrypt_pk` and `expected_public_key_noise` (`poulpy-core/src/test_suite/noise/encryption/glwe_ct.rs:521-608, 646-700`); MHE `test_ciphertext_encryption_noise_tags` (`poulpy-mhe/src/test_suite/public_key.rs:113-357`, fixture `poulpy-mhe/src/test_suite/fixtures.rs:355-388`); MHE GGSW, tensor-key and key-switch tests (`poulpy-mhe/src/test_suite/ggsw.rs:199-239`, `tensor_key.rs:82-106`, `keyswitch.rs:92-94, 200-219`); CKKS `poulpy-ckks/src/test_suite/encryption.rs:115-124` and `copy.rs`; bin-fhe `poulpy-bin-fhe/src/test_suite/parity/lifecycle.rs:450-451`; provenance tests `poulpy-core/src/test_suite/parity/encryption.rs:60-74` and `parity/encryption_keys.rs`; CI suite `poulpy-cpu-ref/src/test_suite/conjugate_invariant.rs:456-498`; parity `poulpy-core/src/test_suite/parity/encryption.rs:284-345`.
  - **Problem:**
    - Metadata tests recompute the implementation's formulas (tail factor, Minkowski bound, Young split) instead of measuring.
    - The only empirical public-key check is one-sided, uses ternary 0.5 and k_pk in {k, k + base2k}, and centers the std, which hides a noncentered bias.
    - Provenance tests cannot tell `sk_in` from `sk_out`.
    - No CI test runs public-key encryption.
    - Reduced-precision public-key encryption (pk.k > res.k) and GGSW public-key encryption run only on FFT64Ref, NTT4x30Ref and the oracles; the commit message's "replay test on every backend" record is wrong.
  - **Steps:**
    - [x] core: loop the secret over {`TernaryProb(0.5)`, `BinaryProb(0.5)`} and the gap over {0, 1, base2k}; pool the uncentered phase second moment (sum err^2 / n) and compare it with `ct.noise().unwrap().phase_noise(n)` read from the ciphertext (two-sided at k_pk == k, an upper bound plus V/2 lower bound otherwise). It must fail before item 07.
    - [x] MHE: the same measured check for collective keys (parties {1, 3}, k in {key_k, key_k - 1}), GGSW (with `Dsize(2)` and binary laws), tensor key and public key switch.
    - [x] CI: run `test_glwe_encrypt_pk` on CI types with ring-aware expectations, including coefficient 0. It must fail before item 06.
    - [x] provenance: give each non-encrypting secret a different law (`TernaryProb(0.5)`) so that recording the wrong secret fails.
    - [x] CKKS and bin-fhe: assert `precision() == ct.k()` and compare the measured residual with the metadata; check per-bit metadata in lifecycle parity.
    - [x] parity: extend `test_glwe_encryption_parity` with pk.k = k + {1, base2k, 3 base2k} and plaintexts wider than the prefix, so every SIMD and Rayon registration covers reduced precision; add a k_pk > k GGSW public-key case.
    - [x] reword the "independent" claims in comments (`glwe_ct.rs` about lines 600 and 646, `fixtures.rs` about line 355).
  - **Depends on:** 06, 07, 10, 11 · **Patched in:** this commit · **Notes:** The proposed universal V/2 lower bound is not valid for conservative metadata: centered rounding can approach V/3 (1/12 actual versus a 1/4 bound), while binary bias and CI use worst-coordinate bounds (binary CI approaches a mean/model ratio of 1/12). Empirical tests use documented floors accounting for that looseness and tighter equal-precision comparisons.

## Phase 3: metadata lifecycle

- [x] **13 · M3 · medium · `ckks_copy` level drop keeps a stale tag**
  - **Where:** `ckks_copy` delegate (`poulpy-ckks/src/delegates/copy.rs:17-19`) and `ckks_copy_reference` (`poulpy-ckks/src/reference/copy.rs:27-29`); shift path `ckks_copy_stamp_unary` then `ckks_shift_stamp_unary` (`poulpy-ckks/src/lib.rs:465-516`, `glwe_lsh` at 494).
  - **Problem:** with `dst.k() < src.k()` the copy runs `glwe_lsh(dst, src, offset)`, which multiplies the torus error by 2^offset, then re-tags with `src.noise()` at the source precision. `variance_at(dst.k())` is then 4^offset too low (offset 1: 2.56 against 10.24). Core clears on `glwe_lsh`, and `ckks_mul_pow2_into` with the same arithmetic returns `None`.
  - **Decide:** clear when offset > 0 (matches core), or relabel each term to precision p - offset with the same variance (lands on `dst.k()` for a fresh ciphertext). Compute the offset with `ckks_offset_unary` before the copy. Keep a single stamping site (per X0).
  - **Steps:**
    - [x] code
    - [x] test: `test_copy_aligned` asserts equal noise; `test_copy_smaller_output` asserts the chosen rule
    - [x] docs: the rule in `poulpy-ckks/src/api/copy.rs`
  - **Depends on:** 00 · **Patched in:** this commit · **Notes:** Decision: clear metadata when CKKS copy drops a level; aligned copies preserve it.

- [x] **14 · L4 · low · Stale fresh tags after data changes**
  - **Where:** `glwe_mask_inner_product` (`poulpy-core/src/delegates/decryption.rs:52-59`); mask-fill delegates `fill_glwe_mask_from_source/seed` and `fill_lwe_mask_from_source/seed` (`poulpy-core/src/delegates/encryption.rs:82-117`); `decompress_glwe` (`poulpy-core/src/layouts/compressed/glwe.rs:328-339`) and `decompress_lwe` (`compressed/lwe.rs:147-156`); row and entry views: GGSW `at_mut` and `GGSWAtViewMut::at_view_mut` (`layouts/ggsw.rs:455-517`), GGLWE (`gglwe.rs:515-610`), `GLWEPublicKey` `at_mut` and `at_view_mut` (`glwe_public_key.rs:76-131`).
  - **Problem:** these overwrite ciphertext data and leave the owner's fresh tag. Row views carry a copy of the container's tag, and changes stay on the view.
  - **Fix:** clear after `glwe_mask_inner_product` and after each mask fill (placement per X0); in `decompress_glwe` and `decompress_lwe`, assign the tag after the fill; document view-local semantics on the row and entry accessors (a caller writing rows must set the container's tag).
  - **Steps:**
    - [x] code
    - [x] docs
    - [x] test: each path ends with `noise() == None` (or the documented tag)
  - **Depends on:** 00 · **Patched in:** this commit · **Notes:**

- [x] **15 · L5 · low · Dropped or misshaped tags**
  - **Where:** `GGLWEPreparedToBackendRef` for `&GLWETensorKeyPrepared<BufRef>` (`poulpy-core/src/layouts/prepared/glwe_tensor_key.rs:195`, `noise: None`); `GGLWECompressedBackendRef::body_as_gglwe` (`poulpy-core/src/layouts/compressed/gglwe.rs:51-64`); `glwe_prepare` (`poulpy-core/src/layouts/prepared/glwe.rs:110-134`); `ComponentNoise::with_rank` (`poulpy-core/src/component_noise.rs:240-248`).
  - **Problem:**
    - A borrowed prepared tensor key reports `None` while the owner reports Some.
    - The rank-0 body view carries rank_out+1 components, which breaks the count invariant.
    - `glwe_prepare` does not check ranks, truncates columns, and keeps a fresh tag through `with_rank` truncation (the only reachable truncation).
  - **Fix:** `noise: self.0.noise.clone()`; `.map(|n| n.with_rank(0))` (or `None`) for the body view; assert equal ranks in `glwe_prepare` and pass the tag through unchanged; restrict `with_rank` to padding (`assert!(rank >= self.rank())`) and update its doc, `core-contracts.md:224` and its unit test.
  - **Steps:**
    - [x] code
    - [x] test: borrowed prepared tensor key keeps the tag; body view has exactly 1 component; `glwe_prepare` rank mismatch panics
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **16 · L6 · low · Validate before stamping in public-key encryption**
  - **Where:** `glwe_encrypt_pk_internal` (`poulpy-core/src/reference/encryption/glwe.rs:405-457`); `ggsw_encrypt_pk_derived` (`poulpy-core/src/oep/derived/encryption.rs:279-291`). Same pattern in the secret-key references (`glwe.rs:154-172, 201-218, 248-266` and the GGSW, GGLWE, LWE and compressed GLWE references).
  - **Problem:** `set_noise(plan.noise)` and `set_canonical(true)` run before the shape asserts and the ephemeral-law guard, so a call that panics leaves the destination retagged.
  - **Fix:** run the checks on `to_backend_ref()` first, then stamp, then take `to_backend_mut()`. Drop the early stamp in the derived GGSW path if the delegate stamps after success (per X0).
  - **Steps:**
    - [x] code
    - [x] test: a base2k mismatch or an unprepared key leaves the destination unchanged (`ct == before`, same canonical flag)
  - **Depends on:** 00 · **Patched in:** this commit · **Notes:**

- [x] **17 · L1 · low · MHE GGSW finalize panics on an untagged key**
  - **Where:** `mhe_ggsw_share_finalize` reference (`poulpy-mhe/src/reference/ggsw.rs:248-256`); fallback in `fresh_ggsw_metadata` (`342-353`).
  - **Problem:** (Some share, None key) panics with "invalid finalization: output key provenance differs", although keys legitimately lose their tags (`gglwe_keyswitch` clears; PNM3 with parties = 0). This is a regression against the base, and the infinite-variance fallback is dead code.
  - **Fix:** assert `same_secret` only when both tags are Some; otherwise take the existing infinite-variance path (or set `None`, mirroring the public-key plan).
  - **Steps:**
    - [x] code
    - [x] test: (Some, None) finalizes; (Some, Some) with different provenance panics
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **18 · note · decide · Narrowing `glwe_copy` versus `glwe_normalize`**
  - **Where:** `glwe_copy` delegate (`poulpy-core/src/delegates/operations.rs:124-133`) and reference (`poulpy-core/src/reference/operations/glwe.rs:1841-1880`).
  - **Problem:** a narrowing copy (dst.k < src.k) re-quantizes exactly like `glwe_normalize`, which clears the tag, while the copy keeps it (as the stated intent says copies preserve estimates). This is a consistency question, not a defect.
  - **Options:** keep and document that a narrowing copy's tag excludes its own rounding; clear on narrowing; or relabel at dst.k and add the 1/4 rounding term.
  - **Steps:**
    - [x] decision recorded, code or docs updated
  - **Depends on:** 13 (same decision shape) · **Patched in:** this commit · **Notes:** Decision: clear metadata on a narrowing core copy, matching normalization.

## Phase 4: sampler

- [x] **19 · M4 · medium (perf) · Carried ENCRYPTION fallback writes every limb**
  - **Where:** small-table branch of `add_noise` (`poulpy-cpu-ref/src/reference/noise.rs:465-485`, fallback test at 474) and `place_small` (`379-391`); benches `poulpy-cpu-ref/benches/noise.rs` and `encrypt_pk`.
  - **Problem:** when k mod base2k is in 1..=5 (every k when base2k <= 5), each coefficient gets a scalar 19-threshold scan, then `place_small` loops over all ceil(k/base2k) limbs with a checked `at_mut` and a serial i128 carry, although only 2 limbs can be nonzero. Public-key encryption samples at k + 3 for a limb-aligned k under a wider key, so it always lands here. Modeled (not measured): at N=2^16 with 30 limbs, about 6.6 ms instead of 0.75 ms per noise column, so rank-1 public-key encryption is about 30-70% slower.
  - **Fix:** in `place_small`, visit only the lowest m limbs: m = size when base2k == 1, otherwise m = min(size, 2 + bitlen(B) / base2k) with integer division and B = `table.len()`. That gives m = 2 for ENCRYPTION at base2k >= 6 and m = 3 at 5. Do not use 1 + ceil(bitlen(B) / b): it drops a nonzero top digit for dynamic tables (Gaussian{5.2, 6} at b=5: v=496 has digits [1, -16, -16]). Output and entropy stay bit-identical, and m depends only on public B and b. Optional: limb-major writes per 64-coefficient batch.
  - **Steps:**
    - [x] code
    - [x] test: replay and reconstruct for (b, k) in {(3,20), (12,49), (17,69), (52,209), (63,253)}: reconstruction equals the drawn sample, digits balanced, padding zero, other column untouched, identical source consumption
    - [x] bench: a fallback case (b=30, k=273) and `encrypt_pk` with pk.k = k + base2k
  - **Verify:** seeded outputs are bit-identical before and after; golden digests do not change because of this item.
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **20 · L2 · low · `cumulative_table` certification blow-up**
  - **Where:** `cumulative_table` (`poulpy-cpu-ref/src/reference/noise.rs:112-148`, check at 138, comment at 136-137).
  - **Problem:** the decoupled upper bound `cum_hi / total_lo` can exceed 1. Once the floor saturates at 2^128 - 1, certification needs every tail weight resolved, i.e. about 0.72 cutoff_factor^2 bits. Measured on a port: sigma=1, c=64 takes 76 ms (0.4 ms after the fix); sigma=0.25, c=256 takes 9.2 s; sigma=2^-60, c=2^62 (which passes `validate` and `assert_valid_for`) doubles precision until allocation aborts. The comment claims bounded precision.
  - **Fix:** `if value != u128::MAX && scaled_hi > (&floor + 1u8) * &total_lo { break; }` (C_i < 1, so a saturated floor is already within one unit), or certify with the coupled bound `cum_hi / (cum_hi + total_lo - cum_lo)`. `ENCRYPTION_CDT` is unchanged. Rewrite the comment. Optional: cache dynamic tables.
  - **Steps:**
    - [x] code
    - [x] test: `cumulative_table(2f64.powi(-60), 4) == vec![u128::MAX; 4]`; `cumulative_table(1.0, 64)` completes quickly; `encryption_table_matches_integer_construction` still passes
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **21 · L11 · low · Feature gating of `noise.rs` tests**
  - **Where:** test module of `poulpy-cpu-ref/src/reference/noise.rs` (imports at 496-501; the tests at about 616 and 701 call `module.vec_znx_add_noise`).
  - **Problem:** these tests need `SamplingImpl`, which exists only with `enable-core`. That feature is not a default and no dev-dependency enables it, so `cargo test -p poulpy-cpu-ref` without features fails to compile.
  - **Fix:** `#[cfg(feature = "enable-core")]` on those two tests and the imports only they use.
  - **Steps:**
    - [x] code
  - **Verify:** `cargo test -p poulpy-cpu-ref --release --no-run` builds with and without `--features enable-core`.
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **22 · T1, T2, T3, T9 · medium (tests) · Sampler conformance tests and golden pins**
  - **Where:** `poulpy-core/src/test_suite/sampling.rs:88-118, 133-227, 280-314`; `poulpy-cpu-oracle/src/sampling.rs` (`full_width`, PMF test at 241-263, bound at 83); tests in `poulpy-cpu-ref/src/reference/noise.rs:503-554`; golden tests `poulpy-cpu-ref/src/tests.rs:630-643` and in `poulpy-cpu-oracle/src/tests.rs`.
  - **Problem:**
    - Conformance draws only ENCRYPTION at base2k 17, k 31 (single-limb path). There is no uniform, rejection, other-sigma table, carried path, sample wider than 128 bits, untouched-limb check or pinned bound.
    - Parity copies placed digits, so the carried path never meets an independent check.
    - The secret laws `base_moments` relies on are untested, and the oracle's `ScalarZnxFill` is byte-identical to cpu-ref's.
    - Golden digests are re-recorded self-pins; the base's oracle == cpu-ref pin is gone.
  - **Steps:**
    - [x] move `full_width<BE>` from the oracle into the core suite, run on the supplied module: Uniform{1}, {base2k - shift}, {base2k - shift + 1}, {135}; Gaussian{2^132, 6}; Gaussian{0.75, 6} with a per-bin PMF check; at k % base2k != 0 and == 0. Assert exact reconstruction, zero padding, digit range, untouched sentinels, sign balance, magnitudes above 2^128.
    - [x] run `test_vec_znx_add_noise`, `test_vec_znx_big_add_noise` and `assert_two_additions` also at k in {2 BASE2K - 3, 2 BASE2K + 1} (carried path); add a mean check next to the std check.
    - [x] pin bounds with literals: (3.2, 6) -> 19, (15.25, 6) -> 91, (8 - 2^-50, 1) -> 7, (subnormal, 6) -> 0; production-path tests Gaussian{1.5, 1} (table) and Gaussian{100.5, 1} (rejection); the oracle PMF support comes from literals, not from `sampler.bound`.
    - [x] secret laws: `TernaryProb(0.25)`, `BinaryProb(0.25)`, `BinaryBlock(3)`, `BinaryBlock(8)`, `TernaryFixed`; pool at least 2^14 coefficients; compare the mean and E[s^2] with `coefficient_mean` and `coefficient_second_moment`; empty-block fraction about 1/(b+1); ternary sign mean about 0.
    - [x] golden: also compute the oracle digests from cpu-ref draws under `with_backend_samples` (FFT64Oracle at base2k 17, NTT4x30Oracle at 52) and assert the same constants. Re-pin every digest once, after items 01-04 and 19.
  - **Depends on:** 01, 02, 03, 04, 19 · **Patched in:** this commit · **Notes:**

- [x] **23 · A3 · assumption · Controlled-sampling stream independence**
  - **Where:** `BackendSamples` (`poulpy-core/src/test_suite/parity/controlled_sampling.rs:24-60`); adapters (`poulpy-cpu-oracle/src/sampling.rs` about 159 and 176; `poulpy-cpu-ref/src/test_suite/controlled_sampling.rs` about 25, 41, 61); `SamplingImpl` docs (`poulpy-core/src/oep/sampling.rs`).
  - **Problem:** the provider draws into column 0 of a minimal buffer, so it assumes a backend's stream depends on neither the column index nor the buffer shape. True in-tree; an out-of-tree backend that keys its stream on them would fail parity spuriously.
  - **Decide:** state and test the precondition in the `SamplingImpl` docs and the sampler suite, or pass `res_col`, the column count and the limb count through `scalar_samples` and `noise_samples` and the adapters.
  - **Steps:**
    - [x] docs and test, or plumbing, per the decision
  - **Depends on:** 22 · **Patched in:** this commit · **Notes:** Decision: document and test column, buffer shape and destination-content independence for sampling streams.

## Phase 5: API, docs and contracts (CHANGELOG last)

- [x] **24 · L10 · low · `noise()` name collision**
  - **Where:** `FheUint::noise` (`poulpy-bin-fhe/src/bdd_arithmetic/ciphertexts/fhe_uint.rs:171`), `FheUintPreparedDebug::noise` (`fhe_uint_prepared_debug.rs:110`), `GGSW::noise` and `GGLWE::noise` (`poulpy-core/src/reference/noise/ggsw.rs:26-46`, `gglwe.rs:25-45`); docs `docs/getting-started.md:70`, `poulpy-mhe/README.md:59`.
  - **Problem:** the inherent `noise(module, ..)` measurement methods shadow `LWEInfos::noise()`, so the documented `ct.noise()` does not compile (E0061).
  - **Decide:** delete the GGSW and GGLWE forwards (call `module.ggsw_noise` / `module.gglwe_noise`) and rename the FheUint ones (for example `noise_stats`), about 30 call sites including `poulpy-cpu-ref/examples/circuit_bootstrapping.rs:226`; or document `LWEInfos::noise(&x)` instead.
  - **Steps:**
    - [x] code or docs per the decision
  - **Depends on:** none · **Patched in:** this commit · **Notes:** Decision: rename measurement forwards to noise_stats, leaving noise() available for metadata.

- [x] **25 · L7 · low · Document the provenance check on retagged keys**
  - **Where:** `public_key_phase_plan` (`poulpy-core/src/fresh_noise_model.rs:198-204`); API docs of `GLWEEncryptPk`, `GLWEEncryptPkSmudged`, `GGSWEncryptPk` (`poulpy-core/src/api/encryption.rs:119-140`); setter `GLWEPublicKeyPrepared::dist_mut` (`poulpy-core/src/layouts/prepared/glwe_public_key.rs:42-46`).
  - **Problem:** retagging a prepared key's ephemeral law panics only when metadata is present; after `set_noise(None)` it silently succeeds. The behaviour change is undocumented.
  - **Fix:** document the rule (a key with metadata must keep its ephemeral law equal to its provenance base; untagged keys skip the check). If item 01 took the fallback, compare canonical wire forms here.
  - **Steps:**
    - [x] docs
  - **Depends on:** 01 · **Patched in:** this commit · **Notes:**

- [x] **26 · L12 · low · MHE aggregate docs**
  - **Where:** `poulpy-mhe/src/api/sharing.rs:14-15`, `poulpy-mhe/src/api/keyswitch.rs:15`.
  - **Problem:** the docs say shares are aggregated with core `glwe_add_assign`, which now clears the tag; a later MHE aggregate then panics on (None, Some).
  - **Fix:** point to `mhe_glwe_enc_to_share_share_aggregate` and `mhe_glwe_private_keyswitch_share_aggregate`.
  - **Steps:**
    - [x] docs
  - **Depends on:** none · **Patched in:** this commit · **Notes:**

- [x] **27 · D3 · low (docs) · OEP contracts and metadata obligations**
  - **Where:** core `EncryptionImpl` public-key methods (`poulpy-core/src/oep/encryption.rs:146-162`) and their delegates (`poulpy-core/src/delegates/encryption.rs:242-331, 474-476`); `PublicKeyEncryptionPlan` and `public_key_encryption_plan` (`poulpy-core/src/fresh_noise_model.rs:74-84, 306`, crate-private); MHE `# Safety` sections (`poulpy-mhe/src/oep/pat.rs:7-10, 25-28, 43-45`; `keyswitch.rs:14-16, 56-58`; `ggsw.rs:13-15`; `evaluation_key.rs:16-18, 61-63`; `tensor_key.rs:10-11`; `sharing.rs:15-17, 58-59`; `public_key.rs:13-15`); `poulpy-mhe/docs/mhe-contracts.md:184-186`; `poulpy-mhe/src/oep/mod.rs:4-6`.
  - **Problem:** the public-key OEP docs do not state what core assumes when it stamps metadata: fresh errors at k_sample, only the leading ceil(work / base2k) key limbs, flood at res.k, one normalization to res.k. Only `core-contracts.md` says so, and the planner is crate-private. The MHE Safety sections still list only seed and layout checks, not the provenance checks or the recorded `noise()`. There is no in-tree impact.
  - **Fix:** depends on X0. Under (b), every Safety section states the metadata obligation. Optionally pass the plan (sample and work precision) to the `EncryptionImpl` public-key methods in place of the removed `enc_infos`, or expose a public planner.
  - **Steps:**
    - [x] docs (and the optional API change)
  - **Depends on:** 00 · **Patched in:** this commit · **Notes:**

- [x] **28 · D2 · low (docs) · Stale and inaccurate docs**
  - **Steps:**
    - [x] `poulpy-cpu-ref/src/reference/ntt4x30/vec_znx_big.rs:27-28` and `poulpy-core/src/api/mod.rs:13-14`: describe `Noise` samples over ceil(k / base2k) limbs, not single-limb Gaussian noise.
    - [x] `poulpy-core/src/dist.rs:108-109`: the BinaryBlock law (a block is all zero with probability 1/(b+1), otherwise holds one 1 at a uniform position; n is a multiple of b). Have `poulpy-core/src/oep/sampling.rs:7` defer to the `Distribution` docs.
    - [x] `poulpy-core/docs/core-contracts.md:263-272`: replace the old smudging precondition (canonical input, normalize between additions) with the actual word-headroom condition.
    - [x] `poulpy-core/src/api/sampling.rs:8-12`: point to `Distribution` for the variant semantics.
    - [x] `docs/getting-started.md:70` and `poulpy-core/docs/core-contracts.md:128-130`: the `noise` field is `pub(crate)`; refer to `LWEInfos::noise()` and `set_noise`.
    - [x] `DEFAULT_BOUND_XE` (`poulpy-core/src/reference/encryption/mod.rs:64-66`): delete it, or derive it from `Noise::ENCRYPTION` and document it as the support bound.
    - [x] after item 06: mark `phase_noise(n)` and the core-contracts component formulas as negacyclic.
    - [x] terminology: "metadata" for `ComponentNoise`, "tag" for `Distribution` (`CHANGELOG.md:69`, `core-contracts.md:215, 225`, `docs/getting-started.md:70`).
    - [x] after item 01: byte layouts in `core-contracts.md` and `docs/ship.md` that describe the distribution word.
  - **Depends on:** 01, 06 · **Patched in:** this commit · **Notes:**

- [x] **29 · D1 · medium (docs) · CHANGELOG net diff (last)**
  - **Where:** `CHANGELOG.md:66-69, 80, 100, 116`; `poulpy-core/docs/core-contracts.md:229-232`.
  - **Problem:**
    - It describes intermediate branch states: the `cutoff` to `cutoff_factor` rename, and PNM1/PNM2 presented as rejected earlier formats although they never shipped.
    - It describes the `Noise`/`SamplingImpl` change three times.
    - It omits breaking items: the required `set_noise` on the `*ToBackendMut` traits, `LWEInfos::noise`, the `ggsw_encrypt_pk` bound `R: GGSWToBackendMut`, and the changed sampler output for fixed seeds.
    - It still gives the old public-key byte layout.
  - **Fix:** once all items above are resolved, rewrite the entries from `git diff 3d3ee09d2..HEAD`: one bullet per change against the base, including every breaking change made by items 01-28 (codec, PNM3 layout, ENCAPSULATED handling, renamed methods).
  - **Steps:**
    - [x] CHANGELOG rewritten from the net diff
    - [x] core-contracts.md PNM sentence fixed
  - **Depends on:** all other items · **Patched in:** this commit · **Notes:**

## Not to fix (refuted during the review)

- A zero-width Gaussian (floor(cutoff_factor sigma) = 0) adds no noise: consistent with the definition, and the recorded sigma^2 stays a conservative estimate.
- The uniform flood's mean of -1/2 is not recorded: its relative effect is 3P/(4^b - 1), immaterial.
- "The controlled-sampling scope docs promise a panic outside a scope": refuted.
- "The core OEP contract is a defect": refuted as a defect; the documentation part is item 27.

## Invariants to preserve (verified correct during the review)

- `ENCRYPTION_CDT` equals floor(2^128 P(|z| <= i)) for the exact binary64 3.2 with B = 19 (independent 120- and 400-digit recomputations). Statistical distance 8.9e-39 <= 19 x 2^-128; truncated variance 10.2399996 < sigma^2 = 10.24.
- Table path: scan semantics, sign mapping and biased batch compares are exact; the AVX2, NEON and scalar paths give identical output and consume 1032 bytes per 64 coefficients.
- The rejection path matches Canonne-Kamath-Steinke 2020 Algorithms 1-3; `uniform_below` is exact; uniform support is exactly [-2^(bits-1), 2^(bits-1) - 1]; `place_small` and `place_integer` produce balanced digits and drop the top carry modulo 1.
- `Noise::assert_valid_for` computes the exact bit length of floor(cutoff_factor sigma).
- Public-key plan: k <= k_sample <= k_pk; `fits` is monotone, so the binary search returns the minimum; the tail bound 0.5/(1 - 2^-b), both truncation branches, and the Minkowski and Young combinations are valid; flooding is excluded from the selection and sampled at res.k; the VMP uses the leading work_size key limbs; every tmp_bytes covers the scratch taken; pk.k < k panics before any mutation; the r ephemerals come from independent child seeds.
- Producers: secret-key encryption records sigma^2 on the body and zero masks at the sampling precision; `DEFAULT_SIGMA_XE` derives from `Noise::ENCRYPTION`; `base_moments` matches the samplers per column.
- Lifecycle: core evaluation operations clear the metadata; prepare, decompress, transfer and clone preserve it; plaintexts never carry one.
- PNM3: no allocation driven by the wire count; counts are checked against shapes everywhere except `LWECompressed` (item 04).
- MHE: provenance guards run before mutation; the collective public-key moments are right; centered common-key aggregation is exact.
- Controlled sampling: the flag is per module and off by default, only samplers change, and the scope cleans up on return and unwind.

## Validation limits of the review

- Static only: no build, test or benchmark ran. H1 and H2 were established by code tracing and byte-level models; M4's costs are modeled from operation counts; Monte Carlo ran in pure Python, mostly at n <= 2048.
- Not reviewed: compiled code generation of the table path, SIMD VMP kernels for narrowed outputs (relied on the HAL tests), aarch64/NEON code generation, CKKS internals (EvalMod, DFT, PaCo, SHIP), and the digest values themselves.

## Implementation validation (2026-10-06)

Validation was completed before committing the fixes. Commands used `CARGO_BUILD_JOBS=2`, one Cargo process at a time, and release tests.

- Core unit tests: 72 passed; doctests: 1 passed, 2 ignored.
- Reference sampler: 12 low-level and 11 shared tests passed. Oracle sampler: 16 passed. Native and crossed reference/oracle golden digests passed after the final format changes.
- Complete reference suite with all features: 1,509 passed, 18 ignored. The final reader hardening was additionally checked by the three serialization tests, including repeated malformed reads and BRK metadata. The malformed LWE body/mask-size regression also passed on FFT64Ref and NTT4x30Ref after reader hardening.
- The ignored `ntt4x30_preset_keys_roundtrip` test was explicitly run and passed, serializing a published CKKS preset's complete generated key set, including encapsulation keys.
- A focused multiparty run passed GGSW finalization (binary/ternary, ranks 1 and 2) and public key-switch checks on FFT64Ref and NTT4x30Ref.
- `cargo test -p poulpy-cpu-ref --release --no-run` passed without features. The all-feature suite above also covers `enable-core`.
- `cargo check --workspace --release --features poulpy-cpu-ref/enable-ckks,poulpy-cpu-ref/enable-bin-fhe,poulpy-cpu-ref/enable-mhe` passed.
- Criterion smoke benchmarks (10 samples, 0.1-second warmup and measurement): FFT64Ref carried noise at n=16,384, b=30, k=273 measured 261–268 microseconds; public-key encryption at n=16,384, b=17, k=68, rank=1, pk.k=85 measured 1.59–1.64 milliseconds. These runs check the benchmark paths and do not establish a speedup against a baseline.
- AVX2/FMA and AVX2/Rayon sampling: 10 passed. Encryption parity: 20 passed with `RUSTFLAGS="-C target-feature=+avx2,+fma"` and `--features enable-avx,enable-rayon`, covering both FFT64 and NTT4x30.
- An all-feature workspace check cannot combine ARM/NEON and x86 backend features on this host. ARM/NEON and AVX512 execution remain untested here.
- Documentation for core, CKKS, bin-fhe and MHE built with `RUSTDOCFLAGS="-D warnings"`. `cargo fmt --all -- --check` and `git diff --check` passed.
