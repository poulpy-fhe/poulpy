# CKKS Linear Transformation via Baby-Step / Giant-Step — Specification

Implementation specification for the homomorphic evaluation of a slot-domain linear
map (a matrix–vector product over the CKKS slots, e.g. `CoeffsToSlots` /
`SlotsToCoeffs`) using the baby-step / giant-step (BSGS) decomposition with hoisting
and lazy normalization.

The engine is **GLWE-level and scheme-agnostic** — it operates on GLWE ciphertexts,
GGLWE automorphism keys, and prepared plaintext diagonals — so it lives in
**poulpy-core**, beside `keyswitching` and `automorphism`. CKKS only contributes the
diagonal *encoding* and the galois-element map; the `LinearTransformation` /
`LinearTransformationOps` traits in
[`poulpy-ckks/src/api/linear_transformation.rs`](../poulpy-ckks/src/api/linear_transformation.rs)
become a thin wrapper that drives the core engine (§11).
The diagram [`lt_bsgs.png`](lt_bsgs.png) is the visual companion to this text.

> **Implementation status.** The engine lives in **poulpy-core** as the scheme-agnostic
> `GLWELinearTransformOps` module tree
> ([`poulpy-core/src/default/linear_transformation.rs`](../poulpy-core/src/default/linear_transformation.rs)):
> the module root re-exports focused files for indexing (§3), preparation (§5), hoisted
> baby rotations (§6.2), prepared inner products (§6.3), lazy giant rotations (§6.3), and
> final normalization (§6.4). The one-shot borrowed API prepares a temporary transform
> internally; the prepared evaluator implements the hoisted BSGS strategy described here. The
> CKKS API
> [`poulpy-ckks/src/default/linear_transformation.rs`](../poulpy-ckks/src/default/linear_transformation.rs)
> owns all `log_delta`/`log_budget` math: it derives the convolution alignment and result
> metadata, then delegates to the core engine.

---

## 1. Goal

Given:

- a CKKS ciphertext `ct` of rank `r` encrypting a slot vector `v` (`r + 1` polynomial
  columns: column `0` is the body `b`, columns `1..=r` are the mask `a_1..a_r`; rank
  `1` is the common case, `(b, a)`);
- a linear map `M` over the `m` slots, given by its set of non-zero generalized
  diagonals `{u_i}_{i ∈ I}` (each `u_i` is a slot vector, `u_i[ℓ] = M[ℓ, ℓ + i mod m]`);

produce a ciphertext encrypting `M · v`, consuming a single multiplicative (rescale)
level, using `O(√|I|)` key-switches.

---

## 2. The diagonal method

A slot-domain linear map is a sum of diagonals times rotations:

```
M · v = Σ_{i ∈ I} u_i ⊙ rot(v, i)
```

where `⊙` is the slot-wise (plaintext × ciphertext) product and `rot(v, i)` is the
cyclic slot rotation by `i`, realized homomorphically by a key-switched ring
automorphism. The naïve cost is `|I|` rotations (key-switches) and `|I|` rescales.

---

## 3. BSGS factorization

Pick a baby-step count `n1` (a tunable parameter, see §10). Factor every diagonal index
as `i = n1·j + k` with the baby index `k ∈ [0, n1)` and the giant index
`j ∈ [0, n2)`, `n2 = ⌈(max_i + 1)/n1⌉`. Using
`rot(v, n1·j + k) = rot(rot(v, k), n1·j)` and the fact that rotation distributes over
`⊙` and composes additively:

```
M · v = Σ_j  rot(  Σ_k  ũ_{j,k} ⊙ rot(v, k),  n1·j )
                   └────── inner sum (PROD, §6.3) ──────┘
        └─────────────── rotate (ROT) and accumulate ───────────────┘

       ũ_{j,k} := rot( u_{n1·j + k}, −n1·j )           (pre-rotated diagonals)
```

The pre-rotated diagonals `ũ_{j,k}` are **plaintexts**: they are encoded and prepared
once at setup. Two rotation families remain:

- the **`n1` baby-step rotations** `rot(v, k)` — computed once, reused by every giant
  step (hoisted, §6.2);
- the **`n2` giant-step rotations** `rot(·, n1·j)` — one per giant step (§6.3).

This replaces `|I|` rotations with `(n1 − 1) + (n2 − 1)` rotations, minimized at
`n1 ≈ n2 ≈ √|I|`.

---

## 4. Representations and primitives

Three coefficient-domain representations of a polynomial column are used, plus the
prepared-convolution and DFT forms:

| Name | Poulpy type | Role |
|---|---|---|
| `SMALL` | [`VecZnx`](../poulpy-hal/src/layouts/vec_znx.rs) | normalized base-`2^K` limbs; required input to gadget decomposition and to `cnv_prepare_*` |
| `BIG` | [`VecZnxBig`](../poulpy-hal/src/layouts/vec_znx_big.rs) | un-normalized extended-precision accumulator; supports exact add and automorphism without normalizing |
| DFT | [`VecZnxDft`](../poulpy-hal/src/layouts/vec_znx_dft.rs) | NTT/FFT domain; where VMP and convolution products live |
| `CnvPVecL` / `CnvPVecR` | [`convolution.rs`](../poulpy-hal/src/api/convolution.rs) | a polynomial prepared as the left / right operand of a bivariate convolution |

Primitives (all already implemented in the HAL):

| Step | Primitive | Note |
|---|---|---|
| DFT (mask) | `vec_znx_dft_apply` | `SMALL → DFT`; shared across baby keys (hoisting) |
| VMP | `gglwe_product_dft` / `vmp_apply_dft_to_dft` | gadget product against an automorphism key, in DFT domain |
| IDFT | `vec_znx_idft_apply` | `DFT → BIG` |
| add body (small) | `vec_znx_big_add_small_assign` | `BIG += SMALL` |
| add body (big) | `vec_znx_big_add_assign` | `BIG += BIG` (giant-step carry, §6.3) |
| normalize | `vec_znx_big_normalize` | `BIG → SMALL` |
| automorphism (small) | `vec_znx_automorphism` | permutation `X → X^p` on `SMALL` |
| automorphism (big) | `vec_znx_big_automorphism_assign` | permutation on `BIG` (deferred-norm path) |
| prepare diagonal | `cnv_prepare_right` | plaintext → `CnvPVecR`, at setup |
| prepare rotation | `cnv_prepare_left` | `SMALL → CnvPVecL` |
| convolution | `cnv_apply_dft` | `CnvPVecL ⊗ CnvPVecR → DFT`; **overwrites** its target column |
| DFT accumulate | `vec_znx_dft_add_assign` / `vec_znx_dft_zero` | accumulate convolution products |

> `cnv_apply_dft` overwrites the destination DFT column (it calls a `reim4_save`, not an
> accumulate). The inner sum over baby steps must therefore be accumulated explicitly:
> convolve into a scratch DFT column, then `vec_znx_dft_add_assign` into the accumulator
> (or write the first term and add the rest).

The composite `VMP → IDFT → add-body → normalize → automorphism` is exactly the GLWE
automorphism in
[`poulpy-core/.../automorphism/glwe.rs`](../poulpy-core/src/default/automorphism/glwe.rs);
the baby-step rotation reuses `glwe_automorphism_default` verbatim, the giant-step
rotation needs one new deferred-normalization variant (§11).

---

## 5. Inputs and precomputation

Done once, at key/transform setup (not per evaluation):

1. **Automorphism keys.** A GGLWE automorphism key `autokey[gal(k)]` for every distinct
   baby galois element `gal(k)`, `k ∈ {1,…,n1−1}`, and `autokey[gal(n1·j)]` for every
   distinct giant galois element, `j ∈ {1,…,n2−1}`. (`gal` is the slot-rotation galois
   map; in Poulpy the key carries it via `GetGaloisElement::p()`.) `k = 0` and `j = 0`
   are the identity — **no keys needed**.
2. **Prepared diagonals.** For every `(j, k)` with `n1·j + k ∈ I`: encode the
   pre-rotated diagonal `ũ_{j,k} = rot(u_{n1·j+k}, −n1·j)` as a plaintext polynomial and
   prepare it with `cnv_prepare_right` into a `CnvPVecR`. Store indexed by `(j, k)`.
   Zero diagonals are omitted (§8).

The prepared linear transformation stores non-empty giant steps; each giant step
maps the real baby rotation `k` to its prepared diagonal `CnvPVecR`. The baby
rotation list and required rotations are derived from those real indexes.

---

## 6. Algorithm

### 6.1 Overview

```
A. Baby steps  : hoist the input once; produce {rot(v,k)}_{k} as prepared CnvPVecL.
B. Giant steps : for each j, PROD = Σ_k ũ_{j,k} ⊙ rot(v,k); ROT it by n1·j; accumulate in BIG.
C. Finalize    : one normalization of the BIG accumulator → output ciphertext.
```

Notation below is for rank `r`; columns are indexed `c ∈ {0,…,r}` (`0` = body).

### 6.2 Phase A — hoisted baby-step rotations

```
# Optional base-2^K alignment of ct to the key's base2k (as glwe_keyswitch does).
ct_b   = ct[0]                                  # SMALL, the shared body
a_dft  = [ vec_znx_dft_apply(ct[c]) for c in 1..=r ]   # DFT of the mask, computed ONCE

for k in 0 .. n1:
    if k == 0:
        rot_k = ct                              # identity: no key-switch, no automorphism
    else:
        g = gal(k)
        res_dft = gglwe_product_dft(a_dft, autokey[g])     # VMP, reuses a_dft  → (r+1) cols DFT
        res_big = [ vec_znx_idft_apply(res_dft[c]) for c in 0..=r ]   # BIG
        vec_znx_big_add_small_assign(res_big[0], ct_b)     # add shared body
        rot_small = [ vec_znx_big_normalize(res_big[c]) for c in 0..=r ]   # SMALL
        rot_k = [ vec_znx_automorphism(g, rot_small[c]) for c in 0..=r ]   # permute slots
    # Prepare for the plaintext multiplications; reused across all giant steps.
    L[k][c] = cnv_prepare_left(rot_k[c])  for c in 0..=r        # CnvPVecL
```

`a_dft` is the single hoisted object: the `n1 − 1` non-trivial baby rotations all reuse
it, differing only in the VMP key. After Phase A we hold the `n1` prepared rotations
`L[0..n1]`.

### 6.3 Phase B — giant-step accumulation and rotation

```
acc_big = zero (r+1 columns, BIG)               # final BIG accumulator

for j in 0 .. n2:
    # ---- PROD: inner sum Σ_k ũ_{j,k} ⊙ rot(v,k) ----
    for c in 0 ..= r:
        prod_dft[c] = 0 (DFT)
        first = true
        for k in 0 .. n1 with (n1*j + k) in I:                 # skip zero diagonals
            R = diagonals[(j,k)]                               # CnvPVecR (precomputed)
            if first: cnv_apply_dft(prod_dft[c], L[k][c], R); first = false
            else:     cnv_apply_dft(tmp_dft, L[k][c], R); vec_znx_dft_add_assign(prod_dft[c], tmp_dft)
        prod_big[c] = vec_znx_idft_apply(prod_dft[c])          # BIG
    # prod_big = (B_j, A_1..A_r), all BIG

    # ---- ROT: rotate prod_big by n1*j and accumulate ----
    if j == 0:
        vec_znx_big_add_assign(acc_big, prod_big)              # identity giant step, stay BIG
    else:
        g = gal(n1*j)
        A_small = [ vec_znx_big_normalize(prod_big[c]) for c in 1..=r ]   # mask → SMALL (for decomposition)
        a_dft   = [ vec_znx_dft_apply(A_small[c]) for c in 1..=r ]
        ks_dft  = gglwe_product_dft(a_dft, autokey[g])         # VMP
        ks_big  = [ vec_znx_idft_apply(ks_dft[c]) for c in 0..=r ]        # BIG
        vec_znx_big_add_assign(ks_big[0], prod_big[0])         # carry body in BIG (BIG += BIG)
        for c in 0..=r: vec_znx_big_automorphism_assign(g, ks_big[c])     # permute on BIG, no add-after
        vec_znx_big_add_assign(acc_big, ks_big)
```

The body column `B_j` is **never normalized** between steps: it rides the `BIG`
accumulator. Only the mask `A_j` is normalized — and only because gadget decomposition
requires `SMALL` limbs. For `j = 0` even that is skipped.

### 6.4 Phase C — finalize

```
res = [ vec_znx_big_normalize(acc_big[c]) for c in 0..=r ]     # the ONE normalization
return res                                                     # SMALL, encrypts M·v
```

---

## 7. Correctness

Let `KS_g` be the key-switch with `autokey[g]` and `φ_g` the ring automorphism, so that
`rot(ct, k) = φ_{gal(k)}(KS_{gal(k)}(ct))` (Poulpy's key-switch-then-permute order, see
`glwe_automorphism_default`). Hoisting only shares the input DFT `a_dft` across the
per-`k` VMPs; it computes the identical `φ_g(KS_g(ct))`, so each `L[k]` is a faithful
encryption of `rot(v, k)`.

For a giant step `j`, `prod_big` encrypts `m_j = Σ_k ũ_{j,k} ⊙ rot(v,k)` under the
input key `s`: each `cnv_apply_dft(L[k][c], R_{j,k})` is the limb-exact product of
column `c` of `rot(v,k)` with the plaintext `ũ_{j,k}`, and the DFT-domain sum over `k`
is `m_j`'s column `c` (still un-normalized in `BIG`). The `ROT` block computes
`φ_g(KS_g(prod_big)) = rot(prod_big, n1·j)`: `KS_g` processes the mask `A_j` (hence the
`SMALL` normalization), `ks_big[0] += B_j` re-injects the carried body exactly (the
key-switch leaves the body additive), and `φ_g` permutes every column. Permutation and
addition on `BIG` are exact integer operations, independent of normalization, so
deferring the limb reduction to Phase C changes nothing but rounding placement.
Summing over `j` (with the `j = 0` identity term added directly) yields
`Σ_j rot(m_j, n1·j) = M·v` by §3. ∎

The only approximation is the final `vec_znx_big_normalize` (one rounding), plus the
key-switch and plaintext-multiply noise inherent to CKKS.

---

## 8. Where the savings are

| # | Saving | Mechanism |
|---|---|---|
| 1 | **Fewer rotations** `O(√|I|)` vs `O(|I|)` | BSGS factorization (§3); choose `n1 ≈ √|I|` |
| 2 | **Hoisting** | DFT/decompose the input mask once (`a_dft`), reuse for all `n1 − 1` baby VMPs (§6.2) |
| 3 | **Free identities** | `k = 0` and `j = 0` are the identity: skip 2 key-switches, 2 automorphisms, and their keys (§5–6) |
| 4 | **Lazy normalize across giant steps** | body `b` stays in `BIG`; a single `vec_znx_big_normalize` at the end instead of one per giant step (§6.3–6.4) |
| 5 | **Lazy normalize inside PROD** | accumulate the `n1` plaintext products in the DFT domain, one IDFT per giant step instead of one per product (§6.3) |
| 6 | **Skip `j = 0` mask normalize** | the `j = 0` accumulator stays fully `BIG` (no `ROT`), saving `r` normalizations (§6.3) |
| 7 | **Prune zero diagonals** | iterate only over `(j,k)` with `n1·j + k ∈ I`; empty baby/giant steps drop their keys and convolutions entirely (§5, §6.3) |
| 8 | **Setup-time preparation** | diagonals prepared into `CnvPVecR` once; each `rot(v,k)` prepared into a `CnvPVecL` map keyed by the real baby rotation `k` and reused across all `n2` giant steps (§5, §6.2) |
| 9 | **One rescale level** | all products share the input/diagonal scale and are summed before normalization, so the whole transform costs a single rescale, not `|I|` |

Further (out of scope here, noted for later):

- **Structure-aware `n1`.** Choosing `n1` to divide the diagonal stride (e.g. for the
  FFT/`CoeffsToSlots` matrices) makes the `(j,k)` grid sparse and minimizes the number
  of *distinct* giant rotations.
- **Double hoisting** of the giant-step key-switches (Bossuat et al., 2021) — the giant
  inputs `A_j` differ, so they cannot share a decomposition directly, but the
  decomposition basis-change can be partly hoisted. Requires restructuring Phase B.
- **Conjugate symmetry** for real-valued maps, and **merging adjacent transforms**
  (e.g. multi-level `CoeffsToSlots`) into one BSGS pass.

**Headroom caveat.** Savings 4–6 are bounded by the width of the `BIG` scalar: the
accumulator absorbs the `n1`-term convolution sum *and* the `n2`-term giant sum without
normalizing. For large `n1·n2` and small base-`2^K`, verify the accumulation fits the
backend's `ScalarBig` (e.g. `i128` for NTT120); if not, insert an intermediate
`vec_znx_big_normalize` of the body. This is the one place laziness has a hard limit.

---

## 9. Cost and level budget

For a transform whose `|I|` non-zero diagonals factor as `n1 × n2`:

| Quantity | Count |
|---|---|
| Input DFT (hoist) | `1` |
| Baby-step key-switches (VMP+IDFT+norm+auto) | `n1 − 1` |
| Plaintext convolutions (`cnv_apply_dft`) | `|I|` (one per non-zero diagonal) |
| Giant-step key-switches (`ROT`) | `n2 − 1` |
| `vec_znx_big_normalize` of the output | `1` (+ `r·(n2−1)` mask normalizations inside `ROT`) |
| Rescale levels consumed | `1` |
| Distinct automorphism keys | `(n1 − 1) + (n2 − 1)`, minus pruned-empty steps |
| Live prepared operands | one `CnvPVecL` per distinct used baby rotation + `|I|` `CnvPVecR` |

Total key-switches `(n1 − 1) + (n2 − 1)` is minimized at `n1 ≈ n2 ≈ √|I|`.

---

## 10. Parameters and tuning

- **`n1`** (baby steps): default `n1 = round(√|I|)`; expose it so callers can align it
  with the diagonal structure of `M`. `n2 = ⌈(max_i + 1)/n1⌉`.
- **Key sizes / `dnum`,`dsize`**: the automorphism keys' gadget parameters set the
  key-switch noise and the `BIG` headroom; reuse the project's existing GGLWE key
  layout conventions.
- **Diagonal encoding scale `Δ_pt`**: chosen so the post-multiply scale `Δ_ct · Δ_pt`
  matches the target rescale; the `cnv_offset` argument of `cnv_apply_dft` aligns the
  limb (the `Y = 2^{-K}` axis) when the diagonals span multiple limbs.

---

## 11. Poulpy integration

**Placement.** The engine is GLWE-level, so it lives in **poulpy-core**, not poulpy-ckks.
This is also forced by visibility: the gadget product
(`gglwe_product_dft` / `glwe_keyswitch_internal`) and the prepared key's `VmpPMat`
(`GGLWEPrepared.data`) are `pub(crate)` to poulpy-core, and the only *public* VMP-bearing
routines (`glwe_keyswitch`, `glwe_automorphism`) re-DFT the input internally and bundle
the IDFT+normalize — i.e. they expose neither the hoisting seam nor the lazy-normalize
seam this algorithm needs. From inside poulpy-core the right primitives are reachable;
from poulpy-ckks they are not.

- **Core engine.** `poulpy-core/src/default/linear_transformation/` mirrors the
  `automorphism` and `keyswitching` split. It exposes prepared-transform evaluation
  through `glwe_prepared_linear_transform`, which takes a prepared transform, prepared
  baby rotations, and the keyed set of GGLWE automorphism keys (`GetGaloisElement`).
  Prepared baby rotations are addressed by the real baby-step rotation `k`, matching
  the BSGS index, not by a dense local list position. The prepared
  `LinearTransformation<BE>` data type also lives core-side.
- **CKKS wrapper.** `LinearTransformationOps::ckks_eval_linear_transformation_into` /
  `_assign` in
  [`poulpy-ckks/src/api/linear_transformation.rs`](../poulpy-ckks/src/api/linear_transformation.rs)
  becomes thin: encode the diagonals (CKKS scale/`Δ_pt`), build the galois-element map,
  and call the core engine. No FHE arithmetic in the CKKS layer.
- **VMP reuse — do not fork.** Both the baby- and giant-step rotations get their gadget
  product from `gglwe_product_dft` applied to a hoisted `a_dft` (DFT computed once with
  `vec_znx_dft_apply`, reused per key). For `dsize == 1` this is a single HAL call,
  `vmp_apply_dft_to_dft` (a fused `vmp_apply_dft_to_dft_accumulate` also exists); for
  `dsize > 1` it is the digit-decomposition loop already implemented and tested in
  `gglwe_product_dft_default`
  ([keyswitching/glwe.rs:123](../poulpy-core/src/default/keyswitching/glwe.rs)) — reuse
  it, do not reimplement it in HAL.
- **Glue is raw HAL.** Everything around the VMP is HAL: `vec_znx_idft_apply`,
  `vec_znx_big_add_small_assign` / `vec_znx_big_add_assign`, `vec_znx_automorphism` /
  `vec_znx_big_automorphism_assign`, `vec_znx_big_normalize`, and the `Convolution` trait
  (`cnv_apply_dft`, `vec_znx_dft_add_assign`, `vec_znx_idft_apply`) for `PROD`. This is
  the same "reused VMP + HAL glue" construction style as `glwe_automorphism_add_default`.
- **New helper needed.** Phase B's `ROT` is a *deferred-normalization plain
  automorphism*: `gglwe_product_dft → idft → big_add(carry body) → big_automorphism`,
  with **no** final normalize and **no** add-after-automorphism. It is a small variant
  of `glwe_automorphism_add_default`
  ([automorphism/glwe.rs:115](../poulpy-core/src/default/automorphism/glwe.rs)) — that
  function has the right `BIG`-automorphism mechanics but the wrong (`res = a + φ(KS a)`)
  add semantics; the BSGS path wants plain `res = φ(KS a)` left un-normalized in `BIG`.
  Factor it next to the existing automorphism variants so it can share their scratch
  layout.
- **Scratch.** Size scratch for: `a_dft` (hoisted), one `(r+1)`-column `BIG` result, one
  `(r+1)`-column DFT accumulator + one DFT temp (for the convolution accumulate), the
  `BIG` final accumulator, plus the max of the VMP / IDFT / normalize / convolution
  `*_tmp_bytes`. Follow the additive-layout pattern of
  `glwe_keyswitch_tmp_bytes_default`.
- **Hoisting seam.** Factor the "DFT the mask once, `gglwe_product_dft` per key" loop so
  both Phase A (many baby keys, one input) and any future hoisted-giant-step optimization
  can share it.

---

## 12. Validation

- **Functional.** Random `M` and `v`: compare decrypted `M·v` to the plaintext
  reference within the expected CKKS precision (cf. the `ckks_poly2` example's error
  thresholds).
- **Identity / permutation `M`.** Exercises `k = 0`, `j = 0`, and single-diagonal paths.
- **Sparse `M`.** Confirms zero-diagonal pruning (saving 7) and that empty steps drop
  their keys.
- **`n1` sweep.** Vary `n1` from `1` (pure giant) to `|I|` (pure baby); result must be
  invariant, cost should bottom out near `√|I|`.
- **Headroom.** A large `n1·n2` / small base-`2^K` case to confirm the `BIG` accumulator
  does not overflow (saving 4 caveat), with and without the intermediate normalize.
