# CKKS Linear Transformation — Default Implementation Walkthrough

Companion to [`lt_bsgs.md`](lt_bsgs.md). That document is the *specification* of the
baby-step / giant-step (BSGS) homomorphic linear transformation; this one is a guided
tour of the **reference implementation** that realizes it, file by file. Read the spec
first for the math (the diagonal method §2, the BSGS factorization §3, the savings table
§8); read this for *where each piece lives, how data flows, and where the code goes
beyond the spec*.

All paths below are under
[`poulpy-core/src/default/linear_transformation/`](../poulpy-core/src/default/linear_transformation.rs)
unless noted. The scheme-agnostic data types and the BSGS schedule derivation live in
[`poulpy-core/src/layouts/`](../poulpy-core/src/layouts/linear_transformation.rs); the
prepared (convolution-domain) caches live in
[`poulpy-core/src/layouts/prepared/`](../poulpy-core/src/layouts/prepared/linear_transformation.rs).

---

## 1. Map of the module

The engine is split so a backend only pulls in the HAL/op bounds a given method needs,
and so the layout/schedule math stays free of any backend.

| File | Role | Spec § |
|---|---|---|
| [`linear_transformation.rs`](../poulpy-core/src/default/linear_transformation.rs) | Module root: re-exports the data types and the `*_default` reference functions a backend forwards to. | §11 |
| [`prepare.rs`](../poulpy-core/src/default/linear_transformation/prepare.rs) | **Setup, RHS:** encode-then-`cnv_prepare_right` the matrix diagonals into `CnvPVecR`. | §5 |
| [`baby_steps.rs`](../poulpy-core/src/default/linear_transformation/baby_steps.rs) | **Setup, LHS / Phase A:** hoisted baby rotations `rot(v,k)`, prepared as `CnvPVecL`. | §6.2 |
| [`inner_product.rs`](../poulpy-core/src/default/linear_transformation/inner_product.rs) | **Phase B, PROD:** per-giant DFT-domain inner sum `Σ_k ũ_{j,k} ⊙ rot(v,k)`. | §6.3 |
| [`lazy.rs`](../poulpy-core/src/default/linear_transformation/lazy.rs) | **Phase B, ROT + Phase C:** lazy DFT giant rotation, DFT accumulate helpers, final normalize. | §6.3–§6.4 |
| [`prepared_giants.rs`](../poulpy-core/src/default/linear_transformation/prepared_giants.rs) | **Phase B driver:** the giant-step loop `glwe_eval_giant_steps`; dispatches lazy vs fallback; the `LinearTransformationRhs` trait that unifies prepared and streamed RHS. | §6.1, §6.3 |
| [`eval.rs`](../poulpy-core/src/default/linear_transformation/eval.rs) | **Public entry points** + scratch sizing (`*_tmp_bytes`). Thin forwards into the files above. | §11 |
| [`tests.rs`](../poulpy-core/src/default/linear_transformation/tests.rs) | Functional / identity / sparse / `n1`-sweep / headroom validation. | §12 |

### Data types (in `layouts`)

- [`LinearTransformationLayout`](../poulpy-core/src/layouts/linear_transformation.rs) —
  the *integer-only* spec of a transform: `{ indexes, slots, strategy }`. `.index()` /
  `.plan(n1)` derive a `LinearTransformationPlan` (`baby_steps`, `giant_steps`, and the
  `index[g]` grouping of baby rotations per giant step). This is where the BSGS
  factorization §3 happens.
- [`LinearTransformation<P>`](../poulpy-core/src/layouts/linear_transformation.rs) — the
  unprepared transform: schedule metadata plus the *encoded* (but not convolution-prepared)
  diagonal plaintexts `P`.
- [`LinearTransformationRhsPrepared<BE>`](../poulpy-core/src/layouts/prepared/linear_transformation.rs)
  — the prepared **right** operand: pruned giant steps, each holding `CnvPVecR` diagonals
  keyed by the real baby rotation `k`, plus the plaintext limb layout the evaluator needs.
- [`LinearTransformationBabySteps<BE>`](../poulpy-core/src/layouts/prepared/linear_transformation.rs)
  — the prepared **left** operand: a `BTreeMap<i64, CnvPVecL>` of baby rotations
  `rot(v,k)`, addressed by the real BSGS index `k` (not a dense position).

---

## 2. The BSGS schedule (Phase 0, setup)

Everything starts from `LinearTransformationLayout::index()`, which calls
[`linear_transformation_plan`](../poulpy-core/src/layouts/linear_transformation.rs#L215).
Given `slots` and a giant step `n1`, it factors each normalized diagonal `i` as
`i = n1·j + k`:

```
baby_rot  = i % n1           # k ∈ [0, n1)
giant_rot = i - baby_rot     # n1·j
```

It accumulates the distinct baby rotations (always including `0`), groups the real baby
rotations under each giant rotation, and de-duplicates. The result `LinearTransformationPlan`
already realizes **saving #7 (prune zero diagonals)**: only `(j,k)` pairs that correspond
to a real diagonal are present, so empty baby/giant steps simply never appear and never
get a key.

**Choosing `n1` (`LinearTransformationStrategy`, spec §10).**

- `Direct` — one giant step per diagonal, no baby rotations (best for ≤ 2 diagonals).
- `Bsgs { giant_step }` — caller-fixed `n1`. This is the "expose it so callers can align
  it with the diagonal structure" knob.
- `Auto` — `≤ 2` diagonals ⇒ `Direct`, else
  [`optimal_bsgs_giant_step`](../poulpy-core/src/layouts/linear_transformation.rs#L256).

> **Beyond the spec.** The spec §10 defaults to `n1 = round(√|I|)`; §8 lists *structure-aware*
> `n1` as future work. `optimal_bsgs_giant_step` already implements the structure-aware
> version: it only tries giant steps that are multiples of the **minimum gap** between
> consecutive sorted diagonals (cheap for stride-`k` matrices like `CoeffsToSlots`), and
> picks the one minimizing `(n1 + n2) + |n1 − n2|` — i.e. `√|I|` balanced, but snapped to
> the matrix's natural stride.

---

## 3. Setup — preparing the operands

The BSGS evaluation is a bivariate convolution `M·v = Σ_k baby_k ⊗ diagonal_k`: the baby
rotations are the **left** operand, the diagonals the **right** operand. Both are prepared
once, at setup (**saving #8**).

### 3.1 RHS — diagonals → `CnvPVecR` ([`prepare.rs`](../poulpy-core/src/default/linear_transformation/prepare.rs))

`LinearTransformationRhsPrepared::alloc{,_from_index}` sizes the cache from the schedule
and a plaintext-shape proxy (one `CnvPVecR(1, pt_size)` per real diagonal), recording the
diagonals' `pt_base2k` / `pt_max_k` so the evaluator never needs the raw transform again.

`glwe_prepare_linear_transformation_rhs_default` then fills each pre-allocated slot with
`cnv_prepare_right(plaintext)`. Zero allocations happen here; it only populates. The
diagonals are expected pre-encoded (and pre-rotated `ũ_{j,k} = rot(u_{n1·j+k}, −n1·j)`) by
the CKKS layer — the core engine is scheme-agnostic.

### 3.2 LHS — baby rotations → `CnvPVecL` ([`baby_steps.rs`](../poulpy-core/src/default/linear_transformation/baby_steps.rs), Phase A)

`glwe_prepare_linear_transformation_lhs` materializes `rot(v,k)` for every baby `k` in the
cache. This is **Phase A** of the spec, and it carries three savings:

- **Saving #3 (free identity).** `k == 0` skips the key-switch and automorphism entirely:
  the input ciphertext `a` is prepared directly into `CnvPVecL`.
- **Saving #2 (hoisting).** For the non-trivial `k`, the mask columns are DFT'd **once**
  into `a_dft` ([baby_steps.rs:233-241](../poulpy-core/src/default/linear_transformation/baby_steps.rs#L233-L241)),
  and every per-`k` VMP reuses that single `a_dft_ref` — only the automorphism key differs.
  Each rotation is then `VMP → IDFT → add body → normalize → automorphism`
  (`glwe_hoisted_baby_rotation`), which is exactly the GLWE automorphism mechanics
  (spec §4), inlined so the hoisting seam is exposed.
- The hoisted route is taken only when the input and key share a `base2k`
  ([baby_steps.rs:225-230](../poulpy-core/src/default/linear_transformation/baby_steps.rs#L225-L230));
  otherwise it falls back to the public `glwe_automorphism` per baby (still correct, just
  un-hoisted).

The result of Phase A is `LinearTransformationBabySteps`: `n1` prepared `CnvPVecL`
rotations reused across **all** `n2` giant steps.

---

## 4. Evaluation — the giant-step loop ([`prepared_giants.rs`](../poulpy-core/src/default/linear_transformation/prepared_giants.rs))

`glwe_eval_giant_steps` is the heart of Phase B/C. It is generic over the `LinearTransformationRhs`
trait so the **same loop** drives both RHS flavors:

- `LinearTransformationRhsPrepared` — the materialized cache (prepared path).
- `LinearTransformation<P>` — streams each diagonal through scratch (unprepared path, §6).

Only the per-giant `accumulate_prod` differs; the rotate/accumulate tail is shared.

### 4.1 Path selection

The loop picks one of two strategies up front
([prepared_giants.rs:218-236](../poulpy-core/src/default/linear_transformation/prepared_giants.rs#L218-L236)):

- **Lazy DFT path** (the hot path) — taken when `res`, the PROD output, and the keys all
  share a `base2k` (or when there is no giant rotation at all). Everything rides in DFT
  through the whole loop; one IDFT + one normalize at the very end.
- **Fallback path** — taken on base mismatch. Still computes PROD in DFT, but normalizes
  each giant contribution to SMALL and uses the public normalized `glwe_automorphism`. It
  is *correct* but gives up savings 4–6; it exists only so mismatched bases don't break.

### 4.2 PROD — the inner sum ([`inner_product.rs`](../poulpy-core/src/default/linear_transformation/inner_product.rs))

For each giant step `j`, `glwe_accumulate_prepared_baby_steps_dft` computes
`Σ_k ũ_{j,k} ⊙ rot(v,k)` and **leaves it in `VecZnxDft`**. Because `cnv_apply_dft`
*overwrites* its target, the baby loop is the outer loop: the first baby `copy`s into
every output column, the rest `vec_znx_dft_add_assign` into it. Keeping the whole inner
sum in DFT with a single IDFT deferred to later is **saving #5 (lazy normalize inside
PROD)**.

The streamed sibling `glwe_accumulate_unprepared_baby_steps_dft` is identical except it
`cnv_prepare_right`s each diagonal on the fly into one reused scratch `CnvPVecR` — see §6.

### 4.3 ROT — the lazy giant rotation ([`lazy.rs`](../poulpy-core/src/default/linear_transformation/lazy.rs))

This is where the implementation is **lazier than the spec**. Spec §11 asked for a
*deferred-normalization BIG automorphism*: `idft → big_add(body) → big_automorphism`, left
un-normalized in `BIG`. The implementation, `glwe_lazy_giant_automorphism_from_dft`, keeps
the giant rotation **entirely in DFT**:

1. Only the **mask** columns are dropped to SMALL (`idft → normalize`) — unavoidable,
   because gadget decomposition needs limb-aligned input.
2. VMP (`gglwe_product_dft`) against the giant automorphism key.
3. The **body** is carried in DFT with `vec_znx_dft_add_assign` — never normalized between
   steps (**saving #4**).
4. The automorphism is applied with `vec_znx_dft_automorphism` (DFT permutation), not a
   BIG permutation.

For the identity giant step `j == 0` (**saving #6**), ROT is skipped entirely: PROD is
folded straight into the accumulator with no mask normalize.

### 4.4 Cross-giant accumulation + Phase C (finalize)

Each giant contribution (rotated, or PROD directly for `j == 0`) is folded into a single
DFT accumulator `lazy_acc_dft` via `glwe_dft_copy_dft` (first) / `glwe_dft_add_dft_assign`
(rest). After the loop there is exactly **one** `glwe_idft_dft_into_big` and **one**
`glwe_normalize_big_into` — the single rounding the spec allows (**saving #4 + #9, one
rescale level**). The sub-limb `cnv_offset_lo` shift PROD never applied is folded into
this final normalize.

### 4.5 Summary of where each spec saving lands

| Saving (spec §8) | Implementation site |
|---|---|
| #1 BSGS `O(√\|I\|)` | `linear_transformation_plan` + `optimal_bsgs_giant_step` |
| #2 Hoisting | `glwe_prepare_linear_transformation_lhs` (`a_dft` computed once) |
| #3 Free identities | `rot == 0` branches in baby_steps.rs **and** prepared_giants.rs |
| #4 Lazy normalize across giants | DFT accumulator + single final normalize (lazy.rs) |
| #5 Lazy normalize inside PROD | DFT accumulation in inner_product.rs |
| #6 Skip `j=0` mask normalize | `rot == 0` giant branch (prepared_giants.rs:258) |
| #7 Prune zero diagonals | schedule only stores real `(j,k)` (linear_transformation.rs) |
| #8 Setup-time preparation | prepare.rs (RHS) + baby_steps.rs (LHS) |
| #9 One rescale level | single `glwe_normalize_big_into` (lazy.rs) |

---

## 5. Public entry points + scratch ([`eval.rs`](../poulpy-core/src/default/linear_transformation/eval.rs))

These `*_default` free functions are what a backend forwards to from its
`crate::oep::LinearTransformationDefault` impl (via `impl_linear_transformation_defaults_full!`):

| Function | Does |
|---|---|
| `glwe_prepare_linear_transformation_rhs_default` | Setup §3.1 (prepare.rs). |
| `glwe_prepare_linear_transformation_lhs_default` | Setup §3.2 / Phase A (baby_steps.rs). |
| `glwe_eval_linear_transformation_into_default` | Prepared eval (§4) — asserts ≥ 1 non-empty giant step, then `glwe_eval_giant_steps`. |
| `glwe_eval_linear_transformation_unprepared_rhs_into_default` | Streamed eval (§6). |
| `*_tmp_bytes_default` siblings | Scratch sizing. |

**Scratch sizing.** The `*_tmp_bytes` functions follow the additive-layout pattern of
`glwe_keyswitch_tmp_bytes_default`: they size each route (lazy DFT vs fallback, hoisted vs
plain) and take the `max`. The eval budget covers the hoisted `a_dft`, the DFT PROD buffer
+ DFT temp, the DFT giant accumulator, the rotation scratch, the final BIG accumulator, and
the per-op VMP/IDFT/normalize/convolution `*_tmp_bytes`. The streamed variant adds one
resident `CnvPVecR` slot and a `cnv_prepare_right` scratch on top.

---

## 6. The streamed (unprepared-RHS) variant

Not in the spec, but a useful complement to **saving #8**: a second evaluation entry point
that consumes the unprepared `LinearTransformation<P>` directly, preparing each diagonal on
the fly (one reused scratch `CnvPVecR`) instead of from a materialized
`LinearTransformationRhsPrepared`. Same result, lower peak memory, higher compute — aimed
at memory-bound backends (e.g. GPU). It reuses the entire giant-step loop unchanged via the
`LinearTransformationRhs` trait; only `accumulate_prod` swaps in the streaming inner product
(`glwe_accumulate_unprepared_baby_steps_dft`). The LHS (baby) cache is still prepared once.

---

## 7. Known caveats / limits

- **Headroom (spec §8 caveat / §12 test) is not auto-guarded.** The lazy accumulator stays
  fully in DFT with no overflow check; for very large `n1·n2` and small base-`2^K`,
  correctness depends on the caller's parameters fitting the backend's `ScalarBig`. The
  spec's suggested "intermediate `vec_znx_big_normalize` of the body" escape valve is not
  wired in.
- **Base-mismatch disables the lazy optimizations.** When `res`/`prod`/`key` base2k differ,
  eval silently takes the fallback path (§4.1), losing savings 4–6. The common,
  base-aligned case always takes the lazy path.
- **Out-of-scope items remain unimplemented** (as the spec intends): double hoisting of the
  giant-step key-switches, conjugate symmetry for real-valued maps, and merging adjacent
  transforms.

---

## 8. CKKS wrapper

The CKKS layer
([`poulpy-ckks/src/api/linear_transformation.rs`](../poulpy-ckks/src/api/linear_transformation.rs),
[`poulpy-ckks/src/default/linear_transformation.rs`](../poulpy-ckks/src/default/linear_transformation.rs))
stays thin per spec §11: it owns all `log_delta` / `log_budget` / `cnv_offset` math,
encodes and pre-rotates the diagonals, builds the galois-element → key map, and delegates
every FHE operation to the core engine above. No FHE arithmetic lives in the CKKS layer.
