# Prepared-Domain Apply API Plan

This document lists the proposed HAL API surface for operations whose reusable
operand can be consumed in one of three forms:

```text
small = coefficient layout
t     = transformed hot-prep layout
p     = packed cold-prep layout
```

`t` is the low-latency prepared form: cheaper to build, intended for short reuse
or one-shot use. `p` is the packed/prepared form: more expensive to build, but
optimized for amortized repeated apply. The current `SvpPPol`, `VmpPMat`, and
`CnvPVecL/R` names remain the packed/cold-prep layouts.

The split is type-level even when a backend initially gives two layouts the same
physical storage shape. Prepared layouts are special read-only operand formats:
users create or overwrite them through prepare/copy APIs, then pass them by
reference to apply APIs.

## Naming

A method name lists the domain of each operand, then the output domain. The
operand order is fixed per family, tiered operand first:

```text
svp_apply_<scalar-domain>_<vector-domain>_to_<output-domain>
vmp_apply_<matrix-domain>_<vector-domain>_to_<output-domain>
cnv_apply_<left-domain>_<right-domain>_to_<output-domain>
```

Both operands are reusable: a `VecZnxDft` is as much a prepared form as an
`SvpPPol`. What separates the slots is that one operand carries the full
three-way prep tier (coefficient / transformed / packed) while the other only
distinguishes coefficient from DFT. The tiered operand leads, because its tier
is the axis you choose a method on. For CNV both operands are tiered, so the
order is positional (left, then right).

This means the name order matches argument order for SVP and CNV, but **not**
for VMP, whose signature takes the vector before the matrix:

```rust
svp_apply_ppol_dft_to_dft(res, res_col, a: &SvpPPol, a_col, b: &VecZnxDft, b_col)
vmp_apply_pmat_dft_to_dft(res, a: &VecZnxDft, b: &VmpPMat, limb_offset, scratch)
```

Each domain token names the layout actually passed, so the tier tokens differ
per family rather than being a shared `t` / `p` vocabulary:

| role | coefficient | transformed (hot-prep) | packed (cold-prep) |
| --- | --- | --- | --- |
| SVP scalar | `small` (`ScalarZnx`) | `tpol` (`SvpTPol`) | `ppol` (`SvpPPol`) |
| VMP matrix | `small` (`MatZnx`) | `tmat` (`VmpTMat`) | `pmat` (`VmpPMat`) |
| CNV left/right | `small` (`VecZnx`) | `tvec` (`CnvTVecL/R`) | `pvec` (`CnvPVecL/R`) |
| vector operand | `small` (`VecZnx`) | | `dft` (`VecZnxDft`) |

These stems are already how the codebase names these layouts (`svp_ppol_alloc`,
`bytes_of_vmp_pmat`, `cnv_pvec_left_alloc`), so the apply names introduce no new
vocabulary.

Output domains:

- `dft`: currently `VecZnxDft`.
- `big`: extended-precision output, obtained by IDFT from a DFT result.
- `small`: coefficient output, obtained by normalizing a big result.

Derived outputs:

```text
*_to_big   = IDFT(*_to_dft)
*_to_small = normalize(*_to_big)
```

The `*_to_small` APIs must carry the same normalization parameters used by
`VecZnxBigNormalize`: `res_base2k`, `res_offset`, and `a_base2k`.

## Layout Tiers

### SVP

**Status: implemented.** The resolved decisions for this family, applied as the
precedent for VMP and CNV:

- Hard break. The old names are gone, with no deprecated aliases.
- `_to_big` and `_to_small` are full OEP primitives, so a backend can fuse the
  IDFT and normalization. They are default-provided once in
  `poulpy-cpu-ref::hal_defaults::SvpDerivedDefault` (flavor-agnostic, built on
  `_to_dft` + `VecZnxIdftApplyTmpA` + `VecZnxBigNormalize`), so a backend pays
  for them only if it wants to specialize.
- The `small` rhs variants are implemented everywhere. They prepare into an
  `SvpTPol` on each call, so they carry no scratch parameter and are documented
  as one-shot paths.
- `SvpTPol` and `SvpPPol` share a physical layout on every current CPU backend,
  so `t` and `p` resolve to the same kernels. The split is type-level exactly as
  planned: the backend kernels take the prepared operand as a coefficient slice,
  and a backend with a cheaper hot-prep form specializes `svp_prepare_tpol` plus
  the `svp_apply_tpol_*` family without touching callers.
- Scratch sizing is two shared traits, `SvpApplyToBigTmpBytes` and
  `SvpApplyToSmallTmpBytes`, rather than one per variant: the cost does not vary
  with the rhs domain. The `_to_small` intermediate is carved at the *input*
  limb count so the product keeps full width until the normalization.

```text
small: ScalarZnx
tpol:  SvpTPol
ppol:  SvpPPol
```

`SvpTPol` is the hot-prep transformed scalar polynomial. `SvpPPol` keeps its
current meaning: scalar-vector-product prepared polynomial, the packed/cold
layout.

New hot-prep layout family:

```rust
SvpTPol
SvpTPolOwned<BE>
SvpTPolBackendRef<'a, BE>
SvpTPolBackendMut<'a, BE>
SvpTPolToBackendRef<BE>
SvpTPolToBackendMut<BE>
```

Existing cold-prep layout family:

```rust
SvpPPol
SvpPPolOwned<BE>
SvpPPolBackendRef<'a, BE>
SvpPPolBackendMut<'a, BE>
SvpPPolToBackendRef<BE>
SvpPPolToBackendMut<BE>
```

Preparation/allocation:

```rust
SvpTPolAlloc::svp_tpol_alloc(cols)
SvpTPolBytesOf::bytes_of_svp_tpol(cols)
SvpPrepareTPol::svp_prepare_tpol(res, res_col, a, a_col)
SvpTPolCopyBackend::svp_tpol_copy_backend(res, res_col, a, a_col)

SvpPPolAlloc::svp_ppol_alloc(cols)
SvpPPolBytesOf::bytes_of_svp_ppol(cols)
SvpPreparePPol::svp_prepare_ppol(res, res_col, a, a_col)
SvpPPolCopyBackend::svp_ppol_copy_backend(res, res_col, a, a_col)
```

`svp_prepare_tpol` and `svp_prepare_ppol` both consume `ScalarZnx`. The former builds
the hot-prep layout; the latter builds the packed/cold layout.

Changed existing API:

```rust
svp_prepare -> svp_prepare_ppol
```

Apply to DFT:

```rust
svp_apply_small_small_to_dft(res, res_col, scalar, scalar_col, a, a_col)
svp_apply_small_dft_to_dft(res, res_col, scalar, scalar_col, a, a_col)
svp_apply_tpol_small_to_dft(res, res_col, tpol, tpol_col, a, a_col)
svp_apply_tpol_dft_to_dft(res, res_col, tpol, tpol_col, a, a_col)
svp_apply_ppol_small_to_dft(res, res_col, ppol, ppol_col, a, a_col)
svp_apply_ppol_dft_to_dft(res, res_col, ppol, ppol_col, a, a_col)
```

Changed existing API:

```rust
svp_apply_dft        -> svp_apply_ppol_small_to_dft
svp_apply_dft_to_dft -> svp_apply_ppol_dft_to_dft
```

In-place DFT input/output:

```rust
svp_apply_small_dft_to_dft_assign(res, res_col, scalar, scalar_col)
svp_apply_tpol_dft_to_dft_assign(res, res_col, tpol, tpol_col)
svp_apply_ppol_dft_to_dft_assign(res, res_col, ppol, ppol_col)
```

Changed existing API:

```rust
svp_apply_dft_to_dft_assign -> svp_apply_ppol_dft_to_dft_assign
```

Derived outputs:

```rust
svp_apply_small_small_to_big(...)
svp_apply_small_dft_to_big(...)
svp_apply_tpol_small_to_big(...)
svp_apply_tpol_dft_to_big(...)
svp_apply_ppol_small_to_big(...)
svp_apply_ppol_dft_to_big(...)

svp_apply_small_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
svp_apply_small_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
svp_apply_tpol_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
svp_apply_tpol_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
svp_apply_ppol_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
svp_apply_ppol_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
```

## VMP

```text
small: MatZnx
tmat:  VmpTMat
pmat:  VmpPMat
```

`VmpTMat` is the hot-prep transformed matrix. `VmpPMat` keeps its current
meaning: vector-matrix-product prepared matrix, the packed/cold layout.

New hot-prep layout family:

```rust
VmpTMat
VmpTMatOwned<BE>
VmpTMatBackendRef<'a, BE>
VmpTMatBackendMut<'a, BE>
VmpTMatToBackendRef<BE>
VmpTMatToBackendMut<BE>
```

Existing cold-prep layout family:

```rust
VmpPMat
VmpPMatOwned<BE>
VmpPMatBackendRef<'a, BE>
VmpPMatBackendMut<'a, BE>
VmpPMatToBackendRef<BE>
VmpPMatToBackendMut<BE>
```

Preparation/allocation:

```rust
VmpTMatAlloc::vmp_tmat_alloc(rows, cols_in, cols_out, size)
VmpTMatBytesOf::bytes_of_vmp_tmat(rows, cols_in, cols_out, size)
VmpPrepareTMatTmpBytes::vmp_prepare_tmat_tmp_bytes(rows, cols_in, cols_out, size)
VmpPrepareTMat::vmp_prepare_tmat(tmat, mat, scratch)
VmpTMatZero::vmp_tmat_zero(res)

VmpPMatAlloc::vmp_pmat_alloc(rows, cols_in, cols_out, size)
VmpPMatBytesOf::bytes_of_vmp_pmat(rows, cols_in, cols_out, size)
VmpPreparePMatTmpBytes::vmp_prepare_pmat_tmp_bytes(rows, cols_in, cols_out, size)
VmpPreparePMat::vmp_prepare_pmat(pmat, mat, scratch)
VmpZero::vmp_zero(res)
```

Changed existing API:

```rust
vmp_prepare_tmp_bytes -> vmp_prepare_pmat_tmp_bytes
vmp_prepare           -> vmp_prepare_pmat
```

Apply to DFT:

```rust
vmp_apply_small_small_to_dft(res, a, mat, scratch)
vmp_apply_small_dft_to_dft(res, a, mat, limb_offset, scratch)
vmp_apply_tmat_small_to_dft(res, a, tmat, scratch)
vmp_apply_tmat_dft_to_dft(res, a, tmat, limb_offset, scratch)
vmp_apply_pmat_small_to_dft(res, a, pmat, scratch)
vmp_apply_pmat_dft_to_dft(res, a, pmat, limb_offset, scratch)
```

Changed existing API:

```rust
vmp_apply_dft        -> vmp_apply_pmat_small_to_dft
vmp_apply_dft_to_dft -> vmp_apply_pmat_dft_to_dft
```

Scratch sizing:

```rust
vmp_apply_small_small_to_dft_tmp_bytes(...)
vmp_apply_small_dft_to_dft_tmp_bytes(...)
vmp_apply_tmat_small_to_dft_tmp_bytes(...)
vmp_apply_tmat_dft_to_dft_tmp_bytes(...)
vmp_apply_pmat_small_to_dft_tmp_bytes(...)
vmp_apply_pmat_dft_to_dft_tmp_bytes(...)
```

Changed existing API:

```rust
vmp_apply_dft_tmp_bytes        -> vmp_apply_pmat_small_to_dft_tmp_bytes
vmp_apply_dft_to_dft_tmp_bytes -> vmp_apply_pmat_dft_to_dft_tmp_bytes
```

Accumulating DFT output:

```rust
vmp_apply_small_small_to_dft_accumulate(res, a, mat, limb_offset, scratch)
vmp_apply_small_dft_to_dft_accumulate(res, a, mat, limb_offset, scratch)
vmp_apply_tmat_small_to_dft_accumulate(res, a, tmat, limb_offset, scratch)
vmp_apply_tmat_dft_to_dft_accumulate(res, a, tmat, limb_offset, scratch)
vmp_apply_pmat_small_to_dft_accumulate(res, a, pmat, limb_offset, scratch)
vmp_apply_pmat_dft_to_dft_accumulate(res, a, pmat, limb_offset, scratch)

vmp_apply_small_small_to_dft_accumulate_tmp_bytes(...)
vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(...)
vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(...)
vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(...)
vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes(...)
vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(...)
```

Changed existing API:

```rust
vmp_apply_dft_to_dft_accumulate
    -> vmp_apply_pmat_dft_to_dft_accumulate
vmp_apply_dft_to_dft_accumulate_tmp_bytes
    -> vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes
```

Derived outputs:

```rust
vmp_apply_small_small_to_big(...)
vmp_apply_small_dft_to_big(...)
vmp_apply_tmat_small_to_big(...)
vmp_apply_tmat_dft_to_big(...)
vmp_apply_pmat_small_to_big(...)
vmp_apply_pmat_dft_to_big(...)

vmp_apply_small_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
vmp_apply_small_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
vmp_apply_tmat_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
vmp_apply_tmat_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
vmp_apply_pmat_small_to_small(..., res_base2k, res_offset, a_base2k, ...)
vmp_apply_pmat_dft_to_small(..., res_base2k, res_offset, a_base2k, ...)
```

## CNV

For convolution, both operands have independent domains.

```text
small: VecZnx
tvec:  CnvTVecL / CnvTVecR
pvec:  CnvPVecL / CnvPVecR
```

The current lazy convolution path is the hot-prep implementation in disguise.
This plan makes that layout explicit:

```text
cnv_prepare_left_tvec  ~= current cnv_prepare_left_lazy
cnv_prepare_right_tvec ~= current cnv_prepare_right_lazy
cnv_apply_tvec_tvec_to_dft ~= current cnv_apply_dft_lazy
```

New hot-prep layout families:

```rust
CnvTVecL
CnvTVecLOwned<BE>
CnvTVecLBackendRef<'a, BE>
CnvTVecLBackendMut<'a, BE>
CnvTVecLToBackendRef<BE>
CnvTVecLToBackendMut<BE>

CnvTVecR
CnvTVecROwned<BE>
CnvTVecRBackendRef<'a, BE>
CnvTVecRBackendMut<'a, BE>
CnvTVecRToBackendRef<BE>
CnvTVecRToBackendMut<BE>
```

Existing cold-prep layout families:

```rust
CnvPVecL
CnvPVecLOwned<BE>
CnvPVecLBackendRef<'a, BE>
CnvPVecLBackendMut<'a, BE>
CnvPVecLToBackendRef<BE>
CnvPVecLToBackendMut<BE>

CnvPVecR
CnvPVecROwned<BE>
CnvPVecRBackendRef<'a, BE>
CnvPVecRBackendMut<'a, BE>
CnvPVecRToBackendRef<BE>
CnvPVecRToBackendMut<BE>
```

Preparation/allocation:

```rust
cnv_tvec_left_alloc(cols, size)
cnv_tvec_right_alloc(cols, size)
bytes_of_cnv_tvec_left(cols, size)
bytes_of_cnv_tvec_right(cols, size)
cnv_prepare_left_tvec_tmp_bytes(res_size, a_size)
cnv_prepare_right_tvec_tmp_bytes(res_size, a_size)
cnv_prepare_left_tvec(res, a, mask, scratch)
cnv_prepare_right_tvec(res, a, mask, scratch)
cnv_prepare_self_tvec_tmp_bytes(res_size, a_size)
cnv_prepare_self_tvec(left, right, a, mask, scratch)

cnv_pvec_left_alloc(cols, size)
cnv_pvec_right_alloc(cols, size)
bytes_of_cnv_pvec_left(cols, size)
bytes_of_cnv_pvec_right(cols, size)
cnv_prepare_left_pvec_tmp_bytes(res_size, a_size)
cnv_prepare_right_pvec_tmp_bytes(res_size, a_size)
cnv_prepare_left_pvec(res, a, mask, scratch)
cnv_prepare_right_pvec(res, a, mask, scratch)
cnv_prepare_self_pvec_tmp_bytes(res_size, a_size)
cnv_prepare_self_pvec(left, right, a, mask, scratch)
```

Changed existing API:

```rust
cnv_prepare_left           -> cnv_prepare_left_pvec
cnv_prepare_right          -> cnv_prepare_right_pvec
cnv_prepare_self           -> cnv_prepare_self_pvec
cnv_prepare_left_lazy      -> cnv_prepare_left_tvec
cnv_prepare_right_lazy     -> cnv_prepare_right_tvec
cnv_apply_dft_lazy         -> cnv_apply_tvec_tvec_to_dft
```

Apply to DFT:

```rust
cnv_apply_small_small_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_small_tvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_tvec_small_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_tvec_tvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_tvec_pvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_small_pvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_pvec_small_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_pvec_tvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
cnv_apply_pvec_pvec_to_dft(cnv_offset, res, res_col, left, left_col, right, right_col, scratch)
```

Changed existing API:

```rust
cnv_apply_dft -> cnv_apply_pvec_pvec_to_dft
```

Scratch sizing follows the same names with `_tmp_bytes`.

Accumulating DFT output:

```rust
cnv_apply_small_small_to_dft_accumulate(...)
cnv_apply_small_tvec_to_dft_accumulate(...)
cnv_apply_tvec_small_to_dft_accumulate(...)
cnv_apply_tvec_tvec_to_dft_accumulate(...)
cnv_apply_tvec_pvec_to_dft_accumulate(...)
cnv_apply_small_pvec_to_dft_accumulate(...)
cnv_apply_pvec_small_to_dft_accumulate(...)
cnv_apply_pvec_tvec_to_dft_accumulate(...)
cnv_apply_pvec_pvec_to_dft_accumulate(...)

cnv_accumulate_small_small_to_dft(...)
cnv_accumulate_small_tvec_to_dft(...)
cnv_accumulate_tvec_small_to_dft(...)
cnv_accumulate_tvec_tvec_to_dft(...)
cnv_accumulate_tvec_pvec_to_dft(...)
cnv_accumulate_small_pvec_to_dft(...)
cnv_accumulate_pvec_small_to_dft(...)
cnv_accumulate_pvec_tvec_to_dft(...)
cnv_accumulate_pvec_pvec_to_dft(...)
```

Changed existing API:

```rust
cnv_apply_dft_accumulate -> cnv_apply_pvec_pvec_to_dft_accumulate
cnv_accumulate_dft       -> cnv_accumulate_pvec_pvec_to_dft
```

Pairwise apply:

```rust
cnv_pairwise_apply_small_small_to_dft(...)
cnv_pairwise_apply_small_tvec_to_dft(...)
cnv_pairwise_apply_tvec_small_to_dft(...)
cnv_pairwise_apply_tvec_tvec_to_dft(...)
cnv_pairwise_apply_tvec_pvec_to_dft(...)
cnv_pairwise_apply_small_pvec_to_dft(...)
cnv_pairwise_apply_pvec_small_to_dft(...)
cnv_pairwise_apply_pvec_tvec_to_dft(...)
cnv_pairwise_apply_pvec_pvec_to_dft(...)
```

Changed existing API:

```rust
cnv_pairwise_apply_dft -> cnv_pairwise_apply_pvec_pvec_to_dft
```

Derived outputs:

```rust
cnv_apply_*_to_big(...)
cnv_apply_*_to_small(..., res_base2k, res_offset, a_base2k, ...)
```

Unchanged:

```rust
cnv_by_const_apply
cnv_by_const_apply_tmp_bytes
```

`cnv_by_const_apply` does not consume `CnvPVecL`/`CnvPVecR`.

## OEP / Delegate Mirror

For every public trait method above, mirror the same method name in:

- `poulpy-hal/src/oep/hal_impl.rs`
- `poulpy-hal/src/delegates/*.rs`
- backend default traits in `poulpy-cpu-ref/src/hal_defaults/*.rs`
- backend implementations in CPU/AVX/AVX512/ARM crates

The old ambiguous names should either be removed in one breaking change or kept
temporarily as deprecated wrappers to the cold-prep equivalents.

## Migration Map

| Current API | New API |
| --- | --- |
| `SvpPPol` | unchanged; packed/cold-prep form |
| `svp_ppol_alloc` | unchanged |
| `bytes_of_svp_ppol` | unchanged |
| `svp_prepare` | `svp_prepare_ppol` |
| `svp_apply_dft` | `svp_apply_ppol_small_to_dft` |
| `svp_apply_dft_to_dft` | `svp_apply_ppol_dft_to_dft` |
| `svp_apply_dft_to_dft_assign` | `svp_apply_ppol_dft_to_dft_assign` |
| `VmpPMat` | unchanged; packed/cold-prep form |
| `vmp_pmat_alloc` | unchanged |
| `bytes_of_vmp_pmat` | unchanged |
| `vmp_prepare_tmp_bytes` | `vmp_prepare_pmat_tmp_bytes` |
| `vmp_prepare` | `vmp_prepare_pmat` |
| `vmp_apply_dft` | `vmp_apply_pmat_small_to_dft` |
| `vmp_apply_dft_to_dft` | `vmp_apply_pmat_dft_to_dft` |
| `vmp_apply_dft_to_dft_accumulate` | `vmp_apply_pmat_dft_to_dft_accumulate` |
| `CnvPVecL` | unchanged; packed/cold-prep left form |
| `CnvPVecR` | unchanged; packed/cold-prep right form |
| `cnv_prepare_left` | `cnv_prepare_left_pvec` |
| `cnv_prepare_right` | `cnv_prepare_right_pvec` |
| `cnv_prepare_self` | `cnv_prepare_self_pvec` |
| `cnv_prepare_left_lazy` | `cnv_prepare_left_tvec` |
| `cnv_prepare_right_lazy` | `cnv_prepare_right_tvec` |
| `cnv_apply_dft_lazy` | `cnv_apply_tvec_tvec_to_dft` |
| `cnv_apply_dft` | `cnv_apply_pvec_pvec_to_dft` |
| `cnv_apply_dft_accumulate` | `cnv_apply_pvec_pvec_to_dft_accumulate` |
| `cnv_accumulate_dft` | `cnv_accumulate_pvec_pvec_to_dft` |
| `cnv_pairwise_apply_dft` | `cnv_pairwise_apply_pvec_pvec_to_dft` |

## Open Questions

Questions 1 and 2 are settled by the SVP pass; see the status note under
[Layout Tiers / SVP](#svp).

1. ~~Should old names remain as deprecated wrappers for one release, or is this a
   hard breaking change?~~ Hard break.
2. ~~Should derived `*_to_big` and `*_to_small` be public HAL primitives, or
   default-provided convenience traits?~~ OEP primitives, with a shared
   flavor-agnostic default implementation.
3. For mixed convolution hot/cold pairs (`t_p`, `p_t`), should every backend
   implement direct kernels immediately, or may some backends reject/default
   these until a higher-level caller needs them?
