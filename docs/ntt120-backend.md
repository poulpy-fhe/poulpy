# NTT120 Backend — Design Notes

## Overview

The NTT120 backend uses four ~30-bit primes (Primes30, Q ≈ 2^120) and Chinese
Remainder Theorem (CRT) to perform exact polynomial multiplication without
floating-point error. It replaces the f64 FFT backend for use cases that require
exact arithmetic at larger moduli.

## Prime Set

`Primes30`: four primes, each ~30 bits, product Q ≈ 2^120.
- Accessed via the `PrimeSet` trait: `Primes30::Q: [u32; 4]`, `Primes30::CRT_CST: [u32; 4]`
- **`PrimeSet` must be in scope** to use associated constants on `Primes30`.
- Also available: `Primes29`, `Primes31` for other bit-width requirements.

## Memory Layouts

| Poulpy type | Bytes/coeff | Format |
|-|-|-|
| `VecZnx`    | 8 (i64)     | coefficient domain |
| `VecZnxBig` | 16 (i128)   | CRT-reconstructed exact value |
| `VecZnxDft` | 32 (Q120bScalar = 4×u64) | NTT domain, one u64 per prime |
| `SvpPPol`   | 32 (Q120cScalar = 8×u32) | NTT+Montgomery domain |
| `VmpPMat`   | 32 (Q120cScalar)          | same as SvpPPol |

## HAL Reference Files (`poulpy-hal/src/reference/ntt120/`)

### `primes.rs`
- `PrimeSet` trait: associated consts `Q`, `CRT_CST`, `OMEGA`, `LOG_Q`, `N_PRIMES`
- `Primes29`, `Primes30`, `Primes31` structs

### `types.rs`
- `Q120aScalar`, `Q120bScalar`, `Q120cScalar` newtype wrappers
- `Q120x2bScalar`, `Q120x2cScalar` paired variants
- `Q_SHIFTED` constant array

### `arithmetic.rs`
- `b_from_znx64_ref`: i64 poly → q120b (4 u64 CRT residues per coeff)
- `c_from_znx64_ref`: i64 poly → q120c (Montgomery form)
- `b_to_znx128_ref`: q120b → i128 (CRT reconstruction)
- `add_bbb_ref`: q120b + q120b → q120b
- `add_ccc_ref`: q120c + q120c → q120c
- `c_from_b_ref`: q120b → q120c

### `mat_vec.rs`
- `BaaMeta<P>`, `BbbMeta<P>`, `BbcMeta<P>`: precomputed lazy-reduction metadata
- `vec_mat1col_product_baa_ref`: q120a × q120a → q120a (1 column)
- `vec_mat1col_product_bbb_ref`: q120b × q120b → q120b (1 column)
- `vec_mat1col_product_bbc_ref`: q120c × q120b → q120b (1 column)
- `vec_mat1col_product_x2_bbc_ref`: 2-coefficient block version
- `vec_mat2cols_product_x2_bbc_ref`: 2-column × 2-coeff block

### `ntt.rs`
- `NttTable<P>`, `NttTableInv<P>`: precomputed twiddle tables
- `ntt_ref<P>`: in-place forward NTT (modifies slice)
- `intt_ref<P>`: in-place inverse NTT (modifies slice)

### `vec_znx_big.rs`
- `ntt120_vec_znx_big_add`, `_add_assign`, `_add_small`, `_add_small_assign`
- `ntt120_vec_znx_big_sub`, `_sub_assign`, `_sub_negate_assign`, `_sub_small_a`, `_sub_small_b`
- `ntt120_vec_znx_big_negate`, `_negate_assign`
- `ntt120_vec_znx_big_from_small`: sign-extend i64 VecZnx → i128 VecZnxBig
- `ntt120_vec_znx_big_normalize_tmp_bytes(n)`, `ntt120_vec_znx_big_normalize`: extract base-2k digits
- `ntt120_vec_znx_big_automorphism_assign_tmp_bytes(n)`, `ntt120_vec_znx_big_automorphism`, `_automorphism_assign`: apply X→X^p
- `ntt120_vec_znx_big_add_normal_ref`: add rounded Gaussian noise to limb

### `vec_znx_dft.rs`
- `NttModuleHandle` trait: `get_ntt_table`, `get_intt_table`, `get_bbc_meta`
- `NttHandleProvider` unsafe trait: implemented by backend handle types
- Blanket: `impl<B: Backend> NttModuleHandle for Module<B> where B::Handle: NttHandleProvider`
- `ntt120_vec_znx_dft_apply`: i64 VecZnx → q120b VecZnxDft (forward NTT)
- `ntt120_vec_znx_idft_apply_tmp_bytes(n) = 4*n*8`
- `ntt120_vec_znx_idft_apply`: q120b → i128 (non-destructive, uses scratch)
- `ntt120_vec_znx_idft_apply_tmpa`: q120b → i128 (uses input as scratch)
- `ntt120_vec_znx_dft_{add,sub,add_assign,sub_assign,sub_negate_assign,add_scaled_assign,copy,zero}`

### `svp.rs`
- `ntt120_svp_prepare`: i64 scalar poly → q120c SvpPPol
- `ntt120_svp_apply_dft_to_dft`: q120c × q120b → q120b (per-coeff, calls bbc)
- `ntt120_svp_apply_dft_to_dft_add`: accumulate variant
- `ntt120_svp_apply_dft_to_dft_assign`: in-place variant

### `vmp.rs`
- `ntt120_vmp_prepare_tmp_bytes(n) = 4*n*8`
- `ntt120_vmp_prepare`: i64 MatZnx rows → q120c VmpPMat (block-interleaved layout)
- `ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)`
- `ntt120_vmp_apply_dft_to_dft`: q120b vector × q120c matrix → q120b
- `ntt120_vmp_apply_dft_to_dft_add`: accumulate variant
- `ntt120_vmp_zero`: zero a VmpPMat

## Reference Backend (`poulpy-cpu-ref/src/ntt120/`)

### `module.rs`
- `NTT120RefHandle { table_ntt, table_intt, meta_bbc, meta_bbb }`
- `unsafe impl NttHandleProvider for NTT120RefHandle` (wires into blanket impl)
- `impl Backend for NTT120Ref` with `ScalarPrep = Q120bScalar`, `ScalarBig = i128`

### `vec_znx_dft.rs` — Key: `compact_all_blocks`
IDFT-consume must convert in-place from Q120b layout (32B/coeff) to i128 (16B/coeff).
Processing blocks in order k=0,1,...,n_blocks-1 is safe because:
- For k≥1: dst end = 16n(k+1) ≤ src start = 32nk → non-overlapping
- For k=0: all four u64 residues are read into locals before any i128 write

### `convolution.rs`
Runtime stub — all methods panic with `unimplemented!()`. Future work.

## AVX Backend (`poulpy-cpu-avx/src/ntt120/`)

All OEP traits are fully AVX2-accelerated. Requires `enable-avx` feature and
`RUSTFLAGS="-C target-feature=+avx2,+fma"`.

### `module.rs`
- `NTT120AvxHandle { table_ntt, table_intt, meta_bbc, meta_bbb }` — same fields as `NTT120RefHandle`
- `unsafe impl NttHandleProvider for NTT120AvxHandle`
- `NttHandleFactory` includes AVX2 runtime CPUID check

### `ntt_avx.rs`
- AVX2 NTT butterfly kernels; variable bit-shifts via `_mm256_srl_epi64(x, _mm_cvtsi64_si128(h))`
- Ported from spqlios-arithmetic (carries DISCLAIMER block)

### `mat_vec_avx.rs`
- `NttMulBbc`, `NttMulBbc1ColX2`, `NttMulBbc2ColsX2` traits
- BBC mat-vec via `_mm256_mul_epu32`; ~8 mul_epu32 reductions per output element

### `arithmetic_avx.rs`
- `b_from_znx64_avx2`: broadcast i64 + sign-check, ~5 instr/element
- `c_from_b_avx2`: Barrett reduction with `mu=floor(2^61/Q[k]) < 2^32`; conditional subtract pattern
- `vec_mat1col_product_bbb_avx2`: 4-bin (s1–s4) accumulation + BbbMeta collapse
- `b_to_znx128_avx2`: hybrid — AVX2 for `xk%Q` and `(xk*CRT)%Q`, scalar i128 for final CRT accumulation

### `vec_znx_big_avx.rs`
- AVX2-accelerated `VecZnxBig` operations (add, sub, negate, normalize, automorphism)

### `vec_znx_dft.rs`, `svp.rs`, `vmp.rs`
- All delegate to AVX2 NTT kernels and mat-vec traits

### `convolution.rs`
Runtime stub — all methods panic with `unimplemented!()`. Future work.

## Test Suite Available

`poulpy_hal::test_suite` provides cross-backend correctness tests.
All helpers compare `Module<FFT64Ref>` (reference) vs `Module<NTT120Ref>` (test) or
`Module<NTT120AvxHandle>` (AVX test).
Signature: `test_foo(base2k: usize, module_ref: &Module<FFT64Ref>, module_test: &Module<NTT120Ref>)`

Available:
- `vec_znx`: add, sub, shift, negate, rotate, automorphism, normalize, copy (~24 tests)
- `vec_znx_big`: add, sub, negate variants; automorphism, normalize (~16 tests)
- `vec_znx_dft`: add, add_assign, sub, sub_assign, sub_negate_assign, copy,
  idft_apply, idft_apply_tmpa, idft_apply_consume
- `svp`: apply_dft, apply_dft_to_dft, apply_dft_to_dft_add, apply_dft_to_dft_assign
- `vmp`: apply_dft, apply_dft_to_dft, apply_dft_to_dft_add
- `sampling`: fill_uniform, fill_normal, add_normal
- `convolution`: test_convolution, test_convolution_by_const, test_convolution_pairwise
- Boundary suites: `ntt_n1024`, `ntt_n8192` (n=1024/8192, base2k boundary testing)

See `poulpy-cpu-ref/src/tests.rs` and `poulpy-cpu-avx/src/ntt120/tests.rs` for usage examples.
