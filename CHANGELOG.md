# CHANGELOG

## [Unreleased]

### `poulpy-ckks`

- Add `ckks_mod_raise[_tmp_bytes]`, `ckks_bootstrap_eval_mod[_tmp_bytes]` and `ckks_eval_lut[_tmp_bytes]`: each step of the bootstrap runs alone, the plan deciding how. Each consumes its input, which a refusal leaves unchanged.
- **Breaking:** `ckks_coeffs_to_slots_assign`, `ckks_slots_to_coeffs_assign`, `ckks_dft_matrix_eval_assign` and `ckks_dft_tmp_bytes` take `dft: &D`, `D: CKKSDFTFactors<BE, Dir, Fmt>`, instead of a `DFTMatrix`, the two transforms still at `Standard`, generics `<D, Dst, H>`. Every DFT evaluation takes `H: GetAutomorphismKey`, without `GetTensorKey`, so a factor of ciphertext diagonals refuses with `MissingRelinearizationKey`, and `BootstrappingLinearKeys::RotationKeys` loses that bound too. An out-of-tree `CKKSDFTImpl` re-signatures the required `ckks_dft_tmp_bytes` and `ckks_dft_matrix_eval_assign`, generics `<Dir, Fmt, D, ..>`, and any override of the two transforms, and reads the factors through `CKKSDFTFactors::factors` and their schedules through `CKKSDFTFactors::factor_plans`. `BootstrappingContextPrepared` and `BootstrappingRun` read as `Standard`; a plain `ctx.slots_to_coeffs()` goes through `ckks_dft_matrix_eval_assign`.
- Add `BootstrappingRun` and `BootstrappingContextPrepared::run`: a compiled bootstrap bound to the labels of its message, so a composition stamps nothing. `ckks_bootstrap_eval_mod`, `ckks_eval_lut` and `ckks_slots_to_coeffs_assign` label their output from it, under a bypass the composer calling `stamp_eval_mod_output` after adding it back. `BootstrappingContextPrepared` implements `CKKSModRaiseInfos` and the encoding `CKKSDFTFactors`, `BootstrappingRun` every infos trait of the steps. `stamp_eval_mod_output` refuses `res` with fewer scale bits than the guard bits.
- Add `ModRaisePlan`, `ModRaisePrepared`, `CKKSModRaiseInfos`, `CKKSEvalModInfos`, `BootstrappingPlan::mod_raise` and `BootstrappingContextPrepared::mod_raise`. `ckks_mod_raise` takes a `ModRaisePlan` on a plan that leaves no CoeffsToSlots factor to the raise, and refuses it on one that does.
- Add `BootstrappingContextPrepared::coeffs_mask[_tmp_bytes]`, `CoeffsMaskPrepared`, `BootstrappingContextPrepared::masked` and `MaskedFactors`: the last `CoeffsToSlots` factor times a selection of the input coefficients, which zeroes the others at no level, as a view of the context's factors for `ckks_coeffs_to_slots_assign`.
- Add `DFTMatrix::head` and `HeadFactors`: a view of every factor of a matrix but the last, for a caller that runs that factor in another form.
- Add `BootstrappingContextPrepared::merged_s2c_factor[_tmp_bytes]`, `merged_s2c_cosets` and `MergedS2CFactorPrepared`: the last SlotsToCoeffs factor with one kernel polynomial per coset of the coefficients, which multiplies each coset by its kernel in the level of that factor, with the factor's rotation keys.
- Add `BootstrappingPlan::with_log_gap` and `BootstrappingPreset::with_log_gap`: an S2C-first bootstrap of plaintexts in `Z[X^g]`, with transforms on `N/2g` slots and a sparse secret in that subring, whose raise projects the ciphertext on the subring in place of a trace. A key under that secret is bounded at the subring's degree, so a preset refuses a gap without a bound; `generate_keys` refuses transforms that do not fill those slots, and `compile_unprepared` only transforms past them at the degree of its module, which may exceed `N`. `coeffs_mask` reads either half of the coefficients, one per real slot, on an identity S2C-first context without EvalRound+, and `merged_s2c_factor` takes kernels of the subring.
- Add `BootstrappingPreset::with_output_k`, `with_slots_to_coeffs`, `with_input_headroom`, `with_eval_mod_polynomial`, `scales` and `with_scales` (`PresetScales`), to fit a preset and its keys to the circuit around the bootstrap.
- Add the suite functions `test_suite::presets::test_ckks_coeffs_mask`, `test_ckks_coeffs_mask_rejects_invalid_inputs`, `test_ckks_coeffs_mask_subring` and `test_merged_s2c_factor`, with their `StepsRunModule` bound, registered per type at radix 52; `test_bootstrapping_parity` also compares the masked `CoeffsToSlots`, the raise of a gap and the merged factor, and takes `F: DftScalar + Sync`.
- `BootstrappingContext::prepare` prepares the diagonals of a first `CoeffsToSlots` factor on the raise, so no bootstrap transforms them again. The output bits are unchanged.
- **Breaking:** `ckks_bootstrap` and `ckks_functional_bootstrap` are `oep::derived` defaults of `CKKSBootstrappingImpl` (`ckks_bootstrap_reference`, `ckks_functional_bootstrap_reference` and their `_tmp_bytes_reference` are replaced by those bodies composing the steps' references), whose supertraits are exactly `CKKSCITwin`, `CKKSDFTImpl`, `CKKSCopyImpl`, `CKKSAddImpl` and `CKKSPow2Impl`; it requires `ckks_mod_raise`, `ckks_bootstrap_eval_mod`, `ckks_eval_lut` and their queries, and drops `ckks_mod_up_into`, `ckks_bootstrap_mod_up` and their queries. `impl_ckks_bootstrapping_reference_full!` takes `($be, $ci)`, and `CKKSCITwin::CI` names the conjugate-invariant twin of a backend.
- **Breaking:** `CKKSBootstrappingCIOps` (`ckks_bootstrap_ci_into` and its query become `ckks_bootstrap` and its query with `Some(ci)`), `CKKSBootstrappingCIImpl`, `CKKSBootstrappingCIReference`, `impl_ckks_bootstrapping_ci_reference_full!`, `BootstrappingCIKeys`, `compile_ci*`, `generate_keys_ci` and `prepare_ci` are removed. A CI plan runs through `ckks_bootstrap`, `compile`, `generate_keys` and `prepare` with `Some(&ci)`, a standard plan with `None`; `BootstrappingKeysPrepared` takes no third parameter, and one arena serves both rings.
- **Breaking:** `BootstrappingKeys::tensor_key()` returns an `Option`, and the trait gains `ci_tensor_key` and `conversion_keys`. `BootstrappingLinearKeys::keys_layout` returns `Option<&BootstrappingKeysLayout>`, read once at preparation, so a key replaced in a `BootstrappingKeysPrepared` later does not update it, and `BootstrappingKeysLayout::rotation_key_layouts` returns an iterator. `BootstrappingKeySet::prepare` and `transfer_and_prepare` return a `Result`, `prepare_tmp_bytes` takes the CI module, and `generate_tmp_bytes` is new.
- **Breaking:** `ckks_mod_up_into` and `ckks_bootstrap_mod_up` are replaced by `ckks_mod_raise`, whose input is clobbered.
- **Breaking:** `CKKSModUpFirstFactorImpl::ckks_mod_up_first_factor[_tmp_bytes]` take `log_gap`, which `impl_ckks_mod_up_first_factor_reference_full!` forwards to the reference; `ModRaisePlan` gains the field `log_gap`, which `CKKSModRaiseInfos::log_gap` reads; `generate_keys` requires `VecZnxSwitchRing`.
- **Breaking:** the suite trait `TestContextSharedModule` requires `VecZnxSwitchRing`, as `generate_keys` does.
- **Breaking, behaviour:** a sparse plan needs its gap. `generate_keys` refuses transforms that do not fill `N/2g` slots, and `ckks_bootstrap` and `ckks_functional_bootstrap` refuse an input of another degree than those transforms fill, where a plan on fewer slots than the ring without a gap returned noise: no projection and no trace kept the overflow of the raise in the subring.
- **Breaking:** test suites: `BootstrappingPresetRun::setup` takes `(preset, ci)`, `Some(ci)` replacing `setup_ci(preset)`, and `None` a standard plan. `PresetRunModule` and `CIBackend` take `BE: CKKSCITwin`. `small_s2c_preset_run_with_client`, `test_ckks_bootstrap_eval_mod` and `parity::run_bootstrap` are new.
- **Behaviour:** `ckks_bootstrap` refuses at entry, before any write, ciphertexts of two degrees or radices, or of a degree the module does not serve, and a C2S-first width without the budget of SlotsToCoeffs. `ckks_functional_bootstrap` checks the key widths in full at entry instead of mid-run, and its refusal reasons change: the identity-context reason no longer names the standard ring, the encapsulation reason loses its "(expected, got)" suffix, and missing conversion keys read "the keys hold no conversion keys of the plan's ring".
- **Breaking, behaviour:** EvalRound+ on the standard ring adds the doubled bypass to `EvalMod(x) - K·x` after recombining, as on the conjugate-invariant ring, with one conjugation less. An S2C-first standard EvalRound+ bootstrap returns the same value in other lazy digits; a normalized output is equal. No other path changes bits.

## [0.10.0] - 2026-11-01

Completes the workspace naming pass ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): names lead with their object, opt-in macros follow their OEP trait (`impl_<op>_reference_full!`), every API method dispatches through a same-named OEP method, and operation-wide scalars lead argument lists. Gadget keys take digits of any width in bits ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)) and a row offset, and every gadget product refuses an input wider than its key ([#451](https://github.com/poulpy-fhe/poulpy/issues/451)). CKKS bootstrapping applies the first CoeffsToSlots factor on the modulus raise, runs EvalMod on the conjugate-invariant ring and takes power-of-two rotation keys, with new `LogN=16` presets; normalization rounds ties to even. Crates move into `schemes/`, `libs/`, `backends/` and `tools/`; new crates: `poulpy-blas` (encrypted matrix arithmetic), `poulpy-libm` (CKKS math functions, moved in from its own repository) and `poulpy-gpu-portable` (NTT4x30 on CUDA, HIP and WGPU through CubeCL).

### Workspace

- The basis crates `poulpy-hal`, `poulpy-core` and `poulpy-mhe` stay at the root, and the others move into `schemes/`, `libs/`, `backends/` and `tools/` with unchanged names and features; a path dependency on a local clone takes the new folder.
- The crates inherit `version`, `edition`, `license`, `repository` and `homepage` from `[workspace.package]`.
- The guides move to `docs/guides/` and the logo to `docs/img/`.

### `poulpy-hal`

- **Breaking:** hal names follow the workspace naming grammar, without deprecated aliases ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `bytes_of_<layout>` becomes `<layout>_bytes_of` (on `Backend`, the `<Layout>BytesOf` traits and `Module`); `Convolution` and `HalConvolutionImpl` become `CnvOps` and `HalCnvImpl`; `VecZnxScalarProduct::vec_znx_scalar_product` becomes `VecZnxBigScalarProduct::vec_znx_big_scalar_product` (test `test_vec_znx_big_scalar_product`).
- **Breaking:** preparations name their layout first: `SvpPrepare::svp_prepare` becomes `SvpPPolPrepare::svp_ppol_prepare`, `VmpPrepare::vmp_prepare` and `VmpPrepareTmpBytes` become `VmpPMatPrepare::vmp_pmat_prepare` and `VmpPMatPrepareTmpBytes`, `cnv_prepare_{left, right, self}[_tmp_bytes]` become `cnv_pvec_{left, right, self}_prepare[_tmp_bytes]`.
- **Breaking:** `NegacyclicFFT` and `NegacyclicFFTNew` become `ReimM::m`, `ReimFFT::reim_fft`, `ReimIFFT::reim_ifft` and `ReimNew::new` (a bound on both is `ReimNew<F>`), and `test_negacyclic_fft[_bit_exact]` becomes `test_reim_fft[_bit_exact]`; `ScratchArenaTakeBasic` becomes one trait per layout, `ScratchArenaTake<Layout>`, and `take_vec_znx_slice_scratch` takes `(n, len, cols, size)`.
- **Breaking:** `ScalarZnxAutomorphism` and `ScalarZnxCIEmbed` move to `api::scalar_znx`, and their OEP methods, now with default bodies, to the new `HalScalarZnxImpl: HalVecZnxImpl` (an empty `unsafe impl` suffices); `HalVecZnxImpl::vec_znx_fill_uniform(.., seed)` becomes `vec_znx_fill_uniform_source(.., source)`.
- **Breaking:** `vec_znx_idft_normalize_consume[_tmp_bytes]`, `VecZnxIdftNormalizeConsume[TmpBytes]` and `test_vec_znx_idft_normalize_consume[_derived]` become `vec_znx_idft_normalize[_tmp_bytes]`, `VecZnxIdftNormalize[TmpBytes]` and `test_vec_znx_idft_normalize[_derived]` ([#430](https://github.com/poulpy-fhe/poulpy/issues/430)).
- **Breaking:** `VecZnxIdftApplyTmpA`, `vec_znx_idft_apply_tmpa` and `test_vec_znx_idft_apply_tmpa` are removed for `vec_znx_idft_apply` with scratch ([#339](https://github.com/poulpy-fhe/poulpy/issues/339)); `VecZnx::{encode, decode}_vec_{i64, i128}_strided` are removed.
- **Breaking:** `cnv_apply_dft` and `cnv_apply_dft_add` take `b_galois` after `b_col` (`1` reads the right operand as before, an even one panics), and `CnvDftAccTerm` has a `b_galois` field, which overrides of `cnv_apply_dft_sum` must read.
- **Breaking:** `HalVecZnxImpl` requires `vec_znx_gadget_decompose[_tmp_bytes]` (new `VecZnxGadgetDecompose[TmpBytes]`, digits of any width `k_digit`) ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
- **Breaking:** every backend implements the new `HalMatZnxImpl: HalVecZnxImpl` (an empty `unsafe impl` suffices) behind `api::mat_znx`: `MatZnxAddAssign` ([#380](https://github.com/poulpy-fhe/poulpy/issues/380)), `MatZnxNormalize[Assign, TmpBytes]` and `MatZnxCopy`.
- **Breaking:** `max_base2k` returns `usize` instead of `Option<usize>`, and a backend without a model, such as `HostBytesBackend`, does not implement `MaxBase2k`; `HalVecZnxMonomialImpl` requires `Backend<Ring = Standard>`.
- **Breaking:** `ScalarZnx` implements `Debug` on host-readable storage only, printing its shape and first 16 coefficients per column, `VecZnxDft`, `VecZnxBig` and the prepared layouts implement a shape-only `Debug`, and `CnvPVecL` and `CnvPVecR` implement `PartialEq` and `Eq` ([#450](https://github.com/poulpy-fhe/poulpy/issues/450)).
- **Breaking:** test suites: `test_vec_znx_dft_step_zero_rejected`, `test_convolution_prepare_shape_rejected`, `test_convolution_by_const_degree_rejected` and `test_vmp_apply_dft_to_dft_shape_rejected` become `test_vec_znx_dft_rejects_zero_step`, `test_cnv_prepare_rejects_mismatched_shape`, `test_cnv_by_const_rejects_degree_mismatch` and `test_vmp_apply_dft_to_dft_rejects_column_mismatch`; scratch arenas are poisoned and exactly the operation's `_tmp_bytes`, and `test_convolution[_by_const, _pairwise]`, `test_vmp_apply_dft` and `test_vmp_zero` require the `_tmp_bytes` traits of their setup calls.
- **Breaking, behaviour:** normalization rounds ties to even instead of toward positive infinity.
- The contract key and `test_suite` module `sparse` are `embed`: `test_vec_znx_sparse_add_sub`, `test_vec_znx_big_sparse_add_sub` and `test_convolution_sparse` become `test_vec_znx_embed_add_sub`, `test_vec_znx_big_embed_add_sub` and `test_convolution_embed`.
- Add `max_base2k_squares` (`MaxBase2k`, `Module`); `max_base2k_ntt` and `max_base2k_fft64` model the conjugate-invariant ring ([#435](https://github.com/poulpy-fhe/poulpy/issues/435)).
- Add `AlignedBuf::region_mut`: host scratch regions carved from one arena no longer invalidate each other under Stacked Borrows.
- **Breaking:** an embedded right operand of the convolution applies is a power of two from `Backend::MIN_DEGREE` to `N`: the applies refuse a smaller one, where the contract left the floor to the backend, and out-of-tree overrides must refuse it too (`test_convolution_embed` runs only that domain). `cnv_apply_dft_sum` and `cnv_pairwise_apply_dft` check every term before their first write.

### CPU backends

- **Breaking:** hal renames ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `hal_impl_convolution!` becomes `hal_impl_cnv!`, and the `hal_defaults` methods `vec_znx_scalar_product_default`, `svp_prepare_default`, `vmp_prepare[_tmp_bytes]_default`, `cnv_prepare_{left, right, self}[_tmp_bytes]_default` and `vec_znx_fill_uniform_default` become `vec_znx_big_scalar_product_default`, `svp_ppol_prepare_default`, `vmp_pmat_prepare[_tmp_bytes]_default`, `cnv_pvec_{left, right, self}_prepare[_tmp_bytes]_default` and `vec_znx_fill_uniform_source_default`, and the `kernels` preparations `svp_prepare_portable`, `vmp_prepare[_tmp_bytes]_portable`, `ntt4x30_svp_prepare_portable`, `ntt4x30_vmp_prepare[_tmp_bytes]_portable` and `ntt4x30_cnv_prepare_{left, right, self}[_tmp_bytes]_portable` become `svp_ppol_prepare_portable`, `vmp_pmat_prepare[_tmp_bytes]_portable`, `ntt4x30_svp_ppol_prepare_portable`, `ntt4x30_vmp_pmat_prepare[_tmp_bytes]_portable` and `ntt4x30_cnv_pvec_{left, right, self}_prepare[_tmp_bytes]_portable`. The encoding tables implement `ReimM`, `ReimFFT` and `ReimIFFT` in place of `NegacyclicFFT`, and every backend implements `HalScalarZnxImpl`.
- **Breaking:** backends register through the renamed scheme opt-ins (`impl_<family>_reference_full!`): an out-of-tree backend invoking one directly renames the invocation.
- **Breaking:** `impl_cpu_core_defaults!` registers the core trace and `impl_cpu_ckks_defaults!` the EvalMod reference on both rings: an out-of-tree backend calling them drops its own `impl_glwe_trace_derived_full!` and `impl_ckks_eval_mod_reference_full!`, which now conflict. A standard one calling `impl_cpu_ckks_defaults!` adds `impl_ckks_ci_ring_map_reference_full!`, `impl_ckks_paco_reference_full!` and `impl_ckks_ship_reference_full!`.
- **Breaking:** CPU backends normalize through one kernel, the trait `ZnxNormalize`, with no scratch: `vec_znx_normalize_tmp_bytes` and `vec_znx_big_normalize_tmp_bytes` return 0. `vec_znx_big_normalize_portable` moves from `kernels::fft64::vec_znx_big` to `kernels::vec_znx_big_normalize`, `poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par` becomes `vec_znx_big_normalize_par`, the public normalize entries lose `carry`, and the suites `test_suite::normalization::test_normalization_kernels` and `test_suite::normalization_i128::test_i128_normalize_fused` become `test_suite::znx_normalize::test_znx_normalize`.
- **Breaking:** removed: `ZnxNormalize*Step*`, `ZnxNormalizeDigit`, `ZnxExtractDigitAddMul`, `I64NormalizeOps`, `I128NormalizeOps` and their kernels, `I128NormalizePlan`, `kernels::normalization`, `AddOp`, `SubOp` and `AssignOp` of `kernels::ntt4x30::vec_znx_big`, `ntt4x30_vec_znx_big_normalize*_portable`, `vec_znx_big_normalize_tmp_bytes_portable`, `vec_znx_lsh_assign_carry_bytes_portable`, `rayon_forward_znx_const!`, `vec_znx_idft_apply_tmpa_portable` and `ntt4x30_vec_znx_idft_apply_tmpa_portable`.
- **Breaking:** `SparseWord` becomes `MixedWord`, `sparse_log_gap_portable` becomes `embed_log_gap_portable`, `reim4_gather_sparse_block[_sum]_portable` becomes `reim4_gather_block[_sum]_portable`, and the test `conjugate_invariant_sparse_convolution` becomes `conjugate_invariant_embed_convolution`.
- **Breaking:** the `Reim4Convolution` and `Fft64RingArith` convolution methods take a `RightRead` (`right_read_fft64`, `right_read_ntt`) in place of `b_log_gap`; its constructors and `embed_log_gap_portable` take the backend's `MIN_DEGREE` last. `convolution_apply_dft_portable`, `ntt4x30_cnv_apply_dft_portable` and their `_add` forms take `b_galois` after `b_col`.
- **Breaking:** `paco_coeff_encodings_staged` and `ship_coeff_encodings_staged` take `(module, base2k, ..)` first and `scratch`, sized by `paco_coeff_encodings_staged_tmp_bytes` and `ship_coeff_encodings_staged_tmp_bytes`.
- **Breaking:** `NTT4x30AvxRayon` and `NTT4x30Avx512Rayon` no longer implement the q120 kernel traits (`NttFromZnx64`, `NttToZnx128`, `NttAdd`, `NttMulBbc` and the like).
- **Breaking, behaviour:** `NTT4x30Portable` stores the transform domain as four `u32` residues per coefficient (`DftWord = CrtWord<Primes30, u32>`) in the NEON layouts, half the previous size for `VecZnxDft`, `SvpPPol`, `VmpPMat` and both convolution operands, and is byte-compatible again with the NEON NTT4x30 backends (`VecZnxDftLayoutCompatible`, `SvpPPolLayoutCompatible`).
- **Breaking, behaviour:** `NTT4x30Avx`, `NTT4x30Avx512`, `NTT3x42Ifma` and their Rayon variants run the reference CKKS encapsulated ModUp (`impl_ckks_encapsulated_mod_up_reference_full!`) in place of their override.
- **Behaviour:** the `FFT64Avx512` convolution applies refuse a right operand below degree 16, where they accepted 8.
- `poulpy-cpu-rayon` adds `FFT64PortableRayon`, `NTT4x30PortableRayon` and their conjugate-invariant aliases `FFT64CIPortableRayon` and `NTT4x30CIPortableRayon`, and `PackedNtt4x30Base`, `impl_packed_ntt4x30_base!` and `impl_ntt4x30_rayon_backend!`, which now build `NTT4x30NeonRayon`, `NTT4x30AvxRayon` and `NTT4x30Avx512Rayon`.
- Add `enable-blas` (implies `enable-ckks`) to every CPU backend crate, selecting the `poulpy-blas` reference on both rings, and the `blas_pipeline` example to `poulpy-cpu-portable`; `poulpy-cpu-oracle` gains `enable-mhe`.
- Every CPU backend implements `DecompressionImpl`, `PreparationImpl`, `SecretImpl` and `vec_znx_gadget_decompose`, the last with one scratch-free portable kernel ([#460](https://github.com/poulpy-fhe/poulpy/issues/460)), parallel on Rayon; `hal_impl_vec_znx_without_normalize!` expands to `hal_impl_vec_znx_gadget_decompose!` and `hal_impl_vec_znx_without_normalize_or_decompose!`, the latter for a backend with its own layout.
- Normalization runs on `std::simd` lanes (widths from the associated types of `ZnxNormalize`); `poulpy-cpu-portable` enables the nightly features `portable_simd`, `hint_prefetch` and `associated_type_defaults`.
- `NTT4x30Portable` runs an allocation-free 32-bit NTT and a fused `GGLWEProductDigitsStridedImpl`. The AVX2, AVX-512 and IFMA `svp_apply_dft` take their buffer from scratch (`svp_apply_dft_tmp_bytes`) instead of allocating, and the FFT64 `vmp_apply_dft_to_dft_add` needs only the scratch of `vmp_apply_dft_to_dft`.
- The AVX-512 rank-one relinearization and the fused IFMA giant step refuse, as the reference does, a key that does not cover their input or has a row offset; the IFMA one leaves power-of-two rotation schedules to the reference.
- Conjugate-invariant backends have a `max_base2k` model, where they returned `None`, and `CKKSEncodingImpl<f32>` and `CKKSDFTMatrixImpl<f32>` serve every CPU type.
- `poulpy-cpu-portable` exports `kernels::noise::gaussian_cumulative_table_portable` and `gaussian_rejection_constants_portable`, and `hal_defaults::take_host_typed` is public.
- `poulpy-cpu-arm` adds the full-size bootstrapping preset precision tests of `NTT4x30Neon`, `NTT4x30CINeon` and `NTT4x30NeonRayon`.
- **Fix:** the portable `znx_add`, `znx_sub`, `znx_negate` and their in-place forms, the power-of-two multiplications and the rotating automorphism wrap on overflow, where they panicked under overflow checks.
- **Fix:** the AVX2, AVX-512 and NEON NTT convolution applies and the FFT64 DFT kernels did not check columns, and wrote past the buffer or read another limb for an out-of-range one; they now panic ([#422](https://github.com/poulpy-fhe/poulpy/issues/422)).
- **Fix:** the NTT4x30 AVX2, AVX-512 and NEON convolution prepares gave each parallel task a mutable view of the whole column; each now writes only its limb's rows.
- **Fix:** the SIMD CRT reconstruction of the IFMA inverse transform gave a wrong coefficient for rare residues.

### `poulpy-core`

- **Breaking:** names follow the workspace grammar, object first ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `decompress_<x>` becomes `<x>_decompress` (`decompress_automorphism_key` and `decompress_tensor_key` become `glwe_automorphism_key_decompress` and `glwe_tensor_key_decompress`); `GLWEMaskFill::fill_glwe_mask_from_{source,seed}` and `fill_glwe_from_source` become `GLWEFill::glwe_fill_mask_from_{source,seed}` and `glwe_fill_from_source`, and `fill_lwe_mask_from_*` become `lwe_fill_mask_from_*`; the in-place `GLWEShift::glwe_rsh` becomes `glwe_rsh_assign`, `ggsw_expand_row` and `ggsw_expand_rows_tmp_bytes` become `ggsw_expand_rows_assign[_tmp_bytes]`, `glwe_keyswitch_internal_tmp_bytes_from_sizes` becomes `glwe_keyswitch_internal_from_sizes_tmp_bytes`.
- **Breaking:** `GLWELinearTransformations` becomes `GLWELinearTransformationOps`, `glwe_eval_linear_transformation_*` become `glwe_linear_transformation_eval_*`, `glwe_prepare_linear_transformation_{rhs,baby_steps}[_tmp_bytes]` become `glwe_linear_transformation_{rhs,baby_steps}_prepare[_tmp_bytes]`, and `LinearTransformationImpl` becomes `GLWELinearTransformationImpl` (`impl_glwe_linear_transformation_reference_full!`).
- **Breaking:** `GLWEExternalProductInternal` and `glwe_external_product_internal_tmp_bytes` become `GLWEExternalProductDft` and `glwe_external_product_dft_tmp_bytes`; `GLWEPacking` becomes `GLWEPack`, `BSGSOps` `GLWEBSGSOps`, `DiagonalProd` `GLWEDiagonalProd`, `GGLWEAutomorphismReference` `GLWEAutomorphismKeyAutomorphismReference`, `PolynomialEvaluationImpl` `GLWEPolynomialEvaluationImpl` (`impl_glwe_polynomial_evaluation_derived_full!`); `GLWEToLWESwitchingKeyEncryptSk`, `LWEToGLWESwitchingKeyEncryptSk`, `GLWEToLWESwitchingKeyDecompress` and `LWESwitchingKeyEncrypt` (and their `*Reference`) become `GLWEToLWEKeyEncryptSk`, `LWEToGLWEKeyEncryptSk`, `GLWEToLWEKeyDecompress` and `LWESwitchingKeyEncryptSk`; `GLWEToLWESwitchingKeyCompressed` becomes `GLWEToLWEKeyCompressed` (deprecated alias).
- **Breaking:** `bytes_of_glwe_key_prepared` becomes `glwe_switching_key_prepared_bytes_of`, `bytes_of_gglwe_to_ggsw[_from_infos]` `gglwe_to_ggsw_key_prepared_bytes_of[_from_infos]`, `alloc_tensor_key_prepared[_from_infos]` and `bytes_of_tensor_key_prepared[_from_infos]` `glwe_tensor_key_prepared_alloc[_from_infos]` and `glwe_tensor_key_prepared_bytes_of[_from_infos]`, `prepare_tensor_key[_tmp_bytes]` `glwe_tensor_key_prepare[_tmp_bytes]`, `ggsw_zero` `ggsw_prepared_zero`, `glwe_prepare_right` `glwe_right_operand_prepare`.
- **Breaking:** one lifecycle trait per object: the allocators of `ModuleCoreAlloc` and `ModuleCoreCompressedAlloc` (which keep only `OwnedBuf` and `ZnxWord`) move to `GLWEAlloc`, `GGSWAlloc`, ..., `GLWECompressedAlloc`, ..., `GGSWCompressedBytesOf`; the non-GLWE byte counts of `GLWEBytesOf` move to `GLWEPlaintextBytesOf`, `GLWESecretBytesOf`, `GLWESecretTensorBytesOf`, `GLWETensorBytesOf`, `LWEBytesOf`, `GGLWEBytesOf`, `GGSWBytesOf` and `GLWELinearTransformationBabyStepsBytesOf`; `ScratchArenaTakeCore` splits into one `ScratchArenaTake<Object>` per object.
- **Breaking:** `SecretConversion` splits into `GLWESecretFromLWESecret` and `LWESecretFromGLWESecret`, whose `lwe_secret_from_glwe_secret` takes a trailing `scratch` (`lwe_secret_from_glwe_secret_tmp_bytes`); `GLWESecretTensorFactory::glwe_secret_tensor_prepare[_tmp_bytes]` becomes `GLWESecretTensorFromGLWESecret::glwe_secret_tensor_from_glwe_secret[_tmp_bytes]`, and `glwe_secret_tensor_prepared_prepare` becomes `glwe_secret_tensor_prepare`.
- **Breaking:** each OEP method's reference is a `*Reference` trait method (new `GLWECIConversionReference`, `GGLWEProductDigitsStridedReference`, `GLWELinearTransformationReference`), which forwarding overrides call; `glwe_tensor_apply_prepared_right[_tmp_bytes]` become `glwe_tensor_apply_prepared_right[_tmp_bytes]_reference`. `impl_glwe_tensoring_reference!`, `impl_gglwe_product_digits_strided_reference!` and `impl_glwe_packing_derived_full!` become `impl_glwe_tensoring_reference_full!`, `impl_gglwe_product_digits_strided_reference_full!` and `impl_glwe_pack_derived_full!`; the name `impl_gglwe_product_digits_strided_reference!` is reused for the step-hook forwarding below, called as `()` inside an `impl`, so an old `impl_gglwe_product_digits_strided_reference!(Backend)` fails with "no rules expected".
- **Breaking:** helpers leave `reference` (still at the crate root): `GLWEDiagonalProd` moves to `layouts`, `GLWEBSGSOps` to `api`, and `gglwe_product_output_size`, `gglwe_product_accumulation_output_size`, `prod_dft_size`, `cnv_offset_to_limb_offset` and `normalize_input_limb_bound_with_offset` to the new `poulpy_core::gadget`; `poulpy_core::oep` no longer re-exports `GLWEAutomorphismReference`, `GLWEAutomorphismKeyAutomorphismReference`, `ConversionReference`, `GGSWConversionReference`, `DecryptionReference`, `GLWEExternalProductReference`, `GLWEKeyswitchReference` and `LWEKeyswitchReference`, imported from `poulpy_core::reference`.
- **Breaking:** decompression, preparation and secrets dispatch through the new OEP families `DecompressionImpl` (only `glwe_decompress` and `lwe_decompress` required), `PreparationImpl` (only the GLWE, GGLWE, GGSW, public-key and GGLWE-to-GGSW-key preparations required) and `SecretImpl` (supertraits `SamplingImpl`, `HalVecZnxImpl`; fills derived); `impl_core_reference_full!` invokes their `impl_decompression_reference_full!`, `impl_preparation_reference_full!` and `impl_secret_reference_full!`, which out-of-tree backends add.
- **Breaking:** the `*Decompress` traits, the `*PreparedFactory` traits and `GLWESecretSampling`, `LWESecretSampling`, `GLWESecretTensorFromGLWESecret` and the secret conversions move to `api`, implemented on `Module<BE>` for the matching OEP family; the `*Decompress` traits become generic over `<BE>` (bound `Module<BE>: GGSWDecompress<BE>`), the factories lose their HAL supertraits (`GLWESecretPreparedFactory` its `SvpPPolPrepare`), and `GGLWEToGGSWKeyPreparedFactory` gains `GGLWEPreparedFactory`.
- **Breaking:** OEP supertraits: `EncryptionImpl` gains `SamplingImpl`, `SecretImpl`, `HalVecZnxImpl` and `GLWENormalizeImpl`, `GLWEKeyswitchImpl` gains `HalVecZnxDftImpl`, `GLWECIConversionImpl` gains `HalVecZnxImpl` and `EncryptionImpl`, and `AutomorphismImpl` drops `GLWEKeyswitchImpl`; `GLWERotateImpl` and `GLWEMulXpMinusOneImpl`, and through them `GGSWRotateImpl`, `GLWEPackImpl` and `ConversionImpl`, require `Backend<Ring = Standard>`.
- **Breaking:** OEP families require new methods, each with its query, which a hand-written impl forwards to the `_reference` bodies: `GLWETensoringImpl` `glwe_mul_relinearize_sum`, `glwe_mul_prepared_relinearize_sum`, `glwe_tensor_apply_prepared`, `glwe_tensor_apply_prepared_sum_dft` and `glwe_square_operand_prepare`; `GLWELinearTransformationImpl` `glwe_linear_transformation_eval_sum_into`, `glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes`, `glwe_giant_step_batch` and `glwe_accumulate_giant_steps`; `AutomorphismImpl` `glwe_automorphism_many`; `GLWEMulConstImpl` `glwe_const_combination` (`ConstTerm`); `GGLWEProductDigitsStridedImpl` `gglwe_product_fused`, `gglwe_product_idft_normalize` and `gglwe_products_from_dft`, forwarded by `impl_gglwe_product_digits_strided_reference!()` inside the impl. Public key-switch and linear-transformation reference functions take `module: &Module<BE>` in place of a module type parameter.
- **Breaking:** `GLWEKeyswitch` and `GLWEKeyswitchImpl` gain `glwe_keyswitch_dft` and `glwe_keyswitch_dft_add` with their queries: the gadget product of a key switch without the body, left in the DFT domain or added to an accumulator, finished with `vec_znx_idft_normalize`.
- **Breaking:** `GLWESecretCIEmbed` moves to `api` and dispatches with `GLWECIKeyEncryptSk` through `GLWECIConversionImpl` (required `glwe_secret_ci_embed`); `glwe_secret_ci_embed(&mut res, &src)` writes into a caller secret instead of returning one; `glwe_ci_key_prepare` takes any `O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees` in place of a `GLWECIKey`.
- **Breaking, behaviour:** `glwe_ci_embed_key_encrypt_sk` and `glwe_ci_trace_key_encrypt_sk` carve the embedded secret from scratch (`glwe_ci_key_encrypt_sk_tmp_bytes` grows) and wipe it, and refuse a secret of mismatched degree or rank or of the `NONE` law before touching it.
- **Breaking:** `glwe_pack_galois_elements` takes the operand degree first, `(n: Degree)` ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)).
- **Breaking:** `glwe_trace` and `glwe_trace_assign` take `levels: Range<usize>`, `reverse` and `normalize` in place of `skip`, and `glwe_trace_galois_elements` the operand degree, levels and direction; level 0 is the conjugation (standard ring only). `impl_core_reference_full!` includes `GLWETraceImpl`, which conjugate-invariant backends now implement, so a backend calling it drops its own `impl_glwe_trace_derived_full!`.
- **Breaking, behaviour:** every gadget product with a GGLWE key (key switches, automorphisms, relinearization, one-pass products, linear transformations, trace, packing, LWE conversions, `ggsw_expand_rows_assign`) panics on an input wider than the key's `max_input_k`, where it dropped the excess digits; narrow it first with `SetK::set_k` ([#451](https://github.com/poulpy-fhe/poulpy/issues/451)); `layouts::assert_key_covers` checks a key.
- **Breaking:** `get_automorphism_key` and `get_tensor_key` return `CoreError::GGLWEKeyUse` for a key not covering the width asked for, and `GetTensorKey` implementors provide `lookup_tensor_key`.
- **Breaking:** gadget layouts store their digit width in bits ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)): the `dsize` field and allocator argument become `k_digit: TorusPrecision`, `k_digit()` is required on `GGLWEInfos` and `GGSWInfos` (`dsize()` derived as `ceil(k_digit / base2k)`), `with_dsize`, `at_dsize` and `set_dsize` become `with_k_digit`, `at_k_digit` and `set_k_digit`, `key_k` drops `base2k` and returns `dnum * k_digit + k_aux`, and `key_size`, `key_work_size` and `dnum_for_input` (without `base2k`) take `k_digit`. The guard rule is `k_aux >= k_digit`; `k_digit = dsize * base2k` reproduces the former keys and noise.
- **Breaking:** GGLWE keys take a row offset `k_offset` in bits, by which a key switch lowers its input (an offset of 0 is the former key): `GGLWEInfos` requires `k_offset`, and `GGLWELayout`, `GLWESwitchingKeyLayout` and `GadgetProductOutputSizeParams` gain the field. Only `glwe_keyswitch` accepts an offset; other consumers panic (`layouts::assert_no_row_offset`).
- **Breaking, format:** serialized GGLWE and GGSW keys store `k_digit` in place of `dsize`, GGLWE keys also `k_offset`, so keys written before do not read back; serialized `GLWE` and `GLWECompressed` carry their plaintext metadata.
- **Breaking:** the reference GGLWE and GGSW encryptions and `GGLWENoise`/`GGSWNoise` require `BE: HalVecZnxImpl` in place of `VecZnxZero`, `VecZnxAddScalarAssign` and `VecZnxNormalizeAssign`; `GLWEKeyswitchInternal` and `GLWEExternalProductDftReference` require `VecZnxGadgetDecompose` and `VecZnxGadgetDecomposeTmpBytes`.
- **Breaking:** `GLWE`, `GLWEPlaintext` and `GLWECompressed` carry an optional `GLWEPlaintextMeta` (`Scale`, `SlotsKind`, `log_sparsity`) through `GLWEPlaintextInfos` and `SetGLWEPlaintextInfos`; `SlotsKind` moves from `poulpy-ckks` and gains `Integer`.
- **Breaking:** `LinearTransformation<P>` and `LinearTransformationPlan` gain the public field `rotations` (`BsgsRotations::PerRotation` as before), and `LinearTransformationStrategy` gains `BsgsPowersOfTwo`; `glwe_right_operand_prepare` panics unless `b_prep` has `ceil(b_k / base2k)` limbs.
- **Breaking:** `glwe_linear_transformation_eval_into` requires `keys: GetAutomorphismKey + GetTensorKey`; its queries take `(res, a, pt, plan, tsk, key)`, `glwe_linear_transformation_baby_steps_prepare_tmp_bytes` the plan before the key, and `alloc_prepared`, `alloc_prepared_from_index` and the rhs query `GLWEInfos`. `PreparedDiagonal` and `LinearTransformation` replace `log_scale`, `set_log_scale`, `real_slots` and `set_real_slots` with `plaintext_meta` and `set_plaintext_meta`.
- **Breaking:** `GLWEDiagonalProd` requires `GLWEInfos` and `operand` (a `DiagonalOperand`), provides `accumulate_giant_prod`, and `glwe_accumulate_streamed_baby_steps_dft` is removed.
- **Breaking:** `LinearTransformationBabySteps<BE, L = CnvPVecLOwned<BE>>` takes its operand type (any `layouts::BabyStepOperand<BE>`, such as scratch views), `baby_step` returns `&L`, and the methods taking a cache gain `L: BabyStepOperand<..>`; `take_glwe_linear_transformation_baby_steps_scratch(module, plan, a)` carves one.
- **Breaking:** `GLWEPlaintext::{encode,decode}_vec_{i64,i128}_strided` are removed.
- **Breaking:** test suites: `test_glwe_encrypt_pk_imprecise_key`, `test_glwe_encrypt_pk_zero_ephemeral`, `test_glwe_encrypt_pk_nan_ephemeral`, `test_glwe_public_key_rank_zero`, `test_glwe_mul_plain_compact` and `test_glwe_prepare_linear_transformation_rhs_rejects_wide_diagonal` become `test_glwe_encrypt_pk_rejects_imprecise_key`, `test_glwe_encrypt_zero_pk_rejects_zero_ephemeral`, `test_glwe_encrypt_zero_pk_rejects_nan_ephemeral`, `test_glwe_public_key_alloc_rejects_rank_zero`, `test_glwe_mul_plain_plaintext_degree` and `test_glwe_linear_transformation_rhs_prepare_rejects_wide_diagonal`, the first three and the last registered without `#[should_panic]`; register `test_glwe_trace_sum` and `test_glwe_trace_parity` beside `test_trace_packing_parity`.
- **Breaking:** `EncryptionParityBackend` gains `DecompressionImpl` and `PreparationImpl` and drops `ConversionImpl`, moving the LWE-matrix case to `test_lwe_matrix_decrypt_parity`; `test_suite::noise::TestBackend` gains `SecretImpl`, and `ParityShapes` gains `unaligned_digits`.
- **Behaviour:** `glwe_public_key_generate` and `glwe_encrypt_pk_smudged` wipe their whole query, as the `EncryptionImpl`, `DecryptionImpl` and `GLWECIConversionImpl` contracts now require of every scratch-taking method.
- **Behaviour:** refusals run before any draw or write (secret-key encryptions refuse a `NONE` secret; `glwe_prepare`, `gglwe_to_ggsw_key_prepare`, `gglwe_to_ggsw_key_decompress`), `lwe_encrypt_sk`, `lwe_decrypt` and `glwe_decrypt` check dimension or rank in every build, and refusals on secret paths print no operand values.
- Gadget products (key switching, external products, relinearization, linear transformations) take digits of any width ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)); whole-limb digits are unchanged bit for bit.
- Add BSGS linear transformations with power-of-two rotation keys only (`BsgsRotations::PowersOfTwo`): about `log2(n1) + log2(n2)` keys instead of `n1 + n2`. Linear transformations accept ciphertext diagonals, and `glwe_linear_transformation_eval_sum_into` sums `M_t(a_t)` over transforms sharing one giant schedule at the giant cost of one.
- Add `glwe_mul_relinearize_sum` (`GLWEMulRelinearizeSum`), `relinearize(Σ_i a[i] * b[i])` in one pass, `GLWEPreparedTensoring` (`glwe_mul_prepared_relinearize_sum`, `glwe_tensor_apply_prepared`) over operands prepared once, and `glwe_left_operand_prepare`.
- Add `GLWEMulPreparedPlainSum` (`glwe_mul_prepared_plain_sum` with `_add`, `_dft` and `_dft_add` forms; `GLWEMulPreparedPlainSumImpl`): `Σ_t a[t] ⊛ σ_{galois[t]}(b[t])` over prepared ciphertexts and plaintexts, normalized once.
- Add `glwe_automorphism_many`, `gglwe_copy` and `ggsw_copy` (`impl_gglwe_copy_reference_full!`, `impl_ggsw_copy_reference_full!`), `glwe_switch_degree` (`GLWESwitchDegree`), `GLWE::write_live_to` and `read_live_from`, `Diagonals::from_slot_map` and `shrink_period`, and `TransferInto` for public keys, compressed GLWE and GGLWE, `GLWECIKey` and into a `GGLWEViewMut`.
- Add the `poulpy_core::gadget` helpers (`gglwe_gadget_row_plaintext`, `gglwe_gadget_dft_apply`, `precision_limbs`, ...), `GGLWELayout::max_input_k`, `From<GGLWELayout>` for the key layouts, `layouts::max_lazy_terms`, `digits_aligned` and `glwe_host_alloc_from_infos`.
- **Fix:** the one-pass products and squares lost precision against a tensor key covering only the operands' limbs at an offset off a limb boundary; they relinearize through `glwe_tensor_relinearize` unless the one-pass modeled variance is within `2^-4` of it (`ONE_PASS_VARIANCE_SLACK`) ([#461](https://github.com/poulpy-fhe/poulpy/issues/461)).
- **Fix:** the external product and key-switch body read only the limbs an operand's `k` covers, where a wider buffer overran the external product's scratch and entered the key-switch result.
- **Fix:** `glwe_tensor_relinearize_tmp_bytes` undercounted for a tensor key of another `base2k` than the operand's.
- **Fix:** `glwe_linear_transformation_eval_tmp_bytes` counts the normalization and iDFT scratch of the lazy giant-step path and the per-giant fallback ([#426](https://github.com/poulpy-fhe/poulpy/issues/426)).
- **Fix:** `glwe_linear_transformation_eval_into` checks every giant step before writing `res`.
- **Fix:** `glwe_secret_tensor_from_glwe_secret` takes its temporaries from scratch and wipes it ([#425](https://github.com/poulpy-fhe/poulpy/issues/425)).
- **Fix:** `take_gglwe_scratch` carves `dnum` rows, where it carved `dnum / dsize`.

### `poulpy-ckks`

- **Breaking:** ckks names lead with the object they act on, with no alias ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `ckks_eval_linear_transformation_*` becomes `ckks_linear_transformation_eval_*`, `ckks_prepare_linear_transformation_{rhs, baby_steps}[_tmp_bytes]` becomes `ckks_linear_transformation_{rhs, baby_steps}_prepare[_tmp_bytes]`, `reference::ckks_encode_linear_transformation_from_diagonals` becomes `encoding::ckks_linear_transformation_encode_from_diagonals`, `ckks_new_dft_matrix` becomes `ckks_dft_matrix_new`, `ckks_dft_evaluate_assign` becomes `ckks_dft_matrix_eval_assign`, the in-place `ckks_coeffs_to_slots` and `ckks_slots_to_coeffs` become `ckks_coeffs_to_slots_assign` and `ckks_slots_to_coeffs_assign`, and `ckks_eval_approximation` and `ckks_approximation_tmp_bytes` become `ckks_polynomial_approximation_eval[_tmp_bytes]`.
- **Breaking:** the encoding-buffer `ckks_coeffs_to_slots_assign` and `ckks_slots_to_coeffs_assign` become `ckks_encoding_buffer_coeffs_to_slots_assign` and `ckks_encoding_buffer_slots_to_coeffs_assign`, `ckks_encode_slots_assign_into` becomes `ckks_encode_slots_into`, and `ckks_encode_reim_into`, `ckks_decode_reim_into` and `ckks_reim_tmp_bytes` become `ckks_encode_slots_host_into`, `ckks_decode_slots_host_into` and `ckks_encoding_host_tmp_bytes`. The required `CKKSEncodingImpl<F>` kernels are `ckks_quantize_coeffs_into`, `ckks_dequantize_coeffs_into` and `ckks_encoding_buffer_{slots_to_coeffs,coeffs_to_slots}_assign_with_plan`.
- **Breaking:** `EncodeBSGS`, `PowerBasisGen`, `PowerBasisInsert`, `PaCoScalar` and `ShipScalar` become `CKKSPolynomialEncode`, `CKKSPowerBasisGen`, `CKKSPowerBasisInsert`, `CKKSPaCoScalar` and `CKKSShipScalar`, and `gen_dft_matrices[_blockwise]` and `compile_eval_mod` become `ckks_dft_matrices_gen[_blockwise]` and `ckks_eval_mod_compile`. `reference::dft::DftScalar` is `layouts::DftScalar`; `reference::dft::matrices`, `reference::linear_transformation_diagonals` and `oep::defaults` are removed.
- **Breaking, behaviour:** `CKKSModuleAlloc` splits into `CKKSCiphertextAlloc`, `CKKSPlaintextAlloc` and `CKKSLinearTransformationAlloc` (prepared operands and DFT matrices in `CKKSLeftOperandPreparedFactory`, `CKKSRightOperandPreparedFactory` and `CKKSDFTMatrixPreparedFactory`), and `ScratchArenaTakeCKKS` into `ScratchArenaTakeCKKSEncodingBuffer`, `ScratchArenaTakeCKKSPlaintext` and `ScratchArenaTakeCKKSCiphertext`. `ckks_pt_vec_alloc` and `ckks_pt_vec_alloc_compact` become `ckks_plaintext_alloc_from_slots(slots, base2k, k)`, allocating at the degree that holds the slots and panicking above `ckks_max_slots` or below the backend's `MIN_DEGREE`, where `ckks_pt_vec_alloc_compact` clamped (repeat the slots up to `MIN_DEGREE`); `ckks_pt_coeffs_alloc` is removed for `ckks_plaintext_alloc(coeff_count.into(), base2k, k)`.
- **Breaking, behaviour:** a plaintext is always dense: encoders take exactly one value per coefficient, or `ckks_slots_of_degree(n)` slots at degree `n`, and return an error for fewer instead of spreading them. The test suite's `compact` module is `plaintext_slots` (`test_plaintext_slots_add_sub_mul`, `test_linear_transformation_diagonal_slots`, `test_diagonal_encoder_slots`), and `test_compact_plaintext_embedding` is removed.
- **Breaking:** `CKKSMeta` is removed for the wrapped GLWE's `GLWEPlaintextMeta` (`Scale::Log(log_delta)`, `slots`, `log_sparsity`); `CKKSInfos` extends the core traits, `SetCKKSInfos` is blanket-implemented over them, and a plaintext's sparsity is `log2(N / n)` ([#342](https://github.com/poulpy-fhe/poulpy/issues/342)). The `impl_ckks_infos!` arms `inner_meta` and `self_meta` are `ckks_inner` and `glwe_inner`.
- **Breaking:** `ckks_prepare_right` becomes `ckks_right_operand_prepare_alloc`, taking a `PrepareHint`, and `CKKSPreparedRight` becomes `CKKSRightOperandPrepared` (deprecated alias kept); add `CKKSLeftOperandPrepared` and, per side, `ckks_{left,right}_operand_prepare` (into, `InvalidOperand` for another layout), `_prepare_alloc`, `_prepare_tmp_bytes`, `_prepared_alloc_from_infos` and `_prepared_bytes_of_from_infos`. `CKKSMulImpl` requires the prepare methods and queries, and `CKKSMulReference::ckks_prepare_right_reference` is removed.
- **Breaking:** OEP methods take their API name without `_impl` (`CKKSAddImpl::ckks_add_into_impl` becomes `ckks_add_into`); `DFTImpl` becomes `CKKSDFTImpl`, `DFTMatrixImpl<F>` `CKKSDFTMatrixImpl<F>`, `CKKSPlaintextZnxImpl` `CKKSPlaintextVecImpl`, `CKKSPlaintextReference` `CKKSPlaintextVecReference`, `CKKSEvalModOpsReference` `CKKSEvalModReference` and `PolynomialEvaluationReference` `CKKSPolynomialEvaluationReference`. The slot-shift hooks `ckks_rotate_by_{tmp_bytes,into,assign}_impl` become `ckks_rotate_{tmp_bytes,into,assign}`, `ckks_conjugate_with_keys_assign_impl` becomes `ckks_conjugate_assign`, and the one-key hooks `ckks_rotate_{tmp_bytes,into,assign}_impl` and `ckks_conjugate_{into,assign}_impl` become `ckks_rotate_with_key_*` and `ckks_conjugate_with_key_*`, `CKKSConjugateImpl` requiring `ckks_conjugate_with_key_tmp_bytes`.
- **Breaking:** `impl_ckks_<family>_reference!` becomes `impl_ckks_<family>_reference_full!` (add, sub, copy, neg, pow2, imag, conjugate, rotate, mul, encryption, polynomial_evaluation, complex_polynomial_evaluation, eval_mod, bootstrapping, encapsulated_mod_up), `impl_ckks_plaintext_reference!` `impl_ckks_plaintext_vec_reference_full!`, `impl_ckks_fold_reference!` splits into `impl_ckks_fold_layout_reference_full!` and `impl_ckks_fold_reference_full!` (implementing `CKKSFoldImpl<Standard>` and `CKKSFoldImpl<ConjugateInvariant>`, no blanket `CKKSFoldImpl<R>`), and `impl_ckks_dft_reference!` into `impl_ckks_dft_reference_full!` and `impl_ckks_dft_matrix_reference_full!`. References are methods of `<Family>Reference` traits on `Module<BE>`, `reference::dft` functions take `_reference`, and `reference::encoding::{encode,decode}_coeffs_into_host` become `ckks_{quantize,dequantize}_coeffs_into_reference`.
- **Breaking:** every API method dispatches through a same-named OEP method. New families: `CKKSDotProductImpl`, `CKKSLinearTransformationImpl`, `CKKSInnerSumImpl` and `CKKSAddManyImpl` (`impl_ckks_{dot_product,linear_transformation,inner_sum,add_many}_reference_full!`), the derived `CKKSMulAddImpl`, `CKKSAffineImpl`, `CKKSMulSubImpl` and `CKKSApproximationImpl` (`impl_ckks_{mul_add,affine,mul_sub,approximation}_derived_full!`), all required by `CKKSImpl`, and the standard-ring `CKKSCIRingMapImpl`, `CKKSPaCoImpl` (replacing `PaCoSlotOps`) and `CKKSShipImpl` (`impl_ckks_{ci_ring_map,paco,ship}_reference_full!`). An out-of-tree backend registers the new umbrella `impl_ckks_reference_full!` (every scalar-independent both-ring family) and, if standard, the ring-map, PaCo and SHIP opt-ins.
- **Breaking:** `CKKSPaCoCoeffEncodingImpl::ckks_paco_coeff_encodings` takes `(ciphertext, context, scratch)` with a `PaCoContext<Self, F>` and `ckks_paco_coeff_encodings_tmp_bytes` `(context, input)`; `CKKSMulImpl::ckks_mul_pt_{vec,const}_tmp_bytes` take the plaintext `b` instead of `b_k`; `CKKSEncryptionImpl::ckks_encrypt_sk<Dct, S, Pt>` becomes `<Dct, Dpt, S>`, `ckks_decrypt<S, Dct, Pt>` `<Dpt, Dct, S>`, `ckks_extract_pt<Dst, Src>` `<D, S>`, and `CKKSMulImpl`'s tensor-key generic `T` is `H`. `CKKSDFTMatrixImpl<F>` requires `ckks_dft_matrix_new_tmp_bytes`, PaCo and SHIP coefficient encodings refuse an arena below their own query, and a hand-written `ckks_decrypt` refuses mismatched degrees itself.
- **Breaking:** `impl_ckks_bootstrapping_reference_full!` requires `CKKSModUpFirstFactorImpl` (`impl_ckks_mod_up_first_factor_reference_full!`), the functional bootstrap reference `CKKSNegOps` instead of `CKKSAffineOps`, and the polynomial-evaluation reference `GLWEMulConstImpl`; `CKKSBootstrappingOps` drops its `CKKSDFTOps` and `CKKSEvalModOps` supertraits.
- **Breaking:** op-wide scalars lead, plans and keys follow the operands: `ckks_rotate_into`, `ckks_rotate_assign` and `ckks_conjugate_rotate_into` take `k` first, `ckks_{mul,div}_pow2_{into,assign}` `bits` first, `ckks_fold_layout`, `ckks_fold_tmp_bytes` and `ckks_fold_count` `degree` first, `ckks_unfold` `(outs, folded, automorphisms, outbound, scratch)`, `ckks_polynomial_approximation_eval_tmp_bytes` `(res, input, approximation, tsk)`, `ckks_all_ops[_with_atk]_tmp_bytes` `(ct_infos, pt_prec, tsk_infos[, atk_infos])`, `ckks_eval_poly_{real,complex}_const_coeffs_from_power_basis` `(res, power_basis, poly, tsk, scratch)`, `ckks_ship_coeff_encodings` `(base2k, complex, ciphertext, plan, scratch)` and `ckks_{add,sub}_pt_const_into` `(dst, dst_coeff, a, pt, pt_coeff, scratch)`. `encoding::paco_coeff_encodings_host` and `ship_coeff_encodings_host` become `ckks_paco_coeff_encodings_host(module, base2k, ct, plan)` and `ckks_ship_coeff_encodings_host(module, base2k, complex, ct, plan)` over a `&GLWE<D, i64>` (pass `&*ct`), as do `poulpy_cpu_portable`'s `paco_coeff_encodings_staged` and `ship_coeff_encodings_staged`.
- **Breaking:** `ckks_add_many(dst, inputs, scratch)` becomes `ckks_add_many(normalize_after, res, a, scratch)`, normalizing after every `normalize_after` additions (`usize::MAX` for never), and `ckks_add_many_tmp_bytes(res_size)` becomes `ckks_add_many_tmp_bytes(res, a)`.
- **Breaking:** `ckks_double_into` and `ckks_bootstrap_mod_up` get their own queries, `ckks_double_tmp_bytes` and `ckks_bootstrap_mod_up_tmp_bytes`, with derived defaults in their OEP families; `sign_composite_coeffs` takes default `RemezOptions` and no options argument, and `sign_composite_coeffs_with_margin(.., opts)` becomes `sign_composite_coeffs_with(tau, error_margin, target_bits, degrees, max_factors, opts)`; `CKKSPlaintextVecReference` loses `ckks_{add,sub}_pt_{vec,const}_into_reference`, now crate helpers ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)).
- **Breaking, behaviour:** `ckks_dot_product_ct` takes any ciphertext types (`<Dst, A, B, H>`) and relinearizes once for two pairs or more sharing one layout and scale per side, which changes its bits (on the one-pass path, pick `base2k` for `a.len()` times one pair's products).
- **Breaking:** `ckks_linear_transformation_eval_tmp_bytes` and `_eval_streamed_tmp_bytes` take `(ct, diagonal, plan, tsk, key)`, `plan` the transform's `LinearTransformationPlan`; evaluations and the DFT, bootstrapping and PaCo key sets require `GetAutomorphismKey + GetTensorKey` and drop `IntPolyInfos`; baby-step and evaluation methods gain `L: BabyStepOperand<..>`; `LtDiagonalMeta` requires `lt_log_sparsity`. `ckks_linear_transformation_rhs_prepare` requires `GLWEToBackendRef + CKKSCtBounds` and returns `CKKSResult<()>`, and its `_tmp_bytes` and `ckks_dft_matrix_prepare_tmp_bytes` take `GLWEInfos`.
- **Breaking, behaviour:** `ckks_linear_transformation_eval_self_{into,assign}` carve their baby-step cache from scratch, sized by the new `ckks_linear_transformation_eval_self_tmp_bytes(streamed, ct, diagonal, plan, tsk, key)`.
- **Breaking:** `ckks_prepare_dft_matrix_tmp_bytes` becomes `ckks_dft_matrix_prepare_tmp_bytes` and the allocating `ckks_prepare_dft_matrix(dft, scratch)` becomes `ckks_dft_matrix_prepare_alloc(dft, scratch) -> CKKSResult<..>`; `ckks_dft_matrix_prepare(res, dft, scratch)` prepares into a caller-owned matrix (`ckks_dft_matrix_prepared_alloc_from_infos`, `ckks_dft_matrix_prepared_bytes_of_from_infos`), and `BootstrappingContext::prepare` returns `anyhow::Result`.
- **Breaking:** the prepared `BootstrappingContext` is `BootstrappingContextPrepared` (with `compile`, `compile_ci`, `compile_unprepared`, `compile_ci_unprepared`, `generate_keys`, `eval_mod`), and `BootstrappingContextUnprepared` is `BootstrappingContext` (deprecated alias kept); `eval_mod` returns an `Option`, `None` for EvalMod on the conjugate-invariant ring (`eval_mod_plan` serves both).
- **Breaking:** `BootstrappingKeys` splits into `BootstrappingLinearKeys` (rotation, encapsulation and first-factor keys), `BootstrappingKeys` (adds the standard tensor key, no longer required to be `GLWETensorKeyPrepared`; parallel paths need `TensorKey: Sync`) and `BootstrappingCIKeys<BE, BCI>`; `BootstrappingKeysPrepared` takes the EvalMod backend as last type parameter. `BootstrappingKeysLayout::automorphism_key` becomes `rotation_keys`, a layout per Galois element of `BootstrappingPlan::key_widths` (no longer `Copy`), key layouts, sets and prepared keys gain first-factor and conversion keys, and `generate_keys` requires `VecZnxAutomorphism`.
- **Breaking:** the sparse-to-dense key is optional, absent when the first factor runs on the raise: `EncapsulationKeysLayout::sparse_to_dense` and the second key of `encapsulation_keys` are `Option`s (alias `EncapsulationKeys`).
- **Breaking, behaviour:** presets: `n16_d35_k600_p19_c2s` becomes `n16_d35_k636_p20_c2s` and `n16_d35_k720_p19_s2c` becomes `n16_d35_k733_p20_s2c` (three factors per transform, 20 bits, 17 levels), and `n15_d35_k180_p18_c2s` runs CoeffsToSlots at `2^50` and SlotsToCoeffs at `2^29`. Their sparse-to-dense key has a row offset, and every key takes the whole-limb digit of fewest rows its use admits under its secret's modulus bound, so 0.9.0 key sets do not match; `with_dsizes` caps the high-modulus keys' digit, and `key_dsize` and the new `key_k_digit` give that of the widest rotation key.
- **Breaking, behaviour:** `EncodedLut::binary` folds its final map into the cosine polynomial and double-angle steps, so `EncodedLut::consumed_bits` and `BootstrappingPlan::functional_bootstrap_k` drop by the LUT's coefficient scale and outputs change in their low bits.
- **Breaking:** `CKKSCompositionError::MissingAutomorphismKey` and `MissingRelinearizationKey` carry the key source's error as `source`, so a key too narrow is no longer reported as absent.
- **Breaking:** `ckks_backend_test_suite!` loses `encoding_determinism` for `ckks_determinism_test_suite!(mod <name>, backend = <type>[, f32])`, and `test_encoding_determinism<BE, F, E>(params, module, host_module)` becomes `test_encoding_determinism::<BE, F>()`. Rejection tests are `test_<op>_rejects_<case>`: `test_add_pt_vec_{alignment,base2k_mismatch}_error` become `test_add_pt_vec_assign_rejects_{alignment,base2k_mismatch}`, `test_mul_prepared_layout_mismatch_error` `test_mul_prepared_assign_rejects_layout_mismatch`, `test_mul_ct_explicit_metadata_error` `test_mul_into_rejects_explicit_metadata`, `test_div_pow2_assign_explicit_error` `test_div_pow2_assign_rejects_insufficient_capacity`, `test_rotate_assign_missing_key_error` `test_rotate_assign_rejects_missing_key`, `test_encryption_degree_mismatch_error` `test_encrypt_sk_rejects_degree_mismatch`, `test_decrypt_extract_output_hom_rem_too_large` `test_decrypt_rejects_output_hom_rem_too_large`, `test_decrypt_extract_{base2k,degree}_mismatch_error` `test_decrypt_rejects_{base2k,degree}_mismatch` and `test_repeated_square_exhausts_capacity` `test_square_into_rejects_exhausted_capacity`.
- **Breaking:** test utilities: `bootstrapping_presets_meet_precision` takes no fixture radix, running each preset at the radix of `test_suite::presets::preset_for_backend` (failure `2^-128`), and requires `CKKSBootstrappingCIOps`, `PresetRunModule`, `HostStaged` and `CKKSDFTOps`; `BootstrappingPresetRun<BE, BC = BE>` takes a client backend. `assert_{mul_pt,mul_ct,unary,binary}_output_meta` take the destination's `k` before the call, and `gen_ci_keys` and `test_conjugate_invariant_ring_map` require `GLWECIKeyPreparedFactory`.
- Add the modulus raise that applies the first CoeffsToSlots factor on the raised ciphertext, consuming no modulus (`DFTPlan::with_first_factor_on_raise`, `BootstrappingPreset::with_first_factor_on_raise`, after ePrint 2025/1403): each diagonal's term switches to the dense secret by its own `first_factor_key`, laid out from `presets::first_factor`. The encapsulated ModUp also takes a sparse-to-dense key with a row offset (`k_offset`, `BootstrappingPlan::raised_window`).
- Add S2C-first bootstrapping with EvalMod once on the conjugate-invariant ring (`EvalModRing`, `with_eval_mod_ring` on the plan and preset): `CKKSBootstrappingCIOps::ckks_bootstrap_ci_into` takes a standard module of degree at least `2N`, the CI module of degree `N`, a context of `BootstrappingContextPrepared::compile_ci` and keys of `generate_keys_ci` and `BootstrappingKeySet::prepare_ci`. Backends override it through `CKKSBootstrappingCIImpl` (`impl_ckks_bootstrapping_ci_reference_full!($be, $ci)`); the folding of `u` into the last CoeffsToSlots factor (`CIConversion::Folded`) costs no level, and EvalRound+ runs on this ring too.
- Add the preset `n16_d35_k685_p19_s2c_ffr_ci` to `presets::bootstrapping::all`: `n16_d35_k733_p20_s2c` with the first factor on the raise and EvalMod on the CI ring, 19 bits and 15 levels.
- Add `with_power_of_two_keys` on `BootstrappingPreset` and `DFTPlan` (`BsgsPowersOfTwo`), and `BootstrappingPreset::with_eval_round_plus`, `max_backend_base2k` (the largest radix a backend's precision admits at failure probability `2^-failure_bits`) and `with_backend_base2k`.
- Add `BootstrappingPlan::key_widths` (`BootstrappingKeyWidths`), `BootstrappingPlan::generate_keys[_ci]` (keys from the plan alone), `BootstrappingKeySet::transfer_and_prepare[_ci]` (one key at a time on the evaluating backends), `DFTPlan::galois_elements_per_factor`, `DFTPlan::first_factor_rotations`, `ckks_dft_matrix_new_tmp_bytes` and `BootstrappingContextPrepared::compile_unprepared_tmp_bytes`.
- Add `CKKSInnerSumOps`: `ckks_inner_sum[_assign]` sums the rotations by `j 2^log_batch` slots for `j < 2^log_count`, or their mean, and `ckks_replicate[_assign]` those by `-j 2^log_batch`, with `ckks_{inner_sum,replicate}_galois_elements` and `ckks_inner_sum_tmp_bytes`.
- Add `ckks_linear_transformation_eval_sum_into`, `Σ_t M_t · src_t` over resident or streamed transforms sharing one giant schedule, at the key-switch cost of one, with `_eval_sum_tmp_bytes` and `_eval_sum_streamed_tmp_bytes`.
- Linear transformations take ciphertext diagonals, multiplied at the rule of `ckks_mul_into` and prepared by `ckks_linear_transformation_rhs_prepare` or streamed (`CKKSCiphertext` implements `GLWEDiagonalProd` and `LtDiagonalMeta`). Add `ckks_plaintext_diagonal_layout`, `ckks_linear_transformation_alloc_from_infos`, `encoding::ckks_linear_transformation_encode_into` and `_encode_tmp_bytes`, and `ckks_linear_transformation_output_k` and `_output_layout`.
- Add `ckks_dot_product_prepared` (`ckks_dot_product_ct` of prepared operands, bit for bit), `ckks_dot_product_prepared_pt_vec` (prepared plaintexts read through odd Galois elements), its forms `_dft`, `_dft_add`, `_from_dft` and `_add`, each with its `_tmp_bytes`; prepared operands are `CKKSCtBounds`.
- `ckks_add_many` and the dot products no longer reject more than `2^(63 - base2k)` terms. `ckks_decrypt` writes into a plaintext of a degree `n` dividing `N` with `log2(N / n) <= ct.log_sparsity`, `PlaintextDegreeMismatch` otherwise.
- Add `CKKSCompositionError::InvalidOperand { op, reason }`, `CKKSModuleInfos::ckks_slots_of_degree`, `layouts::ckks_plaintext_one` and `CKKSCiphertextViewMut::to_backend_view_ref`; `CKKSCiphertext` implements `WriterTo` and `ReaderFrom`, rejecting a flag-clear ciphertext.
- The SHIP mux-rotate takes keys of any digit width, its key generation taking the width in bits, and refuses a key with a row offset; `ShipKeySet::generate` wipes every copy of the secrets.
- `test_suite::parity::{helpers, keys, layout}` are public under `test-utils`; add `poisoned_scratch`, `assert_invalid_operand` and `assert_ckks_internal`.
- **Behaviour:** the bootstraps check every key at entry and refuse one used above its `max_input_k`; every operation returns `MissingAutomorphismKey` or `MissingRelinearizationKey` for a key not covering its input, and `ckks_rotate_into` and `ckks_conjugate_rotate_into` look keys up at `min(src.k(), dst.k())`.
- **Behaviour:** caller-fixable refusals return typed `CKKSCompositionError` variants instead of `CKKSError::Internal`, the ring maps refuse where poulpy-core panicked, and an error's `op` names the invoked public method (`ckks_rotate_into`, not `rotate`).
- **Behaviour:** call-local temporaries (polynomial evaluation, DFT, PaCo, SHIP) come from scratch, so their queries may report more, and a short arena is refused with `CKKSError::Internal` before any write.
- **Behaviour:** refusal messages on secret paths, `ShipSecretSpec`'s `Debug` and `assert_precision` print no secret-derived values.
- **Fix:** CKKS operations refuse before their first write, so a refused call leaves its outputs unchanged.
- **Fix:** `ckks_mul_into`, `ckks_mul_assign`, `ckks_mul_prepared_assign` and their compositions decoded too small by `2^min(budget gap, scale gap)` when the operand with the larger budget had the smaller scale.
- **Fix:** `ckks_mul_pt_vec_*` and `ckks_mul_pt_const_*` return `PlaintextBase2KMismatch` for another radix, where they computed wrong bits, and `ckks_mul_pt_const_*` `PlaintextCoefficientOutOfRange` for a coefficient past the plaintext.
- **Fix:** `ckks_decrypt` checks its radix, alignment and arena before writing and wipes its scratch; `ckks_fold` and `ckks_unfold` check a key's `max_input_k`, not its gadget width.
- **Fix:** `ckks_ci_trace`, and so `ckks_unfold` of conjugate-invariant inputs, lose their first-slot bias: the halving rounds ties to even.
- **Fix:** `ckks_linear_transformation_eval_into` and `_eval_sum_into` label their result sparse only where the sources and diagonals are.
- **Fix:** the linear-transformation, BSGS and EvalMod encoders return `InvalidOperand` for a `CoeffsMeta` with `k = 0`, where they panicked.
- **Fix:** `From<anyhow::Error> for CKKSError` recovers a `CKKSCompositionError` anywhere in the chain.
- **Fix:** linear transformations and bootstrapping key generation read Galois elements at the operands' degree, not the module's.
- **Fix:** the linear-transformation, PaCo and SHIP scratch queries left out the selected `ckks_copy` and `vec_znx_idft_apply`.

### `poulpy-bin-fhe`

- **Breaking:** names lead with their object ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `Cmux::{cmux, cmux_assign, cmux_assign_neg}` is `GLWECmux::{glwe_cmux, glwe_cmux_assign, glwe_cmux_neg_assign}`, `Cswap::cswap` is `GLWECswap::glwe_cswap_assign`, `ExecuteBDDCircuit*` and `execute_bdd_circuit*` are `BDDCircuitExecute*` and `bdd_circuit_execute*` (`execute_bdd_circuit_tmp_bytes_for` is `bdd_circuit_execute_circuit_tmp_bytes`), `scalar_to_ggsw_blind_rotation` is `ggsw_blind_rotation_from_scalar`, `ggsw_to_ggsw_blind_rotation_tmp_bytes` is `ggsw_blind_rotation_tmp_bytes`, `glwe_blind_retrieval_statefull[_rev]` is `glwe_blind_retrieval_stateful[_rev]_assign`, `lookup_table_rotate` is `lookup_table_rotate_assign`, `fhe_uint_debug_prepare` is `fhe_uint_prepare_debug`, `prepare_blind_rotation_key` is `blind_rotation_key_prepare` and `prepare_bdd_key` is `bdd_key_prepare`; `_tmp_bytes` queries and OEP families follow.
- **Breaking:** the `FheUint` arithmetic traits (`Add`, `Sub`, ..., `Identity`) take the `FheUint` prefix and `define_bdd_*_trait!` and `impl_bdd_*_trait!` are no longer exported; `alloc_fhe_uint_prepared[_from_infos]` is `fhe_uint_prepared_alloc[_from_infos]`, `ScratchArenaTakeBDD::take_fhe_uint` is `ScratchArenaTakeFheUint::take_fhe_uint_scratch`, and `BlindRotationAlgo::alloc_key` is `blind_rotation_key_alloc`, bounded by `GGSWAlloc + ModuleN`; `BlindRotationKeyAlloc` is removed ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)).
- **Breaking:** prepared and compressed key allocation are provided factory methods with no OEP method, requiring the core prepared factories: `blind_rotation_key_prepared_alloc` is `blind_rotation_key_prepared_alloc_from_infos`, `alloc_bdd_key_from_infos` is `bdd_key_prepared_alloc_from_infos` and `blind_rotation_key_compressed_alloc` is `blind_rotation_key_compressed_alloc_from_infos`, beside new `*_bytes_of_from_infos` counts. The `*KeyPreparedFactoryImpl` allocators, `BlindRotationKeyCompressedFactoryImpl` with its opt-in, and the allocation references are removed.
- **Breaking, behaviour:** key allocators and byte counts panic in every build on infos of another degree than the module.
- **Breaking:** `poulpy_bin_fhe::api::<family>::X` is `poulpy_bin_fhe::api::X`, and so for `oep`.
- **Breaking:** references are `<Family>Reference` trait methods on `Module<BE>`, and `*_ref` functions end in `_reference`; `mod_switch_2n_ref(modulus, res, lwe, direction)` is `blind_rotation_mod_switch_reference(module, modulus, direction, res, lwe)`.
- **Breaking:** opt-in macros are named after their OEP trait: `impl_bin_fhe_<op>_reference!` is `impl_<op>_reference_full!`, umbrellas included, where `blind_rotation_key_encrypt` becomes `blind_rotation_key_encrypt_sk`, `blind_rotation_key_compressed` `blind_rotation_key_compressed_encrypt_sk`, `*_key_prepared` `*_key_prepared_factory`, `lookup_table` `lookup_table_factory`, `execute_bdd_circuit*` `bdd_circuit_execute*`, and `cmux` and `cswap` `glwe_cmux` and `glwe_cswap`. `impl_bin_fhe_glwe_blind_retrieval_derived!` is `impl_glwe_blind_retrieval_derived_full!`; `impl_bin_fhe_reference_full!` keeps its name.
- **Breaking:** the executor picks the blind-rotation schedule, parallel exactly when `BE::TaskExecutor::IS_PARALLEL`, with the same bytes on every executor: the `scheduling = parallel` macro arms, `impl_bin_fhe_blind_rotation_execute_parallel!`, `blind_rotation_execute_parallel` and `blind_rotation_execute_tmp_bytes_parallel` are removed, and the reference bounds `VecZnxDftSubNegateAssign<BE>`.
- **Breaking, behaviour:** the FFT64 serial block blind rotation at extension factor one computes `acc + (X^{a_i}·v - v)` instead of `(acc + X^{a_i}·v) - v`, so its rounding can differ from 0.9.0.
- **Breaking:** `glwe_blind_rotation_assign_tmp_bytes`, `ggsw_blind_rotation_assign_tmp_bytes` and `circuit_bootstrapping_execute_tmp_bytes` are removed: use `glwe_blind_rotation_tmp_bytes(res, res, k)`, `ggsw_blind_rotation_tmp_bytes(res, res, k)` and `circuit_bootstrapping_execute_to_constant_tmp_bytes`.
- **Breaking:** op-wide scalars lead the argument lists: `sign, bit_rsh, bit_mask, bit_lsh` in the GLWE and GGSW blind rotations, `bit_rsh, bit_mask` in `glwe_blind_selection` and `glwe_blind_retrieval_stateful[_rev]_assign`, `log_domain, extension_factor` (after `log_gap_out`) in the circuit bootstrapping executions, prepares and `CircuitBootstrappingKeyPrepared` methods, `direction` in `blind_rotation_mod_switch(modulus, direction, res, lwe)` and `k` in `lookup_table_set(k, res, f)`. Swapped equally typed scalars still compile.
- **Breaking:** plans and keys follow the operands: `bdd_circuit_execute_{1w,2w}_to_1w(out, circuit, a[, b], key, scratch)` is `(out, a[, b], circuit, key, scratch)`, their queries take `(res_infos, ggsw_infos, circuit, key)`, `bdd_circuit_execute_tmp_bytes` `(state_size, res_infos, ggsw_infos)`, `bdd_circuit_execute_circuit_tmp_bytes` `(res_infos, ggsw_infos, circuit)` and `circuit_bootstrapping_execute_prepared` `(res, lwe, plan, key, scratch)`. Convenience methods take `module` right after `self`: the `FheUint` arithmetic `<op>_multi_thread(module, threads, ..)`, `FheUintPrepared::prepare_custom_multi_thread(module, threads, ..)` and `LookupTable::set(module, k, f)`.
- **Breaking:** the blind-rotation key encryptions take `sk_lwe` before `sk_glwe`, and `BlindRotationKeyCompressedEncryptSk<B, BRA>` is `BlindRotationKeyCompressedEncryptSk<BRA, BE>`.
- **Breaking:** `FheUintPrepared` and `FheUintPreparedDebug` allocators take `k_digit` in place of `dsize`.
- **Breaking:** `BlindRotationExecuteImpl`, `LookupTableFactoryImpl`, `CircuitBootstrappingExecuteImpl`, `GLWEBlindRotationImpl`, `GGSWBlindRotationImpl`, `BDDCircuitExecute{1WTo1W,2WTo1W}Impl` and `FheUintPrepareImpl` require `Backend<Ring = Standard>`; a backend registering `impl_blind_rotation_key_decompress_reference_full!` implements `poulpy_core::oep::DecompressionImpl`.
- **Breaking:** `poulpy_bin_fhe::<family>::test_suite` is `poulpy_bin_fhe::test_suite::<family>`; `bin_fhe_backend_test_suite!` requires `OwnedBuf = AlignedBuf`, `HostBackend`, `HalModuleImpl` and `ModuleNew`, and the parity, blind-rotation (also `BlindRotationModSwitch`) and circuit-bootstrapping test bodies and `TestContext::new` require the factories of the prepared keys they allocate.
- Add `FheUint::{sext, splice_u8, splice_u16, get_bit_glwe, get_bit_lwe, get_byte, from_fhe_uint_prepared, zero_byte}_tmp_bytes`; the blind rotation, the BDD key encryption and several `FheUint` operations take their temporaries from scratch instead of allocating.
- `BDDKey` implements `BDDKeyInfos`.
- **Behaviour:** the blind rotations and selection, CMux, CSwap, BDD execution, `FheUint` preparation and the `FheUint` operations above panic at entry, before any write, with a static `"<method>: insufficient scratch"` on an arena below their query; `bdd_circuit_execute_tmp_bytes` and `blind_rotation_execute_tmp_bytes` round each temporary to the scratch alignment.
- **Behaviour:** key encryptions wipe their scratch, refusals on secret paths print no operand values, and `FheUint::encrypt_sk`, `decrypt` and `noise_stats` check width and degrees in every build.
- **Behaviour:** `FheUint::get_bit_lwe` narrows its input to each switching key's `max_input_k`; the circuit bootstrapping traces at the precision its automorphism keys cover and panics on a tensor key that does not cover the output GGSW.

### `poulpy-mhe`

- **Breaking:** mhe items lead with `MHE` or `mhe_`, renamed types keeping `#[deprecated]` aliases ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)): `<Thing>MHEProtocol[Impl, Reference]` becomes `MHE<Thing>Protocol[Impl, Reference]`; `{GLWEPatCompressed, GGLWEPatCompressed, GGLWEPat}Ops` become `MHE<Pat>Ops` with methods `mhe_<pat>_aggregate_assign` and `mhe_<pat>_finalize[_tmp_bytes]`; the PAT and nine share types take the prefix (`MHEGGLWEPat`, `MHEGGSWShare`, ...); `MHEModuleAlloc` splits into `MHE<Type>Alloc` with `mhe_<type>_alloc[_from_infos]`, the share allocators taking `k_digit` in place of `dsize`.
- **Breaking:** one opt-in macro per OEP trait: `impl_mhe_pat_reference!` splits into `impl_mhe_{glwe_pat_compressed, gglwe_pat_compressed, gglwe_pat}_reference_full!`, `impl_mhe_evaluation_key_reference!` into `impl_mhe_glwe_{switching_key, automorphism_key}_protocol_reference_full!`, `impl_mhe_keyswitch_reference!` into `impl_mhe_glwe_{private, public}_keyswitch_protocol_reference_full!` and `impl_mhe_sharing_reference!` into `impl_mhe_glwe_{enc_to_share, share_to_enc}_protocol_reference_full!`; `impl_mhe_{ggsw, public_key, tensor_key}_reference!` become `impl_mhe_{ggsw, glwe_public_key, glwe_tensor_key}_protocol_reference_full!`.
- **Breaking:** the `api` family modules are private: `poulpy_mhe::api::<family>::X` becomes `poulpy_mhe::api::X`.
- **Breaking:** the blanket `MHEGGSWProtocolReference` and `MHEGLWETensorKeyProtocolReference` require `BE: HalVecZnxImpl`; `MHEGLWEPublicKeyProtocolImpl` requires `HalMatZnxImpl` and `poulpy_core::oep::EncryptionImpl` instead of `MHEGLWEPatCompressedImpl`.
- **Breaking:** the 24 refusal and ten guard tests of `mhe_backend_test_suite!` are renamed `test_mhe_<op>_rejects_<case>`, guard tests split by operation (`test_pat_aggregate_seed_mismatch` becomes `test_mhe_glwe_pat_compressed_aggregate_assign_rejects_seed_mismatch`).
- **Breaking, behaviour:** `mhe_*_alloc_from_infos` panics unless `infos.n()` is the module's degree (`MHE*Alloc: ModuleN`); the keyswitch and encryption-to-shares share generations panic on a flag-clear mask.
- **Breaking, behaviour:** `mhe_ggsw_share_finalize` checks the key's `max_input_k` against the share's precision (was `dnum * k_digit`) before any write, with a new panic message.
- The GGLWE PAT and public key share aggregations and finalizations run whole-matrix `mat_znx` operations ([#380](https://github.com/poulpy-fhe/poulpy/issues/380)).
- Add `mhe_parity_test_suite!`, comparing every protocol bit for bit across two backends; the PAT types implement `poulpy_core::TransferInto`.
- **Behaviour:** share generations and finalizations refuse an arena below their `_tmp_bytes` before any draw or write; refusal messages on secret paths no longer print operand values.

### `poulpy-blas`

- New crate: backend-agnostic encrypted matrix arithmetic on `poulpy-ckks`, on the standard and conjugate-invariant rings. A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block (`BlasCiphertext`, `BlasLayout`, `BlasCiphertextAlloc`, `layouts::grid::{split_into_blocks, join_blocks}`), and every operation acts on the batch of matrices the slots hold (`BlasGridInfos::mat_slots`); a product of two batches broadcasts the smaller.
- `BlasMatMulOps` adds `blas_matmul` (the JKLS product of two encrypted grids), `blas_matmul_prepared`, `blas_matmul_pt_right` and `blas_matmul_pt_left` (by public matrices); `BlasMatIntraOps` adds `blas_transpose`, `blas_rotate_rows`, `blas_rotate_cols` and the axis reductions `blas_sum_rows` and `blas_sum_cols`; `BlasMatInterOps` adds `blas_rotate` and `blas_sum` across the batch; each has its `_tmp_bytes` query. Operations take any `BlasGrid` (`BlasGridMut` for `res`) and `Sync` key stores (rotation keys as `keys`, the tensor key as `tsk`), put operation-wide scalars before `res` (`blas_rotate(r, res, a, keys, scratch)`), and leave `res` untouched on a rejected call (`CKKSCompositionError::InvalidOperand`, a missing key or a `poulpy-ckks` budget error).
- `BlasPlan` fixes the block dimension and lists each operation's rotation keys (`transpose_galois_elements`, `matmul_galois_elements`, `sum_galois_elements`, …). A `BlasContextPrepared` holds the plan's maps (`blas_context_encode`, `blas_context_prepare`) and a `BlasPtOperandPrepared<BE, Right>` or `BlasPtOperandPrepared<BE, Left>` a public matrix (`blas_pt_operand_encode`, `blas_pt_operand_prepare`), matrix slot `s` reading matrix `s mod count`. Encodings are `BlasEncodingOps<BE, F>` at poulpy-ckks's encoding scalar `F`, preparations `BlasContextPreparedFactory` and `BlasPtOperandPreparedFactory`.
- `BlasPtAlgorithm` on `BlasPlan` picks the product by public matrices: `Diagonal` (default), `Jkls`, or `JklsStored` and `JklsGalois`, which put JKLS's masks on the ciphertext, store `d` (`2d - 1` on the left) or one plaintext per block, and take one public matrix on the right. `blas_matmul_pt_right`, `blas_matmul_pt_left` and their queries take the context.
- `blas_matmul_left_operand_prepare` and `blas_matmul_right_operand_prepare` (`BlasMatMulLeftOperandPreparedFactory`, `BlasMatMulRightOperandPreparedFactory`) make the operands `BlasMatMulLeftOperandPrepared<BE>` and `BlasMatMulRightOperandPrepared<BE>` of `blas_matmul_prepared`, hot (`PrepareHint::OneShot`) or cold (`Reuse`, every term converted once), allocated from `blas_matmul_left_operand_layout` and `blas_matmul_right_operand_layout`. Every mix of operands gives the same bits, `blas_matmul` is the prepared product of two hot operands, and each output block is one JKLS sum relinearized once.
- Add vectors, `BlasVector` (`BlasVectorOwned`, `BlasVectorShape`, `BlasVectorLayout`, `BlasVectorAlloc`): `len` entries over ciphertexts of `chunk` entries, entry `e` at slot `offset + stride e`. `BlasMatVecOps::blas_matvec_pt` multiplies a vector by a public matrix of any shape, through a resident (`BlasMatVecOperandPrepared`) or streamed (`BlasMatVecOperand`) operand with the same bits, on a `Direct` or `Bsgs` schedule (`with_strategy`, `with_optimal_bsgs`); `blas_grid_to_vector` and `blas_vector_to_grid` repack a grid's line (`BlasGridLine`) through such an operand.
- Add diagonal ciphertexts, `BlasDiagonalCiphertext` (`BlasDiagonalCiphertextPrepared`): a square matrix over one ciphertext's slots holding a chosen set of its diagonals as ciphertexts. `blas_append_row_diagonal` and `blas_append_col_diagonal` add a ciphertext to a row or a column, and `blas_matvec_diagonal` multiplies the matrix by a ciphertext, resident or streamed (`BlasDiagonalCiphertextMap`) with the same bits.
- `layouts` gives each operation its validator (`validate_*`), its key-use query (`*_key_uses`, returning `BlasKeyUses`) and its output-layout query (`*_output_layout`), and the products and repacks a radix query (`*_max_base2k`: the largest `base2k`, at most 62, that `Module::max_base2k` allows for every lazy sum, 0 if none does). A validator rejects a key that does not cover the width it is used at as a missing key.
- When a grid's label proves one matrix, `blas_rotate_rows` of one block row and the row sum need no map and no scale, and an encrypted right factor takes its rotations at no scale; `blas_matmul_pt_left` of a replicated input by a `Diagonal` operand of one public matrix takes `d`-diagonal block maps (fewer key switches, same keys).
- `blas_sum_rows` and `blas_sum_cols` require a flag-clear input block to hold at most two canonical digits per coefficient.
- `BlasCiphertext` and `BlasVector` implement `WriterTo` and `ReaderFrom`: a header, the batch record, then each block at its live limbs; a grid of mixed block layouts or with a flag-clear block is `InvalidInput`.
- Backends implement the `oep` families `BlasMatInterImpl`, `BlasMatIntraImpl` (map hook `blas_map_blocks`), `BlasMatMulImpl`, `BlasJklsImpl` (hooks `blas_jkls_block` and `blas_jkls_pt_unit`), `BlasEncodingImpl` and `BlasMatVecImpl`, whose methods have their API method's name, or select the `reference` traits (`BlasMatMulReference`, …) with `impl_blas_reference_full!` or one family at a time with `impl_blas_mat_inter_reference_full!`, `impl_blas_mat_intra_reference_full!`, `impl_blas_mat_mul_reference_full!`, `impl_blas_jkls_reference_full!`, `impl_blas_encoding_reference_full!` and `impl_blas_mat_vec_reference_full!`. Implementations validate before any write; an override fills prepared storage through `BlasMatMulOperandPrepared::{blocks, terms}`, `set_prepared` or `diagonals_mut`.
- Calls with distinct `res` and `scratch` may run concurrently over a shared module, context, operands and key stores.
- Add `blas_backend_test_suite!`, `blas_conjugate_invariant_test_suite!`, `blas_parity_test_suite!` and `blas_max_base2k_test_suite!` (`test-utils`).

### `poulpy-libm`

- New in the workspace: moves in from [poulpy-libm](https://github.com/poulpy-fhe/poulpy-libm) 0.1.0 (`5cfddab`), published on crates.io: CKKS math functions with `libm` names, built on the approximation machinery of `poulpy-ckks`. It is versioned and released with the other crates from 0.10.0.
- **Breaking:** ported to the current Poulpy API: operations take any operand implementing the backend view and CKKS info traits, tensor keys as `GetTensorKey<BE>` and rotation and conjugation keys as `GetAutomorphismKey<BE>`; `ckks_sum_slots`, `ckks_fmax_slots`, `ckks_fmin_slots` and `ckks_softmax` lose their key type parameter, and conjugating operations require `Backend<Ring = Standard>`.
- **Breaking:** `ckks_goldschmidt_division`, `ckks_rsqrt` and `ckks_inverse_{positive,negative,full}_domain` take the `_assign` suffix ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)).
- **Breaking:** each `*_tmp_bytes` query takes one layout per operand in operation order, then its plan, `coeff_prec` or `params`, then the keys (`(res, input, plan, tsk)`), and covers its whole body; `ckks_inverse_tmp_bytes` splits into `ckks_goldschmidt_division_assign_tmp_bytes` and `ckks_rsqrt_assign_tmp_bytes`, `ckks_div_sqrt_tmp_bytes` into `ckks_div_tmp_bytes` and `ckks_sqrt_tmp_bytes`; add `ckks_{comparison_const,select,sincos,ldexp,fmin_slots}_tmp_bytes`.
- **Breaking:** operation-wide scalars lead: `ckks_scalbn(res, input, n, scratch)` is `ckks_scalbn(n, res, input, scratch)`, likewise `ckks_ldexp`, `ckks_powi(e, res, input, tsk, scratch)` and `ckks_div(iters, res, a, b, tsk, scratch)`, and `iters` or `r` in `ckks_sqrt`, `ckks_goldschmidt_division_assign`, `ckks_rsqrt_assign` and the `ckks_inverse_*_domain_assign` operations ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)); plan fields are private behind accessors.
- **Breaking:** `ckks_interval_normalization(norm, ct, ..)`, which compresses `ct` in place, becomes `ckks_interval_normalization_assign(ct, norm, ..)`, with `ckks_interval_normalization_assign_tmp_bytes`; `ckks_fmax_slots` and `ckks_fmin_slots` take one automorphism-key source `keys` for their rotations and conjugation, where they took `rotation_keys` and `conj_key` ([#376](https://github.com/poulpy-fhe/poulpy/issues/376)).
- **Breaking:** the features `ref`, `avx`, `avx512`, `ifma` and `all-backends` are removed; backends register `libm_backend_test_suite!` (`test-utils`).
- **Breaking, behaviour:** operations leave `res` and in-place operands unchanged on `Err`.
- An evaluation needs the plan's consumption plus `max(1, floor(log2(M)) + 2)` bits of headroom for a largest result magnitude `M` (`docs/api.md`): a budget just above the consumption, as the 0.1.0 guide asked, could wrap the result without an error.
- Add `ckks_sum_slots_galois_elements`; errors wrap the `poulpy-ckks` error, which still downcasts to `CKKSError`.
- Benchmarks move to `poulpy-bench`, walkthroughs to the `libm_*` examples of `poulpy-cpu-portable`.

### `poulpy-gpu-portable`

- New crate: the `NTT4x30` subfamily on GPUs through CubeCL 0.11, bit-identical to `NTT4x30Portable` on both rings, as `NTT4x30Cuda`, `NTT4x30Hip` and `NTT4x30Wgpu` (`enable-cuda`, `enable-hip`, `enable-metal`, `enable-vulkan`, `enable-wgpu`), with `enable-core`, `enable-ckks` and `enable-blas`. See `docs/guides/backends.md`.
- Runs on the device, sampling included, except CKKS encoding and decoding, uniform fills of a coefficient column and normalizations above 62-bit radices. `Replay` replays a recorded call sequence as one graph; `handle` reaches a module's `GpuHandle` (`set_convolution_capacity_step`).

### `poulpy-bench`

- **Breaking:** `bench_ckks_bootstrapping` drops `FIXTURE_BASE2K` and runs each preset at the radix `preset_for_backend` selects (ids read `base2k=53` and `base2k=17` instead of 52 and 18), with EvalMod on the conjugate-invariant ring; it requires `CKKSBootstrappingCIOps`, `PresetRunModule`, `HostStaged` and `CKKSDFTOps`.
- **Breaking:** `schemes::suites::all_ops` and `bench_ckks` require `CKKSLinearTransformationOps`, `CKKSDotProductOps`, `GLWENormalize`, `GLWEFill`, `ModuleCoreAlloc`, `CnvPVecAlloc` and `ScratchOwned`; `hal::suites::bench_normalization` requires `VecZnxGadgetDecompose[TmpBytes]` and sweeps `vec_znx_gadget_decompose`; the bin-fhe runners require the factories of the prepared keys they allocate (`GGSWPreparedFactory`, `BlindRotationKeyPreparedFactory`, `GGLWEToGGSWKeyPreparedFactory`, `GLWEAutomorphismKeyPreparedFactory`).
- Add the `blas` layer (`bench_blas`, tables `blas_dense_ops` and `blas_ops`): matrix product, transpose, row and column sum, row rotation, matrix-vector and grid-to-vector runners in `blas::{matmul,intra,matvec}`, and `blas::scale`, whose `max_base2k_table` and `validate` check `poulpy-blas` at large blocks (`d = 128`, `n = 2^15` and `2^16`).
- Add the `libm` layer (`runner_ckks_{exp,log,pow}`, `libm_ops`, `bench_libm`), `runner_glwe_automorphism_many[_composed]`, `ckks_linear_transformation_ops`, `bench_ckks_bootstrapping[_replayed]_with_client` and `heap::Counting`.
- **Behaviour:** the bootstrapping runners also run each preset with power-of-two rotation keys only, suffixed `+pow2`, and every timed closure ends with `Module::synchronize`.
- The HAL, core and CKKS runners no longer bound `OwnedBuf = AlignedBuf` or host views, except `bench_ckks_bootstrapping`.

### Dependencies

- Lock `chacha20`, a dependency of `rand`, at 0.10.2, as 0.10.0 is yanked.
- `poulpy-gpu-portable` depends on `cubecl` and `cubecl-environment`, pinned at 0.11.3, and on `pollster` 0.4.0.
- The lock bumps `cc` from 1.2.57 to 1.9.0, `bitflags` from 2.11.0 to 2.13.2 and `wasm-bindgen` from 0.2.99 to 0.2.129, with the CubeCL dependencies.

## [0.9.0] - 2026-10-08

Reworks the HAL around a smaller operation basis with documented contracts and backend-generic derived defaults ([#234](https://github.com/poulpy-fhe/poulpy/issues/234)), adds the conjugate-invariant ring, multi-degree modules and compact operands ([#266](https://github.com/poulpy-fhe/poulpy/issues/266)), moves sampling into `poulpy-core` behind a unified `Noise` with per-component noise metadata ([#372](https://github.com/poulpy-fhe/poulpy/issues/372)), and makes CKKS encoding byte identical across backends and targets. `poulpy-cpu-ref` is renamed `poulpy-cpu-portable`; new crates: `poulpy-mhe` (multiparty HE) and the unpublished `poulpy-cpu-oracle`.

### `poulpy-hal`

- **Breaking:** api traits, methods and OEP methods drop the `_backend` and `_into` suffixes (`VecZnxAddIntoBackend::vec_znx_add_into_backend` is `VecZnxAdd::vec_znx_add`); `vmp_apply_dft_to_dft_accumulate` is `vmp_apply_dft_to_dft_add`, `cnv_apply_dft_accumulate` is `cnv_apply_dft_add`, `cnv_accumulate_dft` is `cnv_apply_dft_sum`.
- **Breaking:** `HalModuleImpl`, `HalVecZnxImpl`, `HalVecZnxBigImpl`, `HalVecZnxDftImpl`, `HalSvpImpl`, `HalVmpImpl` and `HalConvolutionImpl` lose their backend type parameter and gain supertraits; `vec_znx_rotate*` and `vec_znx_mul_xp_minus_one*` move to the new `HalVecZnxMonomialImpl`.
- **Breaking:** removed operations with their OEP methods: `vec_znx_add_const*`, `vec_znx_add_scalar_into`, `vec_znx_sub_scalar*`, `vec_znx_automorphism_rotate`, `vec_znx_split_ring`, `vec_znx_merge_rings`, `vec_znx_transpose`, `vec_znx_hadamard_product_scalar_znx`, `scalar_znx_automorphism_assign`, the `*_coeff` normalize and shift variants, `vec_znx_copy_range` and `vec_znx_extract_coeff` (use window views), `vec_znx_big_from_bytes*`, `vec_znx_dft_from_bytes`, `vec_znx_big_alloc_n`, `ModuleNew::new_with`, `HalModuleImpl::Config`, the `cnv_*_lazy` forms and `vec_znx_dft_add_scaled_assign`.
- **Breaking:** every sampler except `vec_znx_fill_uniform_source` is removed (`scalar_znx_fill_*`, `vec_znx_fill_normal*`, `vec_znx_add_normal*`, `vec_znx_big_add_normal*`, the seeded `[u8; 32]` forms) with `FillUniform`; `ScalarZnx::fill_*` moves to `poulpy_cpu_portable::ScalarZnxFill` and noise to `poulpy-core`.
- **Breaking:** `Backend::Ring` (`Standard` or `ConjugateInvariant`) selects the ring at compile time; `HostBytesBackend<R: Ring = Standard>`; `impl_backend_from!` accepts `; generic R: Ring`.
- **Breaking:** a `Module` of degree `N` serves every power-of-two degree from `Backend::MIN_DEGREE` to `N`: `*_alloc`, `bytes_of_*`, `take_*_scratch` and `vec_znx_dft_automorphism_plan` take `n` first, the `_n` variants merge into the unsuffixed forms, and construction panics above `layouts::MAX_RING_DEGREE` (`2^17`).
- **Breaking:** `SvpPPol`, `VmpPMat`, `CnvPVecL` and `CnvPVecR` carry a `PrepareHint`, taken by `bytes_of_*` (including `Backend::bytes_of_*`), `*_alloc`, `take_*_scratch` and `from_data`; `SvpPPol::shape()` returns `SvpPPolShape`.
- **Breaking:** `vec_znx_idft_normalize_consume` takes `res_offset`; `vec_znx_normalize_assign` takes `a_offset`; `vec_znx_lsh_tmp_bytes`, `vec_znx_rsh_tmp_bytes` and `vec_znx_mul_xp_minus_one_assign_tmp_bytes` take the destination size; `svp_apply_dft` and `vec_znx_dft_automorphism_add_with_plan` take scratch (`svp_apply_dft_tmp_bytes`, `vec_znx_dft_automorphism_add_with_plan_tmp_bytes`); add `cnv_apply_dft_add_tmp_bytes` and `cnv_by_const_apply_add_tmp_bytes`.
- **Breaking:** `cnv_prepare_left`, `cnv_prepare_right` and `cnv_prepare_self` drop `mask`.
- **Breaking:** `vec_znx_lsh`, `vec_znx_rsh` and their `_add`/`_sub`/`_assign` forms run as `vec_znx_normalize` with an offset: a shorter destination rounds the dropped tail jointly, and `rsh` past the destination precision yields zero.
- **Breaking:** `poulpy_hal::reference::znx` moves to `poulpy_cpu_portable::kernels::znx`; `vec_znx_host_backend_ref` and `vec_znx_host_backend_mut` are removed.
- **Breaking:** `PrimeSet` requires `LOG_Q_PRODUCT`; add `PrimeSet::validate` and `PrimeSet::MAX_LOG_N`.
- **Breaking:** `Backend::DFT_IS_EXACT` is removed; partial `VecZnxDft` limb-range views and indexed zeroing require `Backend::DFT_LIMBS_CONTIGUOUS`.
- **Breaking:** owned host storage is `AlignedBuf` (`AlignedVec<u8>`): `alloc_aligned` and `alloc_aligned_custom` return `AlignedVec<T>`, layout `from_bytes` takes `impl Into<AlignedBuf>`, and `assert_alignment`/`assert_alignement` are removed.
- **Breaking:** `test_suite`: cross-backend tests no longer take a host module, `TestBackend` requires `HalVecZnxImpl`, `TestParams` carries the operand degree `n`, `serialization::test_reader_writer_interface` takes `[T; 2]`, `test_vec_znx_copy` is `test_vec_znx_dft_copy`.
- Derived operations have backend-generic default bodies on the OEP traits (`oep::derived`), so a backend implements only the basis; add `Backend::scratch_aligned`.
- Every api trait documents a structured contract (`op`, `class`, `mutation`, `definition`, `domain`, `ensures`, `test`, `sparse`).
- Add window views on `VecZnx` and `VecZnxBig` (`window_coeffs`, `window_limbs`, `from_shape`, `with_shape`), accepted by the coefficient-wise operations.
- Add sparse operands ([#266](https://github.com/poulpy-fhe/poulpy/issues/266)): the slots a contract's `sparse` line names accept a degree-`n` operand standing for its ring embedding.
- Add `VecZnxCIEmbed`, `VecZnxCITrace` and `ScalarZnxCIEmbed` behind `HalVecZnxCIImpl`.
- Add `Module::max_base2k` (`MaxBase2k`, [#312](https://github.com/poulpy-fhe/poulpy/issues/312)), `VecZnxFillUniformSourceAll`, `ModuleSynchronize`, `ScratchArena::wipe` and `Ring::product_moments`.
- `test_suite` adds the `derived`, `window`, `sparse`, `transfer` and `reim` modules.
- **Fix:** `vec_znx_rsh_assign` read an uninitialized carry at `k == 0` and underflowed when the shift exceeded the limb count.
- **Fix:** NTT `vec_znx_dft_apply` and `vec_znx_dft_copy` accepted `step == 0`.
- **Fix:** `Backend::from_host_bytes` panicked on lengths that are not a multiple of 64 bytes; `copy_view_to_host` and `copy_host_to_view` accept a host slice shorter than the view.
- **Fix, soundness:** `alloc_aligned` freed its allocation with a layout different from the allocating one.

### `poulpy-core`

- **Breaking:** the `default` module is `reference` ([#234](https://github.com/poulpy-fhe/poulpy/issues/234)): reference algorithms are `*Reference` traits and `*_reference` functions; the `*Default` override traits and `impl_*_defaults_full!` macros are removed, and backends implement the `poulpy_core::oep` `*Impl` traits explicitly or through `impl_*_reference_full!`, `impl_*_derived_full!` and `impl_core_reference_full!`.
- **Breaking:** the `oep` traits lose their backend type parameter; the GGSW methods of `ConversionImpl` move to `GGSWConversionImpl`; `impl_core_reference_full!` excludes `SamplingImpl` and the standard-only families (LWE conversion, rotation, `mul_xp_minus_one`, trace, packing).
- **Breaking:** the `enable-avx`, `enable-avx512f`, `enable-ifma` and `enable-neon` features are removed.
- **Breaking:** owned layouts, keys and compressed forms are `X<AlignedBuf, W>` instead of `X<Vec<u8>, W>`.
- **Breaking:** `Noise` (`Gaussian { sigma }`, `Uniform { bits }`, `Noise::ENCRYPTION`) replaces `EncryptionInfos`, `EncryptionLayout` and the HAL `NoiseInfos`; encryptions and key generations drop `enc_infos` ([#372](https://github.com/poulpy-fhe/poulpy/issues/372)).
- **Breaking:** secrets and noise are sampled on the backend through the new `SamplingImpl` family (`ScalarZnxFillDistribution`, `VecZnxAddNoise`, `VecZnxBigAddNoise`); fixed-seed secrets, keys and ciphertexts change.
- **Breaking:** ciphertexts and keys carry `ComponentNoise` metadata (`LWEInfos::noise`, required `set_noise` on the `*ToBackendMut` traits): encryptions record it, evaluation clears it, equality compares it.
- **Breaking, format:** serialized ciphertexts and keys start with a `PNM3` noise-metadata prefix; `Distribution` serializes as a tag byte and a `u64` payload.
- **Breaking:** `GLWE` carries a canonical flag ([#319](https://github.com/poulpy-fhe/poulpy/issues/319); `is_canonical`, required `GLWEToBackendMut::set_canonical`): operations reading digits through a DFT normalize a flag-clear operand first, and `GLWE::write_to` rejects a flag-clear GLWE.
- **Breaking:** `glwe_lsh`, `glwe_lsh_assign` and `glwe_rsh` round at the destination's `k` instead of its allocated width.
- **Breaking:** `glwe_copy`, `glwe_prepare`, `glwe_public_key_prepare` and `glwe_public_key_generate` take scratch (new `*_tmp_bytes`); `glwe_copy` rounds narrowing copies at the destination's `k` and converts radices.
- **Breaking:** `glwe_shift_tmp_bytes` takes the destination size, `ggsw_from_gglwe_tmp_bytes` the source layout and `glwe_encrypt_pk_tmp_bytes` the public-key layout; add `glwe_trace_assign_tmp_bytes` and `glwe_mul_xp_minus_one_assign_tmp_bytes` ([#385](https://github.com/poulpy-fhe/poulpy/issues/385)); `take_glwe_secret_prepared_scratch` takes the secret degree.
- **Breaking:** a rank-`r` `GLWEPublicKey` is `r` encryptions of zero stored as a one-row GGLWE (new byte format, `GLWEPublicKeyToBackendRef`/`Mut` views); public-key encryption draws `r` ephemerals, returns a canonical ciphertext and panics on a key less precise than the output.
- **Breaking:** `glwe_tensor_apply_prepared_right` and its `_tmp_bytes` become `GLWETensoring` methods.
- **Breaking:** linear-transformation diagonals and `glwe_mul_plain` right operands are prepared at their own degree; `Diagonals` stores each diagonal as one period (`period`, `period_lcm`).
- **Breaking:** `GGLWE::noise` and `GGSW::noise` are renamed `noise_stats`.
- **Breaking:** removed `GLWESwitchingKeyEncryptPk`, `GLWEAutomorphismKeyEncryptPk`, `GLWEPlaintext::alloc_with_meta` and `msb_mask_bottom_limb`; `fill_glwe_mask_from_source` and `fill_glwe_mask_from_seed` drop `base2k`, `res_col` and `rank`.
- **Breaking:** test suites: add `core_encryption_parity_test_suite!`; `core_parity_test_suite!` accepts `test_size`; `keys::fill_by_digit` takes the module and any GGLWE-shaped key.
- **Behaviour:** gadget products keep fewer guard limbs, so key switching and external products are not bit identical to 0.8.3.
- `GiantStepTensorBounds` is deprecated.
- Operations accept operands of any degree up to the module degree.
- Reference encryptions, key generations and decryptions zero their scratch before returning.
- Add `glwe_mul_relinearize`, `glwe_square_relinearize`, `glwe_mul_prepared_relinearize` and their `_assign` forms to `GLWETensoring`.
- Add `GLWEEncryptPk::glwe_encrypt_pk_at_col`, `GGSWEncryptPk`, `GLWEEncryptPkSmudged`, `GLWEEncryptSk::glwe_encrypt_sk_with_mask` and `GLWECompressedEncryptSk::glwe_compressed_encrypt_zero_sk`.
- Add `GLWEPublicKeyCompressed` (`GLWEPublicKeyCompressedGenerate`, `GLWEPublicKeyDecompress`), `GLWEMask` and `GLWEMaskInnerProduct`, and `GLWEMaskFill::fill_glwe_from_source`.
- Add `public_key_encryption_noise`, `public_key_sample_precision` and `PublicKeyBodyNoise`.
- Add the conjugate-invariant maps `GLWECIEmbed`, `GLWECITrace` and `GLWESecretCIEmbed` and the switching keys `GLWECIEmbedKey` and `GLWECITraceKey` (`GLWECIKeyEncryptSk`).
- Add `Polynomial::chebyshev_interpolate_with_cos`, `PreparedDiagonal::real_slots` and `LinearTransformationBabySteps::baby_steps_mut`.
- **Fix:** `glwe_prepare_linear_transformation_rhs` dropped the limbs of a diagonal stored wider than its `k`; it now panics ([#386](https://github.com/poulpy-fhe/poulpy/issues/386)).
- **Fix:** `glwe_lsh_add` and `glwe_lsh_sub` read absent columns of a lower-rank operand; `glwe_sub_negate_assign` with a rank-zero operand did not negate the mask.
- **Fix:** `ggsw_keyswitch` failed on a destination with fewer rows than the source.
- **Fix:** an out-of-place `glwe_trace` with nothing to trace did not copy its input.
- **Fix:** `lwe_encrypt_sk_tmp_bytes` could undercount.

### `poulpy-ckks`

- **Breaking:** the normalization typestate is removed ([#319](https://github.com/poulpy-fhe/poulpy/issues/319)): `Normalized`, `Unnormalized`, `CKKSNormalizationState`, `UnnormalizedCKKSCiphertext`, `UnnormalizedCKKSCiphertextRefMut`, `take_unnormalized_ckks_ciphertext{,_like}_scratch`, the `ckks_{add,sub}_*_unnormalized` methods and `ckks_mul_add_pt_{const,vec}_into_unnormalized`.
- **Breaking, behaviour:** additions and subtractions no longer normalize: they clear the GLWE canonical flag and the next operation reading the operand through a DFT normalizes it; `CKKSPolynomialEvaluationImpl` requires `GLWENormalizeImpl`.
- **Breaking:** `CKKSCiphertext<D, W, R>` and `CKKSPlaintext<D, W, R>` take the ring as a type parameter; a module accepts only operands of its backend's `Ring`.
- **Breaking:** the `default` module is `reference` ([#234](https://github.com/poulpy-fhe/poulpy/issues/234)): `*Default` traits, `*_default` methods and `impl_ckks_*_defaults!` are `*Reference`, `*_reference` and `impl_ckks_*_reference!`.
- **Breaking:** the `oep` traits lose their backend type parameter (`CKKSEncodingImpl<F>` and `DFTMatrixImpl<F>` keep the scalar); the blanket `*Impl` implementations, `DFTDefault` and `DFTMatrixDefault` are removed, so backends register every family (new `impl_ckks_eval_mod_reference!`, `impl_ckks_polynomial_evaluation_reference!`, `impl_ckks_bootstrapping_reference!`).
- **Breaking:** `CKKSBootstrappingOps` dispatches through `CKKSBootstrappingImpl`; `DFTImpl`, `CKKSEvalModImpl`, `CKKSPow2Impl`, `CKKSRotateImpl` (now requiring `CKKSCopyImpl`) and `CKKSConjugateImpl` gain methods.
- **Breaking:** the `ckks_*_tmp_bytes` queries of add, sub, `add_many`, `neg`, `mul_pow2`, `div_pow2`, `mul_i`, `div_i`, `mod_up` and `extract_pt` take the destination size; `ckks_copy_tmp_bytes` and `ckks_decrypt_tmp_bytes` take both layouts; add `ckks_add_one_tmp_bytes` and `ckks_sub_one_tmp_bytes`.
- **Breaking:** `ckks_encrypt_sk` drops `enc_infos`; encryption and key generation use `Noise::ENCRYPTION` and ciphertexts carry `ComponentNoise` metadata ([#372](https://github.com/poulpy-fhe/poulpy/issues/372)).
- **Breaking, behaviour:** `CKKSScalar` requires the new `numerics::CKKSFloat`, as do `minimax*`, `degree_for_precision*`, `precision_at_depth*`, `sign_composite_coeffs*` and `DftScalar`; setup math and quantization return the same bits on every target and can differ in the last bit from 0.8.3, so cached bootstrapping parameters must be rebuilt.
- **Breaking:** rotations reduce the shift modulo the slot count, so a negative shift rotates backwards on the standard ring and needs the key of `ckks_galois_element(k)`.
- **Breaking:** complex-coefficient polynomial evaluation moves to `CKKSComplexPolynomialEvaluationOps` (`impl_ckks_complex_polynomial_evaluation_reference!`).
- **Breaking:** conjugation, multiplication by `i`, DFT, EvalMod, encapsulated ModUp, complex polynomial evaluation, SHIP, PaCo and `BootstrappingContext::generate_keys` require `Backend<Ring = Standard>`.
- **Breaking:** encoded plaintexts require `SetCKKSInfos` and encoding narrows the slot kind to the ring's (`SlotsKind::meet`); `CKKSEncodingImpl` gains `ckks_encode_slots_assign_into_impl` and `ckks_decode_slots_into_impl`.
- **Breaking:** linear transformations propagate the slot kind and reject mixed or misreported diagonals; `LtDiagonalScale` is `LtDiagonalMeta`.
- **Breaking:** `EncodedLut::general` takes exactly `p >= 2` entries without power-of-two padding; LUTs in a batch share their length.
- **Breaking:** test utilities: `TestContextHostModule` is removed; `test_bootstrapping_*_e2e` take only their parameters; `preset_for_backend` is `preset_with_max_base2k`.
- Add conjugate-invariant CKKS (`N` real slots): encoding, leveled operations, rotations, real polynomial evaluation and linear transformations; `CKKSModuleInfos`.
- Add compact plaintexts ([#266](https://github.com/poulpy-fhe/poulpy/issues/266)): `ckks_pt_vec_alloc_compact`, accepted by `ckks_{add,sub,mul}_pt_vec*`; sparse linear transformations and DFT factors store compact diagonals.
- Add `CKKSFoldOps` and `CKKSFoldLayoutOps` (`impl_ckks_fold_reference!`): fold standard or conjugate-invariant ciphertexts into a smaller bootstrapping degree with `RingSwitchKeys`, and unfold them.
- Add `CKKSCIRingMapOps` (`ckks_ci_embed`, `ckks_ci_trace`).
- Add `BootstrappingPlan::with_functional_bootstrap`, `BootstrappingContext::compile_unprepared` and `BootstrappingContextUnprepared::prepare`.
- Add `CKKSFloat::ckks_root_of_unity`, `ckks_quantize` and `ckks_dequantize`, `ckks_double_into`, `ckks_prepare_dft_matrix_tmp_bytes`, `ckks_dft_tmp_bytes` and checked `DFTMatrix` factor access (`factor_operands`, `try_from_factor_operands`, `try_with_factor_operands`).
- Add `ckks_parity_test_suite!`, `ckks_encryption_parity_test_suite!`, `conjugate_invariant_ckks_test_suite!` and `test_suite::determinism` (`POULPY_UPDATE_FIXTURES=1` refreshes the fixtures).
- `ckks_encrypt_sk` and `ckks_decrypt` return `EncryptionDegreeMismatch` on mismatched degrees.
- Rank-1 multiplication and squaring use the one-pass `GLWETensoring` products when the tensor key matches the operands' radix.
- **Fix:** `ckks_encrypt_sk` copies the plaintext's sparsity and slot kind to the ciphertext.
- **Fix:** coefficient encoding rejects non-finite and overflowing inputs, including `Quad` conversions that saturated.

### `poulpy-bin-fhe`

- **Breaking:** operation traits are implemented for `Module<BE>` only when `BE` implements the `poulpy_bin_fhe::oep` `*Impl` traits, selected with `impl_bin_fhe_reference_full!` or the per-operation `impl_bin_fhe_*_reference!` macros; the CPU backends do this behind `enable-bin-fhe`.
- **Breaking:** `BDDEncryptionInfos`, `CircuitBootstrappingEncryptionInfos` and the `enc_infos` arguments are removed; components encrypt with `Noise::ENCRYPTION` ([#372](https://github.com/poulpy-fhe/poulpy/issues/372)).
- **Breaking:** `glwe_blind_rotation_tmp_bytes` and `ggsw_to_ggsw_blind_rotation_tmp_bytes` take the source infos, with separate `glwe_blind_rotation_assign_tmp_bytes` and `ggsw_blind_rotation_assign_tmp_bytes`; `glwe_blind_selection_tmp_bytes` takes the infos of every input.
- **Breaking:** `BlindRotationKeyCompressedFactory` is implemented by the module (`blind_rotation_key_compressed_alloc`).
- **Breaking:** `FheUint::noise` and `FheUintPreparedDebug::noise` are renamed `noise_stats`; `FheUint::from_glwe_to_mut` and `FheUint::from_glwe_to_ref` are removed.
- Operation traits live in `poulpy_bin_fhe::api` (re-exported at their former paths); the reference algorithms are in `poulpy_bin_fhe::reference`.
- Remove the `'static` bounds on `UnsignedInteger`, backends and key providers, and the host-backend bounds of `BDDKeyEncryptSk`, `FheUintPreparedEncryptSk` and `BDDKey::encrypt_sk`.
- Add `BlindRotationKeyDecompress`, `BlindRotationModSwitch`, `execute_bdd_circuit_1w_to_1w_tmp_bytes`, `execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes`, `fhe_uint_prepared_encrypt_sk_tmp_bytes`, `GLWEBlindRetriever::add_tmp_bytes` and `GLWEBlindRetriever::flush_tmp_bytes`.
- `CircuitBootstrappingExecute` gains `circuit_bootstrapping_prepare_to_constant`, `circuit_bootstrapping_prepare_to_exponent` and `circuit_bootstrapping_execute_prepared` over `CircuitBootstrappingPlanLayout`.
- Blind-rotation keys, `BDDKey`, `BDDKeyPrepared`, circuit-bootstrapping keys and `LookupTable` expose their parts for backend overrides.
- Add `bin_fhe_parity_test_suite!` (`test_suite::parity`) and `bin_fhe_reference_test_suite!`.
- **Fix:** `mod_switch_2n` truncated multi-limb LWE coefficients instead of rounding them.
- **Fix:** re-preparing a `BlindRotationKeyPrepared` kept a stale block-binary monomial table.
- **Fix:** `prepare_bdd_key_tmp_bytes` omitted the GLWE switching key, and the standard CGGI `blind_rotation_execute_tmp_bytes` omitted `glwe_mul_xp_minus_one_assign`.
- **Fix:** `GLWEBlindRetriever` failed on capacities 0 and 1 and did not enforce its exact capacity.

### `poulpy-mhe`

- New crate: backend-agnostic multiparty homomorphic encryption over public aggregatable transcripts (`GLWEPatCompressed`, `GGLWEPatCompressed`, `GGLWEPat`, allocated through `MHEModuleAlloc`), selected with `impl_mhe_reference_full!` and tested with `mhe_backend_test_suite!`; `poulpy-cpu-portable` wires it behind `enable-mhe`.
- Collective keys under the ideal secret: `GLWEPublicKeyMHEProtocol`, `GLWESwitchingKeyMHEProtocol`, `GLWEAutomorphismKeyMHEProtocol`, `GLWETensorKeyMHEProtocol` and `GGSWMHEProtocol`.
- Collective key switching (`GLWEPrivateKeyswitchMHEProtocol`, `GLWEPublicKeyswitchMHEProtocol`) and conversions between ciphertexts and additive shares (`GLWEEncToShareMHEProtocol`, `GLWEShareToEncMHEProtocol`), flooded with `Noise`.
- Finalized collective keys record the base secret distribution and party count in their noise metadata.

### CPU backends

- **Breaking:** `poulpy-cpu-ref` is renamed `poulpy-cpu-portable`: `FFT64Ref`, `NTT4x30Ref` and their handles are `FFT64Portable`, `NTT4x30Portable`, `FFT64PortableHandle` and `NTT4x30PortableHandle` (deprecated aliases keep `FFT64Ref` and `NTT4x30Ref`); `capabilities::reference_backends` is `capabilities::portable_backends`.
- **Breaking:** the `reference` module of `poulpy-cpu-portable` is `kernels`; `ZnxRef`, `ReimFFTRef` and `ReimIFFTRef` are `ZnxPortable`, `ReimFFTPortable` and `ReimIFFTPortable`; exported kernel functions take the `_portable` suffix (`znx_add_ref` is `znx_add_portable`).
- **Breaking:** every CPU backend and handle takes a ring parameter `R: Ring = Standard` (expression paths read `<FFT64Portable>::..`); `FFTHandleProvider` and `NttHandleProvider` declare `type Ring`.
- **Breaking:** per-ring kernels live in `kernels::{fft64, ntt4x30, znx}::{standard, conjugate_invariant}`; FFT64 kernels take `&FFT64Plan` and NTT4x30 kernels `&Module<BE>`; FFT64 backends implement `Fft64RingArith` per ring (`fft64_ring_arith_standard!`, `fft64_ring_arith_ci!`); plans are built through `FFT64PlanNew` and `NttPlanNew`.
- **Breaking:** `NttDFTExecute` gains `ntt_automorphism_plan`, `ZnxAutomorphism` gains `znx_automorphism_i128`, `ReimArith` gains `reim_automorphism` and `reim_automorphism_add`; `impl_fft64_rayon_backend!` drops its automorphism argument.
- **Breaking:** `FFTHandleProvider::{get_fft_table, get_ifft_table}` and `NttHandleProvider::{get_ntt_table, get_intt_table}` are removed for `get_*_table_for` the operand degree; backends serve every degree from `Backend::MIN_DEGREE` (8; 16 on `FFT64Avx512`) to the module degree.
- **Breaking:** NTT4x30 and NTT3x42 use new primes and roots supporting ring degrees up to `2^17`; regenerate raw NTT and prepared data.
- **Breaking:** `NTT4x30Neon` and `NTT4x30NeonRayon` run a native 32-bit NTT with `DftWord = CrtWord<Primes30, u32>` and lose their DFT and SVP layout compatibility with `NTT4x30Portable`.
- **Breaking:** `hal_defaults` and the `hal_impl_*!` macros keep only the basis and the fused overrides (standard-only monomial operations in `hal_impl_vec_znx_monomial!`); the kernels of operations the HAL derives or removed are deleted from `kernels::vec_znx`.
- **Breaking:** `FFT64ReimTable`, `FFT64AvxReimTable`, `FFT64Avx512ReimTable` and `FFT64NeonReimTable` are deprecated aliases of `ckks_encoding::EncodingFFTTable`, `FFT64AvxEncodingTable`, `FFT64Avx512EncodingTable` and `FFT64NeonEncodingTable` (`enable-ckks` only); `ckks_encoding::{CpuEncodingPlan, encode_coeffs_into, decode_coeffs_into, slots_to_coeffs_assign, coeffs_to_slots_assign}` are removed.
- **Breaking:** binary FHE on a CPU backend requires the new `enable-bin-fhe` feature; `poulpy-cpu-avx512` depends on `poulpy-cpu-avx`.
- **Breaking:** CPU backends sample `Noise` through `impl_sampling_host!` with exact integer samplers; seeded noise streams change ([#372](https://github.com/poulpy-fhe/poulpy/issues/372)).
- **Behaviour:** CKKS encoding is byte identical across CPU backends; NEON ring transforms and the small sizes AVX2 and AVX-512 delegate to portable kernels can differ in the last bit.
- Add `poulpy-cpu-oracle` (unpublished): `FFT64Oracle`, `NTT4x30Oracle`, `FFT64CIOracle` and `NTT4x30CIOracle`, independent scalar backends that the cross-backend and parity suites compare against.
- Add the conjugate-invariant backends `FFT64CIPortable`, `NTT4x30CIPortable`, `FFT64CIAvx`, `NTT4x30CIAvx`, `FFT64CIAvx512`, `NTT4x30CIAvx512`, `NTT3x42CIIfma`, `FFT64CINeon`, `NTT4x30CINeon` and their `*Rayon` forms; standard backends implement the ring maps and the CKKS fold.
- Add `impl_cpu_core_defaults!`, `impl_cpu_ckks_defaults!`, `ScalarZnxFill`, `ReimFFTTable::new_with_roots`, `ReimIFFTTable::new_with_roots` and `kernels::fft64::reim::{fft_portable_fused, ifft_portable_fused}`.
- Add `conjugate_invariant_test_suite!`, `conjugate_invariant_core_test_suite!` and, behind `enable-test-suite`, `ControlledSamplingFFT64Portable` and `ControlledSamplingFFT64CIPortable`.
- Add `I128NormalizePlan` (`apply_unchecked`, `base`, `active`, `limb_offset`, `boundary`) and the `*_unchecked` methods of `I128NormalizeOps`.
- Add `poulpy_cpu_rayon::{set_idle_polling, idle_polling, DEFAULT_IDLE_POLLING}`; `RayonTaskExecutor` budgets nested regions by available threads instead of depth.
- Faster NEON backends (fused `cnv_apply_dft_sum` and strided digit products, VMP, convolution, CRT, automorphism, normalization), faster `cnv_by_const_apply` on NTT backends, and AVX-512/IFMA rank-one tensor specializations from degree `2^13`.
- **Fix:** the FFT64 `vec_znx_dft_apply` left output limbs whose source lies past the input untouched instead of zeroing them.
- **Fix, soundness:** the AVX2, AVX-512 and NEON FFT/IFFT butterflies and `reim_to_znx_i64_assign_bnd63_*` invalidated their own pointers under Stacked Borrows.

### `poulpy-bench`

- **Breaking:** `core::fill` is removed; core runners sample on the tested backend and require `GLWEMaskFill` (blind-rotation runners also `LWEFillMask`).
- **Breaking:** `bench_ckks_bootstrapping` takes a `FIXTURE_BASE2K` const generic.
- **Breaking:** HAL runners and bench ids follow the HAL rename (`runner_vec_znx_add_into` is `runner_vec_znx_add`, `runner_cnv_apply_dft_accumulate` is `runner_cnv_apply_dft_add`).
- Add HAL runners for the shift `_add`/`_sub` forms, `vec_znx_add_scalar_assign`, `vec_znx_idft_normalize_consume`, `vmp_apply_dft_to_dft_add`, `cnv_prepare_self`, `cnv_by_const_apply_add` and `cnv_apply_dft_sum`.

### Dependencies

- Bump `dashu-float` and `dashu-int` to 0.6.2, `rand` to 0.10.3, `rayon` to 1.12.0, `num-traits` to 0.2.19, `libc` to 0.2.190 and `f128` to 0.2.9.
- `poulpy-ckks` depends on `libm` (without default features) and `astro-float-num` on every target, not only on targets without a native binary128 libm.
- `poulpy-core` and `poulpy-cpu-portable` depend on `dashu-int`; `poulpy-cpu-avx512` depends on `libc` with `enable-rayon`.
- Remove the unused `cmake` build dependency of `poulpy-hal`.

## [0.8.3] - 2026-09-09

Adds opt-in Rayon parallelism to every accelerated CPU family behind a backend-selected task executor, packs NTT4x30 transform words into `u32` pairs on AVX2/AVX-512, fuses paired gadget digits in strided key switching, and resolves evaluation keys per precision, including reading a prepared key at a coarser digit size. Normalization takes an explicit target precision, fixing cross-base and partial-limb noise. CKKS gains `LogN=16` bootstrapping presets, non-power-of-two LUTs, an even Han–Ki EvalMod and a native NTT ModUp; bin-FHE runs on device backends.

### `poulpy-hal`

- **Breaking:** `Backend` gains the `TaskExecutor` associated type. The new `execution` module provides `TaskExecutor` (`join`, `for_each`, `for_each_chunked`), `SerialTaskExecutor`, `ScratchWorkers` and the `worker_count`, `scratch_workers`, `worker_scratch_bytes` and `for_each_with_scratch` helpers, so default algorithms schedule independent work over per-worker scratch with `*_tmp_bytes` independent of the pool width.
- **Breaking:** the open extension points gain `cnv_by_const_apply_add`, `vec_znx_dft_automorphism_add_with_plan`, `vec_znx_idft_normalize_consume` (with `_tmp_bytes`) and `vmp_extract_selected_rows`, exposed through the API as methods on the existing traits plus `VecZnxIdftNormalizeConsume` and `VmpExtractSelectedRows`; every CPU backend implements them.
- **Breaking:** `vec_znx_normalize*` and `vec_znx_big_normalize*` take the target precision `res_k` and round once at it.
- **Breaking:** `vec_znx_fill_uniform*` take `k`, masking the unused low bits of the last live limb and clearing the limbs above it.
- Fix cross-`base2k` normalization leaving limbs outside `[-2^(base2k-1), 2^(base2k-1))` in `VecZnx` and `VecZnxBig` ([#256](https://github.com/poulpy-fhe/poulpy/issues/256)).
- Kernels check source and carry slice lengths with `assert` instead of `debug_assert`.
- **Breaking:** `NoiseInfos::target_limb_and_scale` and `target_limb_and_shift` are removed; add `vec_znx_alloc_zeroed`.

### `poulpy-core`

- **Breaking:** every operation consuming an evaluation key takes its backend view, written `&key.to_backend_ref()`: `&GGLWEPreparedBackendRef`, `&GLWEAutomorphismKeyPreparedBackendRef`, `&GGSWPreparedBackendRef` or `&GGLWEToGGSWKeyPreparedBackendRef`.
- **Breaking:** `GLWEAutomorphismKeyHelper` and `automorphism_key_infos()` are removed, with the key type parameter of automorphism-consuming signatures. Add `GetAutomorphismKey` and `GetTensorKey`: a source answers a function and the precision it is used at with a prepared-key view; a map or a bare key implement them as stored. Linear transformations resolve baby-step keys at the source precision and giant-step keys at the post-product precision.
- `GGLWEInfos` gains `stride` (digit `i` is stored at row `(i + 1) * stride - 1`), `GGLWELayout` gains `at_dsize`, `dnum_for_input` and `gadget_k`, and `with_dsize` re-tags a prepared GGLWE, automorphism or tensor key at a coarser `dsize`; a product over that view gathers the selected rows once.
- Apply the gadget-product limb window on every backend. It was gated on `DFT_IS_EXACT` because of the FFT64 VMP bug fixed below, not a property of approximate transforms ([#222](https://github.com/poulpy-fhe/poulpy/issues/222)).
- Fix precision loss at non-`base2k`-aligned ciphertext widths: masks and Gaussian noise are sampled at `k` bits for secret- and public-key encryption.
- Fix cross-radix tensor relinearization sizing the DFT operand from the storage precision instead of `a.k()`.
- Fix noise in the first slots of linear transformations at non-limb-aligned precisions: outputs are normalized at their declared precision.
- GLWE key switching consumes inverse-DFT results directly into normalized outputs, lazy and prepared-giant linear transformations use DFT automorphism accumulation, and generic polynomial evaluation gains an optional one-pass baby-step linear-combination hook.
- Add `TransferInto` for `GLWEToLWEKey` and `GGLWEToGGSWKey`; `GGSWPrepared::data` no longer requires host data.

### `poulpy-ckks`

- Add `presets::bootstrapping` with the `LogN=16` bundles `n16_d35_k600_p19_c2s` and `n16_d35_k720_p19_s2c` (plan, widths, secret weights and key layouts), re-derivable at another radix or digit size with `with_base2k` and `with_dsizes`. See `docs/bootstrapping.md`.
- `EncodedLut::general` accepts any nonempty table length, zero-padded to the next power of two; LUTs with the same padded length share a batch.
- Add `EvalModType::CosHKEven`, a centred Han–Ki approximation folded through `T₂` when it lowers multiplication cost within the same modulus budget.
- **Breaking:** `BootstrappingPlan::new` validates the EvalMod plan and derives the CoeffsToSlots input scaling from it.
- **Breaking:** `BootstrappingPlan::bootstrap_k(output_k, log_delta)` takes the input/output scale; C2S-first bootstrapping restores the input scale at the end, including EvalRound+. Add `with_c2s_guard_bits` for S2C-first plans.
- **Breaking:** the ciphertext-ciphertext operations (`ckks_mul_*`, `ckks_square_*`, `mul_add`/`mul_sub`/`dot_product`, polynomial evaluation, approximation, EvalMod and the PaCo slot product) take a tensor-key source resolved at their working precision. `CKKSAtkBounds` is replaced by the `GetAutomorphismKey<BE>` bound, `CKKSCompositionError::MissingAutomorphismKey` carries the lookup precision and `MissingRelinearizationKey` is added.
- **Breaking:** `ckks_coeffs_to_slots_{split,repack}` drop `conj_key`; `BootstrappingKeys::conjugation_key` and its `AutomorphismKey` type are removed and generation puts the conjugation key in `rotation_keys`. Add `ckks_conjugate_rotate_into(dst, src, k, ..)` for PaCo's fused conjugate-and-rotate; `PaCoPsiTailMaterial::Mask` carries the rotation instead of the Galois element.
- Fuse each BSGS baby-step linear combination into one accumulator, run the real and imaginary EvalMod halves concurrently on parallel backends, and size the `ckks_mul_tmp_bytes` tensor intermediate from the operands' precision rather than `res`'s.

### `poulpy-bin-fhe`

- Device-resident backends can prepare CGGI keys and execute BDD circuits and every blind-rotation variant, with explicit host staging.
- FHE integer preparation, BDD evaluation and block-binary CGGI blind rotation run through the backend executor with disjoint scratch arenas; serial backends keep one worker.
- **Breaking:** multithreaded BDD execution requires its output buffer to implement `Send`.

### CPU backends

- Add `poulpy-cpu-rayon` (shared executor, nested-parallelism guard, thresholds, FFT64 and normalization kernels) and, behind `enable-rayon`, the `FFT64AvxRayon`, `NTT4x30AvxRayon`, `FFT64Avx512Rayon`, `NTT4x30Avx512Rayon`, `NTT3x42IfmaRayon`, `FFT64NeonRayon` and `NTT4x30NeonRayon` backends, layout-compatible with their serial families and wired into HAL, core, CKKS and bin-FHE. Transform, convolution, VMP and normalization kernels parallelize within one operation; nested Rayon serializes its inner level. `docs/performance.md` covers backend selection and thread tuning.
- Pack the four NTT4x30 transform-domain residues into `u32` words on AVX2 and AVX-512, halving DFT and prepared-key storage from 32 to 16 bytes per coefficient.
- Fuse adjacent gadget digits in strided VMP key-switch kernels for AVX2 NTT4x30, AVX-512 NTT4x30 and AVX-512-IFMA NTT3x42.
- Canonicalize rank-one AVX-512/IFMA tensor products after diagonal subtraction, including partial precision and Rayon paths.
- The AVX2 and AVX-512 NTT backends and their Rayon variants override the encapsulated CKKS ModUp with a native kernel.
- Speed up normalization with precision-boundary and carry kernels on AVX2, AVX-512 and NEON, and add `cargo bench --bench normalize` to `poulpy-cpu-ref`, `poulpy-cpu-avx` and `poulpy-cpu-avx512`. `I128NormalizeOps` now requires the new `I64NormalizeOps`.
- Fix the FFT64 `vmp_apply_dft_to_dft` limb window when `res` is narrower than the prepared matrix, which dropped the top `limb_offset` limbs of every narrowed gadget digit on every FFT64 backend ([#222](https://github.com/poulpy-fhe/poulpy/issues/222)). Fix the NTT4x30 reference `vmp_apply_dft_to_dft_accumulate_tmp_bytes` under-reporting scratch when `res` is wider than the matrix.
- Add an NTT4x30-versus-FFT64 parity suite to `poulpy-cpu-ref`, and sweep `res` sizes and `limb_offset` in the cross-backend VMP tests.

### `poulpy-bench`

- Replace the bootstrapping benchmark with `ckks_bootstrapping`, driven by the CKKS presets.

## [0.8.2] - 2026-08-22

### `poulpy-hal`

- Add `reference::znx`, the portable scalar kernels over `[i64]`, moved out of `poulpy-cpu-ref` so that a crate can reach them without depending on a backend. `poulpy_cpu_ref::reference::znx` re-exports them, so existing paths are unchanged.

### `poulpy-bin-fhe`

- **Breaking:** the crate no longer depends on any backend, in `dependencies` or `dev-dependencies`, matching `poulpy-core` and `poulpy-ckks`. The `enable-avx`, `enable-avx512f`, `enable-ifma` and `enable-neon` features are removed (`enable-neon` was inert: nothing referenced `poulpy-cpu-arm`), and `enable-bin-fhe` no longer enables anything.
- **Breaking:** the per-backend test modules are replaced by public `blind_rotation::test_suite`, `circuit_bootstrapping::test_suite` and `bdd_arithmetic::test_suite`, instantiated by a backend crate through the new `bin_fhe_backend_test_suite!`. The `bdd_arithmetic` / `circuit_bootstrapping` / `max_array` examples move to `poulpy-cpu-ref` and run on `FFT64Ref`.

### `poulpy-core`

- Add the missing `GGLWEPreparedToBackendRef` impl for `GLWETensorKeyPrepared`, the twin of the `Mut` one. The newtype field and every `GGLWEPrepared` field are `pub(crate)`, so no downstream crate could supply it. The relinearization path (`GLWETensoringImpl::glwe_tensor_relinearize` and the `poulpy-ckks` `mul` / `polynomial_evaluation` / `composite` chains that reach it) carries the bound alongside `GLWETensorKeyPreparedToBackendRef`, so a backend override can read the key's `VmpPMat` without naming the newtype field.

### `poulpy-ckks`

- ModUp lifts `Δ·m` to `k - log_msg_ratio`, then to `f_mod_log_delta` fused into the widening shift, both before the sparse-to-dense switch. Recovers `f_mod_log_delta - k` bits (19.7 → 26.9 at `LogN=16`, `Δ=2^40`, ratio `2^8`). C2S-first only.
- **Breaking:** `ckks_mod_up_into` takes `&EvalModPlan` and returns a labelled ciphertext; new `ckks_bootstrap_mod_up` does the whole raise step; `CKKSEncapsulatedModUpImpl::ckks_encapsulated_mod_up` takes `scale_up`.

### `poulpy-bin-fhe`

- **Breaking:** drop the `enable-avx` / `enable-avx512f` / `enable-ifma` / `enable-neon` features and the backend dependencies; only `enable-bin-fhe` remains. The tests are backend-generic and instantiated by backend crates through `bin_fhe_backend_test_suite!`; the examples move to `poulpy-cpu-ref/examples`.

### CPU backends

- `mod tests` is gated on the backend feature alone, not on `enable-ckks`, so the core-parity and bin-FHE suites run without it.

## [0.8.1] - 2026-08-20

- Update `dashu-float` to 0.6 and `astro-float-num` to 0.3.7, and refresh the `bytemuck`, `serde`, `serde_json` and `anyhow` pins. The dashu 0.6 context operations return a `Result`, which the internal binary128 helpers now unwrap; no public API changes.
- Publish from CI through crates.io trusted publishing (`.github/workflows/release.yml`, triggered by a `v*` tag). The internal workspace dependencies carry a `version` requirement so `cargo publish` can package them.

## [0.8.0] - 2026-08-20

Adds two native CKKS bootstrapping families, **PaCo** (Coron & Seuré, [ePrint 2025/886](https://eprint.iacr.org/2025/886)) and **SHIP** (Cheon, Hanrot, Kim & Stehlé, [ePrint 2025/784](https://eprint.iacr.org/2025/784)), moves CKKS encoding onto a backend-resident op family, and lands a production-readiness pass over `poulpy-ckks` (typed errors, constructor-validated plans, scratch-carved intermediates, binary128-exact tables, four-layer consolidation). Keys every polynomial layout by an explicit word type (`ZnxWord` / `BigWord` / `DftWord`) and the DFT/big containers by their backend, collapses the limb-width model to a claimed precision plus an allocation, and reparameterizes evaluation keys by an auxiliary guard `k_aux`. Trims gadget products to their live limbs on exact-DFT backends, opens the strided GGLWE product and tensoring to backend overrides, and lands a round of AVX2/AVX-512 NTT, VMP and convolution optimization. Opens the layouts, key containers and benchmarks to non-host device backends, and adds a reference-vs-backend byte-parity test suite.

### `poulpy-hal`: word-keyed layouts

- Add `layouts::word` with the word traits `ZnxWord` (coefficient domain), `BigWord` (accumulator) and `DftWord` (prepared/DFT). A word names a byte-layout convention and governs sizing, element views, and serialization interpretation.
- **Breaking:** `Backend::ScalarPrep` → `DftWord` and `ScalarBig` → `BigWord` (with `size_of_scalar_prep`/`size_of_scalar_big` → `size_of_dft_word`/`size_of_big_word`), and every backend declares a coefficient word `type ZnxWord`.
- **Breaking:** the DFT/big layouts are re-keyed by word and backend: `VecZnxDft<D, W: DftWord, B>`, `VecZnxBig<D, W: BigWord, B>`, `SvpPPol`, `VmpPMat`, `CnvPVecL/R`. Containers of different backends are distinct types, so cross-backend interchange is a compile error. The `*Owned` / `*BackendRef` / `*BackendMut` aliases absorb the change.
- `VecZnx` / `ScalarZnx` / `MatZnx` gain a defaulted word parameter; `ReaderFrom`/`WriterTo` become word-generic over the unchanged wire format.
- **Breaking:** backend-bound constructors spell the backend through the owned alias (`VecZnxDftOwned::<B>::alloc`), and the DFT→big re-tag is `x.into_big()`.
- `x.to_backend_ref()` / `x.to_backend_mut()` infer the backend everywhere: no turbofish.
- Add `layouts::layout_compat`: per-container `unsafe` markers a backend pair declares to assert byte-identical layouts, unlocking the zero-copy `x.into_backend::<B2>()`. Validated by the new `test_suite::word_compat`.
- `ZnxView::raw()`/`at()` assert the element view fits the buffer instead of reading out of bounds.
- **Breaking:** `Backend` requires `PartialEq + Eq`. The DFT/big containers implement `PartialEq`/`Eq` on the representation.

### `poulpy-cpu-ref` / CPU backends: prime-set-parameterized NTT words

- **Breaking:** `PrimeSet` is lane-generic (`type PrimeElem`, GAT `type Lanes<T>`), and it, `LaneElem`, `LaneArray` and `CrtWord` move to `poulpy_hal::layouts::crt` (re-exported from `ntt4x30` for compatibility). CRT reconstruction constants move to the family extensions `PrimeSetCrt4` and `PrimeSetNtt3x42Ifma`.
- **Breaking:** `Q120bScalar` becomes an alias of the new block type `CrtWord<P: PrimeSet, T: LaneElem>`; tuple construction must spell `CrtWord([…])`. `Q126Scalar` stays a sizing-only `DftWord`.
- All CPU backends declare `type ZnxWord = i64`, and every backend marker derives `PartialEq, Eq, Hash`.
- Each accelerated crate declares `layout_compat` markers against its reference sibling (FFT64 and NTT4x30 families), validated by `word_compat`. `VmpPMat` and `CnvPVec` markers are intentionally absent; `NTT3x42Ifma` pairs with nobody.
- `poulpy-cpu-arm`: the NEON ntt4x30 modules are repaired and a native aarch64 `neon` CI job checks, lints, and tests the crate.

### `poulpy-bench`

- All 19 `core` runners drop `OwnedBuf = Vec<u8>`, the host-view bounds and `HostBackend`, so a device backend can run them. Non-measured operands are built on a host staging module and transferred in (`core::fill`) rather than encrypted, so the tested backend needs no sampling kernel.
- Fixed: the operation and tensor runners timed on zeroed buffers, which can hit float-FFT denormals.

### `poulpy-core`

- The prepared layouts follow the HAL re-key transparently; their derived `Eq` is dropped in favor of `PartialEq`.
- **Breaking:** remove `GGLWEPreparedVmpPMatRef`; use the inherent `GGLWEPrepared::data()`, which is generic over the buffer.
- Remove 47 dead `#[allow(private_bounds)]` from `oep`.
- **Breaking:** remove the exported `impl_glwe_rotate_impl_from!`; the blanket `GLWERotateImpl` impl supersedes it.
- **Breaking:** `GLWEKeyswitchInternal` and `GGLWEProductDefault` become public; they appear in the bounds of the public `glwe_keyswitch*_default` functions.
- The `*Default` override surfaces are no longer `#[doc(hidden)]`; `oep`'s module docs describe the `*Impl` / `*Default` split with a worked override.
- The gadget-digit width rule moves to the `GLWEKeyswitchDefault` contract and shared `gglwe_product_digit_output_size` helper; `GGLWEProductDefault` passes its accumulation count through `GGLWEProductDigitsStridedImpl` as a shape-derived `product_limbs` spill window instead of a fixed two limbs.
- Gadget products on an exact-DFT backend trim their working width to the limbs that can still affect the result, through the single `gadget_product_output_size`. It takes both operand precisions, so a cross-radix product is sized from its own shape. Approximate-DFT backends keep the full `key_work_size` width.
- **Breaking:** remove `GGLWEInfos::work_size` and `GGSWInfos::work_size`; the width is derived by `gadget_product_output_size` from the operand precisions, which `work_size` did not see.
- CKKS encapsulated ModUp composes ordinary dense-to-sparse and sparse-to-dense key switches around ModUp; backends can override `CKKSEncapsulatedModUpImpl` to fuse it.
- Relax to `D: Data`: `GGLWE` / `GLWEPlaintext` / `LWEPlaintext` / `GLWETensor` `data()` / `data_mut()`, `GLWESwitchingKeyDegrees(Mut)`, `Get`/`SetGaloisElement`, `GetDistribution(Mut)`, `SetBase2k`.
- **Breaking:** `GGLWEAtBackendRef`/`Mut` and `GGSWAtBackendRef`/`Mut` become public; `GLWESwitchingKey`, `GLWEAutomorphismKey` and `GLWETensorKey` delegate them.
- **Breaking:** remove `ModuleTransfer` and the inherent `Layout::to_backend` methods (29 in total) in favour of `api::TransferInto`, which writes into a destination the caller allocates: `src.transfer_into(&mut dst)`. Checks shape, not just byte length.
- **Breaking:** `test_suite` splits into `test_suite::noise` (the existing scheme-correctness suite, moved wholesale) and `test_suite::parity`. `core_backend_test_suite!` is unchanged.
- Add `BackendGLWESwitchingKey<BE>` / `BackendGLWEAutomorphismKey<BE>`.
- Add `test_suite::parity` and `core_parity_test_suite!`: runs one operation on a reference and a tested backend over identical uniform inputs and asserts byte equality. Covers key-switch (GLWE, assign, GGLWE), the strided GGLWE product hook, automorphism, external product, tensoring and the other keyless GLWE operations. Needs no secrets, encryption or noise model. An optional `shapes = ParityShapes { .. }` restricts the rank and `dsize` sweep for a backend with a narrower envelope.
- `poulpy-cpu-avx`, `-avx512` and `-arm` run the parity suite against their `poulpy-cpu-ref` sibling, for the FFT64 and NTT4x30 families.

### `poulpy-hal`

- **Breaking:** `Backend` gains `len_bytes_ref` / `len_bytes_mut`.
- **Breaking:** replace `TransferFrom` with the buffer traits `CopyToHost` / `CopyFromHost` and the free `transfer_buf_into(src, dst)`; the ~44 hand-written per-backend-pair impls are removed. `HostStaged` becomes `Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost>`.
- Add `layouts::vec_znx_backend_ref` / `vec_znx_backend_mut` / `vec_znx_reborrow_backend_mut`; `test_suite` re-exports them.
- Add `HalModuleImpl::Config` (defaulting to `()`) and `new_with(n, config)`, mirrored on `ModuleNew`, for device selection at construction. Requires `#![feature(associated_type_defaults)]`.
- Add `ModulePlanCache`, a per-`Module` typed cache of immutable plan families with a `with_or_create` accessor, plus `unsafe trait ModulePlanCacheProvider` for backend handles that own it.
- **Breaking:** `Backend` gains two required methods, `copy_view_to_host` and `copy_host_to_view`.
- **Breaking:** remove the public `set_size` / `with_size` resizing API from `VecZnx`, `VecZnxDft` and `VecZnxBig`; temporary compute widths are scoped views.
- **Breaking:** remove `max_size` from the layout family. `size` is both the working and the allocated width, fixed at construction, and narrowing is exclusively a borrowed view (`with_size_mut`, and the new `vec_znx_backend_mut_with_size`). Removed: `VecZnxShape::max_size`, the inherent `max_size()`, `from_data_with_max_size`, `vec_znx_alloc_with_max_size`; `VecZnxShape::new` loses its fourth argument. The `VecZnx` wire format drops one `u64` field.
- **Breaking:** split `ZnxInfos` into a shape trait plus `VecZnxInfos` (adding `cols()`) and `MatZnxInfos` (adding `rows()`, `cols_in()`, `cols_out()`); `poly_count()` becomes required. `ZnxView` now requires `VecZnxInfos` and the matrix containers expose inherent `raw()` / `raw_mut()` instead.
- Re-document `Module` as a multi-ring execution context: `n()` is the maximum ring degree, with the new alias `max_n()`.
- Add `Backend::DFT_IS_EXACT` (default `false`), declaring that the DFT round-trip is exact rather than floating-point. Consumers use it to decide whether a working width may be trimmed.
- Add `VecZnxDft::with_limb_range_mut(start, end)`, a borrowed view of a limb window, replacing manual `region_mut` / `from_data` carving.

### `poulpy-core`

- **Breaking:** the limb-width model collapses to two quantities, the claimed precision `k()` (with derived `size()`) and the allocation `max_size()`/`max_k()`. `LWEInfos::max_size()` now means the allocation everywhere. The `Compact` and `SetSize` traits are removed: narrowing a result means allocating the destination at the `k` you want. Plaintext operands are the one typed exception, declaring their consuming width through the new `IntPolyInfos::encoded_k()`, which every plaintext-consuming op now bounds.
- Fixed, uncovered by the width-model collapse: the inherent `GLWE::max_size()` shadowed the `LWEInfos` impl with different semantics; `GLWETensor` reported its allocation where siblings reported the stored width; the linear-transformation scratch budget read the stored width while claiming capacity; `SetSize::set_size` wrote a width `size()` did not read back.
- `LWEInfos::max_size()` is read off the payload for buffer-backed types; the two `LWE::validate_shape` checks asserting `size <= max_size` are dropped as no longer representable.
- Add `GLWESecret::fill_binary_coeffs(col, coeffs)`, installing a caller-provided binary coefficient vector and tagging `BinaryFixed(hamming_weight)`, for structured secrets.
- Add `GLWEPlaintextReborrowBackendRef` / `GLWEPlaintextReborrowBackendMut`.
- Export the `view_wrapper!` and `impl_glwe_infos!` macros for downstream nominal view types.

### `poulpy-core`: evaluation-key parameterization (`k` → `k_aux`)

- **Breaking:** every gadget-key layout stores an auxiliary guard `k_aux` in place of the total precision `k`, derived as `k() = dnum·dsize·base2k + k_aux` and enforcing `k_aux ≥ dsize·base2k`. A former `Dnum(d)` key with an implicit zero guard migrates to `Dnum(d−1)` with `k_aux = dsize·base2k`.
- **Breaking:** the gadget-key operations (`glwe_keyswitch`, `glwe_external_product`, `glwe_automorphism`, `glwe_tensor_relinearize`, and their compositions) no longer take an output/working size argument; it is derived from the input's `k` and the key's `(dsize, k_aux)`.
- **Breaking:** key allocation and byte-sizing signatures move to a uniform `(…, dnum, dsize, k_aux, rank…)` order.
- The noise model follows: `GGLWENoiseModel` / `GGSWNoiseModel`, blanket-implemented over the layouts, take the operand and the error variances only (`ksk.log2_std_noise_keyswitch(&ct_in, …)`). Evaluated at the key's real precision, the bounds gain the operand's carried error and the uncovered decomposition residue.

### `poulpy-ckks`: reusable approximation planning
- Add the `approximation` module with host-side single- and multi-interval Remez minimax fitting, precision/depth-based degree selection, and composite sign coefficient generation. Add `PolynomialApproximation` and `CKKSApproximationOps` for reusable interval mapping and prepared BSGS evaluation, including exact power-of-two scale and modulus-consumption accounting.

### `poulpy-ckks`: PaCo bootstrapping

- Add PaCo as a native CKKS operation refreshing selected polynomial coefficients without ModUp or EvalMod: blind rotation → partial CoeffsToSlots → product fold (the EvalMod substitute) → SlotToCoeff′, driven by a structured low-weight secret.
- Evaluation is `api::CKKSPaCoOps` on `Module<BE>`, all outputs caller-allocated: `ckks_paco_bootstrap_direct_into` (input already under the PaCo secret), `ckks_paco_bootstrap_into` (dense→PaCo switch first), the bounded-parallel variants, the `_tmp_bytes` queries, and `ckks_paco_coeff_encodings`.
- The direct entry points produce a leveled output: the caller allocates at any `k ≤ PaCoContext::max_output_k`, and the branch runs the whole circuit at the correspondingly narrower width.
- Invariant-bearing data lives in `layouts`: `PaCoPlan` and `PaCoDFTPlan`, `PaCoContext::compile` (plaintext material only), `PaCoSecretSpec`, `PaCoKeySet` → `PaCoKeysPrepared`, the `PaCoKeys<BE>` access trait, and `PaCoKeyParameters` fingerprinting the key-defining dimensions independently of any schedule.
- Parallel evaluation uses the caller plus a borrowed slice of reusable `PaCoWorker` contexts under `std::thread::scope`; branches recombine in sequential order, workers are validated before any output mutation, and an empty slice degenerates to the sequential path.
- The only PaCo-specific backend hook is coefficient encoding, via `oep::CKKSPaCoCoeffEncodingImpl<BE>`. The host reference `encoding::paco_coeff_encodings_host` is public and `poulpy-cpu-ref` exports `impl_ckks_paco_coeff_encoding!`, which every CPU backend invokes.
- `PaCoSlotOrder` (`Natural` or `BitRevLow`) selects the mid-pipeline slot convention: same budget and recovered coefficients, different Galois key set, so persisted bundles must pin both schedule and slot order.
- Scalar precision is sealed behind `api::PaCoScalar` (`f64`, `Quad`); compilation rejects `log_q ≥ F::MANTISSA_BITS`. The DFT-convention divergence from the paper's reference implementation is specified in `docs/spec/paco_dft_convention.md`.
- Backend-generic tests cover plan rejection, secret packing, coefficient encoding, the folds, both partial transforms, direct and encapsulated bootstraps, ordered parallel recombination, and output metadata, against an independent cleartext oracle.
- Security note (`docs/paco.md`): PaCo assumes a non-standard structured sparse secret; encapsulation keeps the application key dense but does not remove that assumption on the bootstrap key material.

### `poulpy-ckks`: SHIP bootstrapping

- Add SHIP as a native CKKS half bootstrap refreshing a one-limb bottom ciphertext into a slots-domain ciphertext without ModUp or EvalMod: dense→sparse encapsulation, then per support slot theta-column masking and hoisted base-B mux blind rotations, closed by a binary product tree over the `h + 1` factors and one conjugation.
- Evaluation is `api::CKKSShipOps<BE, F>`: `ckks_ship_bootstrap_into`, `ckks_ship_bootstrap_complex_into` (requires the `omega_2` mask set), `ckks_ship_bootstrap_tmp_bytes`, and `ckks_ship_coeff_encodings`. The scalar is sealed behind `api::ShipScalar`.
- Invariant-bearing data lives in `layouts`: `ShipPlan`, `ShipSecretSpec`, `ShipKeySet` → `ShipKeysPrepared`, fingerprinted by `ShipKeyParameters`. Key generation is integrated (`ShipKeySet::generate`): the mux keys are non-standard rank-2 → 1 switching keys no core keygen produces.
- The only SHIP-specific backend hook is coefficient encoding, via `oep::CKKSShipCoeffEncodingImpl<BE>`, with the public host reference `encoding::ship_coeff_encodings_host` and the `impl_ckks_ship_coeff_encoding!` macro.
- Backend-generic tests cover a cleartext replica of the paper's Algorithm 1, the HMuxRot primitive, and end-to-end real and complex bootstraps on every CPU backend at `f64` and `Quad`.

### `poulpy-ckks`: backend-resident encoding

- **Breaking:** the host-side `Encoder<T>` (and `encoding::reim`) is replaced by `api::CKKSEncodingOps<BE>` on `Module<BE>`: `ckks_encode_coeffs_into` / `ckks_decode_coeffs_into` and `ckks_slots_to_coeffs_assign` / `ckks_coeffs_to_slots_assign`, plus the provided `ckks_encode_slots_assign_into` / `ckks_decode_slots_into`.
- Add `CKKSEncodingHostOps` with the host-slice adapters, and the scalar-generic backend seam `oep::CKKSEncodingImpl<BE, F>` over the new `api::CKKSEncodingScalar`, whose per-precision plans resolve through the HAL plan cache so `f64` and `Quad` coexist on one module.
- Add the `CKKSEncodingBuffer<D, F>` layout family: a backend-resident planar `[re|im]` workspace with views, scratch carving, and host transfer helpers.
- **Breaking:** `CKKSPlaintextVecHostCodec` drops its separate sparse codec; `encode_host_floats` / `decode_host_floats` are the unified stride-aware path driven by `CKKSMeta::log_sparsity`.
- **Breaking:** `EvalMod::from_literal` is replaced by the free generic `compile_eval_mod` / `compile_eval_mod_exp`, which encode on any backend.

### `poulpy-core` / `poulpy-ckks`: polynomial evaluation
- Add explicit parity folding for real and complex BSGS polynomials through `Polynomial::fold_parity`, `Polynomial::decompose_bsgs_folded_with`, and `EncodeBSGS::encode_bsgs_folded_with`: even/odd monomial polynomials become `Q(x²)` / `x·Q(x²)`, while even/odd Chebyshev polynomials become `Q(T₂(x))` / `x·Q(T₂(x))`. The encoded degree is halved, the input transform is carried by `BSGSPolynomial`, and depth/budget accounting includes the initial square or `T₂` evaluation and the odd final multiplication. Folding remains opt-in because it can increase multiplicative depth for some degree/split-strategy combinations.
- Avoid EvalMod's scratch-input → owned-input copy for identity base polynomials by transferring the relabelled owned input directly into the shared power basis. Plans without an inverse no longer reserve a full scratch ciphertext for that copy; inverse plans retain only their post-composition working copy.

### `poulpy-ckks`: hardening, typed errors, and plan validation
- **Breaking:** every fallible op-trait method across all four layers now returns `CKKSResult<T>` (= `Result<T, CKKSError>`) instead of `anyhow::Result<T>`. `CKKSError` is `#[non_exhaustive]` with `Composition(CKKSCompositionError)` and `Internal(anyhow::Error)` variants, a `composition()` accessor for recoverable conditions, and a downcasting `From<anyhow::Error>` bridge; `CKKSCompositionError` is now `#[non_exhaustive]` and gains `InvalidPlan` and `PreparedOperandLayoutMismatch`. Callers using `?` into `anyhow` keep working; explicit `anyhow::Result` signatures must be updated.
- **Breaking:** naming sweep on the api layer: `CKKSEncrypt` → `CKKSEncryptOps`, `CKKSDecrypt` → `CKKSDecryptOps`, `DFTOps` → `CKKSDFTOps`, aligning every op family on the `CKKS*Ops` convention.
- **Breaking:** `ckks_encrypt_sk` (and `BootstrappingContext::generate_keys`) now take the randomness sources in the order `(source_xe, source_xa)` — a positional swap of the two `&mut Source` parameters; call sites must be updated by hand since the types are identical.
- **Breaking:** `ckks_decrypt` now decrypts into the **destination plaintext's preset** `(log_delta, log_budget)` frame instead of stamping the ciphertext's metadata onto it, so a caller can extract at a different precision; preset `pt.set_meta(ct.meta())` for the previous behavior. Frame mismatches surface as typed `PlaintextAlignmentImpossible` / `PlaintextBase2KMismatch` errors.
- **Breaking:** the CKKS multiplications now produce the result at the **destination's requested `dst.k()`** — rounding the low bits value-preservingly when `k` is narrower than the natural product — instead of filling the destination to its limb-aligned buffer capacity (`dst.max_k()`). This is uniform across `ct × ct` (`ckks_mul_*` / `ckks_square_*`, `get_mul_ct_params`), `ct × pt` (`ckks_mul_pt_vec_*`, `get_mul_pt_params`), the prepared `ckks_mul_prepared_assign`, and the diagonal `ct × pt` inside linear transformations. Allocate the destination at exactly the `k` you want the product at; a max-width destination is unchanged. This is what lets a leveled consumer (the PaCo blind rotation) evaluate its whole downstream circuit at a lower, cheaper torus width, and it fixes value corruption when the reduction target is non-`base2k`-aligned under a wide scale (e.g. `Quad`'s `log_delta > base2k`). The `CKKSMulOps` metadata rule is documented against `dst.k()`.
- **Breaking:** `DFTPlan` is now constructor-validated: fields are private, plans are built with `DFTPlan::new(kind, schedule, format, meta)` + `with_scaling` (rejects non-finite/non-positive values) + `with_bit_reversed`, the per-factor schedule is the new `FactorSchedule` of `FactorStep { depth, giant_step }` (making the parallel-array length invariant unrepresentable), the `meta` field is renamed `coeffs_meta` and carries the new lightweight `CoeffsMeta` (`k` + `CKKSMeta`) instead of a full `CKKSLayout`, `DFTPlan::check` is removed, and the schedule-introspection queries (`galois_elements`, `diagonal_indexes`, `num_diagonals`) are infallible instead of panicking on malformed plans.
- Add `DFTPlan::with_optimal_bsgs(log_n)` to derive structure-aware, rotation-count-balanced BSGS widths for every DFT factor without changing factor depth or modulus consumption.
- **Breaking:** `BootstrappingPlan` is a constructor-validated recipe: `new` requires the pipeline and explicit techniques, checks stage directions and sparse-secret weight, and enforces EvalRound+'s power-of-two `f_mod_interval` constraint. Sparse-secret encapsulation is recipe-owned; `EncapsulationKeysLayout` now contains only the two switching-key shapes, and key generation/execution reject layouts or key bundles whose presence disagrees with the recipe.
- Add S2C-first (slim) CKKS bootstrapping as `BootstrappingPipeline::S2CFirst`: the standard `ckks_bootstrap` orchestrator evaluates `SlotsToCoeffs → ModUp → CoeffsToSlots → EvalMod` with scratch-carved intermediates and the shared ModUp, DFT, and EvalMod blocks. `BootstrappingPlan::{pre_mod_up_consumed_bits, post_mod_up_consumed_bits, input_k, bootstrap_k}` place the bottom S2C cost below ModUp and the C2S plus EvalMod cost at the top of the raised modulus. `ckks_bootstrap_real` skips the imaginary nonlinear branch for known-real slots. Backend-generic end-to-end tests cover both slot domains, output precision and metadata, and the typed insufficient-input-budget error.
- Add CKKS functional bootstrapping (`ckks_functional_bootstrap` and the shared-power-basis `ckks_functional_bootstrap_multi`) on the standard S2C-first `BootstrappingPlan` / `BootstrappingContext`. The implementation reuses slim bootstrapping's S2C, ModUp, C2S, key generation, scratch layout, and recipe budget accounting, replacing EvalMod with encoded general or binary LUT evaluation; `ckks_functional_bootstrap_real` skips the imaginary LUT branch for known-real slots. LUT arity automatically determines `log_msg_ratio`, `functional_bootstrap_k` accounts for general and binary LUTs without charging binary evaluation for an unused EvalMod, and `ckks_functional_bootstrap_tmp_bytes` includes the LUT coefficient layout. Backend-generic tests cover the general, multi-LUT, binary, real-slot, mismatched-ratio, invalid-context, layout, scratch-sizing, and insufficient-input paths.
- Support EvalRound+ with S2C-first CKKS bootstrapping: after the low-modulus SlotsToCoeffs and ModUp, the orchestrator evaluates the low- and high-precision CoeffsToSlots branches and cancels the raised integer and low-precision transform error.
- Bootstrapping, eval_mod, and the linear-transformation wrappers now carve every working ciphertext from scratch instead of heap-allocating (up to seven owned ciphertexts per bootstrap previously); the new `ckks_bootstrap_tmp_bytes` sizing entry point is wired through all four layers and charges the full pipeline, and `ckks_eval_mod_tmp_bytes` charges its working copy. `ensure_uniform_diagonal_scale` rejects hand-built linear transformations with heterogeneous diagonal scales.
- Generalize DFT-matrix and eval-mod table generation over the scalar via the new `DftScalar` trait: per-factor scale roots use `nth_root_scalar` (f64 seed + Newton–Raphson refinement in `F`), the 256-bit CosHK solve lands through the mantissa-exact `fbig_to_scalar` triple-double decomposition, and `approximate_cos` is scalar-generic — so `Quad` matrices are binary128-exact instead of f64-rounded. `Quad` is now `bytemuck::Pod`, with new full-mantissa parity tests.
- Make `Quad`'s complete libm-backed `Float` surface portable on targets without a native binary128 libm: algebraic operations use pure-Rust `libm`, while transcendentals use guarded arbitrary-precision evaluation through `astro-float-num` followed by one round-to-nearest-even conversion to binary128. The `f128`/libquadmath dependency and its parity tests are now positively gated to x86_64 Linux GNU, fixing Intel macOS builds without changing `Quad`'s public type or ABI; the optional `libquadmath` feature remains the accelerated backing on that supported target and is a no-op elsewhere.
- **Breaking:** the `crate::leveled::*` backwards-compatibility re-export shim is removed; use the canonical `api::` / `oep::` / `delegates::` / `default::` / `layouts::` paths.
- **Breaking:** the backend-conformance `test_suite` module is now gated behind the new `test-utils` feature (backends enable it from dev-dependencies; production dependents no longer compile ~17k lines of test code), and the dead `enable-ckks` / `enable-avx` / `enable-neon` feature flags are removed from `poulpy-ckks` itself (backend crates keep their own opt-in features). Dependencies: `rand` / `rand_distr` dropped, `num-traits` and `paste` added.
- The bootstrapping conformance test now asserts a measured precision floor (`MIN_AVG_LOG2_PREC = 24.0` bits, ~4 bits under the observed 27.5–28.3) on every configuration, and the retired host reim encoder survives only as the test-suite reference oracle (`test_suite/reference_encoder.rs`).
- **Breaking:** with the `poulpy-core` two-quantity collapse, CKKS ciphertexts no longer compact: the per-op `dst.compact()` calls, the compacted-destination re-expansion in `ckks_mul_pt_vec`, and the `Compact`/`SetSize` impls and bounds are all removed — a ciphertext's buffer stays at its allocated width for its whole life, and `max_k()` uniformly means the allocation. In its place the **effective-`k` width rule is uniform across the ciphertext-side API**: the ct×ct tensor intermediates (`ckks_mul_*`/`ckks_square_*`/`ckks_mul_add_*`) are carved at the operands' effective `max(a.k, b.k)` (matching the `cnv_offset` rule, which was already expressed on effective `k`), the add/sub/copy/neg/rotate/pow2/composite alignment offsets are computed against the destination's requested `dst.k()` (previously the buffer's limb-aligned allocation), and `ckks_mod_up_into` widens to `dst.k()` — everything narrows by allocating (or relabeling) the destination at the `k` you want, and the api-doc offset formulas are updated accordingly. Plaintext operands are the one *typed* exception: they are integer polynomials, not Torus elements, and every plaintext-consuming path (ct×pt masking and `cnv_offset`, codec digit alignment, LT diagonal-scale uniformity, carry-verb constants, the encoding ops) reads their `IntPolyInfos::encoded_k()` — the declared width of the encoded integer — with the operand bounds requiring the trait across the api/oep/delegates/default layers. Masking an integer polynomial at its `k` (which labels claimed precision only) would destroy data limbs; the trait makes that width a stated property instead of a `max_k` convention.
- Fixed, uncovered by the width-model collapse: the ct×ct tensor intermediate was sized from the physical width while its own `cnv_offset` rule used effective `k` (masked by compaction; surfaced as a measured 12-15% EvalMod cost once compaction was removed, until the sizing was aligned); `validate_backend_storage_capacity` compared the key's reported stored width against its backend view's allocation and passed only because keys were full-width; and the add/sub/copy/rotate/pow2 families aligned results to the destination's allocation while the mul family used the requested `dst.k()`. Measured on the standard bootstrapping pipeline, every stage's precision is bit-identical to the compacting implementation (C2S snr 44.65/46.29, EvalMod snr 27.09/28.11, bootstrap avg 27.56/28.05) at unchanged EvalMod timing.

### `poulpy-ckks`: PaCo branch count derived from the input

- **Breaking:** `ckks_paco_bootstrap{,_direct,_parallel,_parallel_direct}_into` no longer take `kappa`; it is derived as `N/(C*2^log_sparsity)`, the branch count that refreshes every live coefficient of the input and no known-zero one. Sparse inputs are accepted (the `log_sparsity == 0` requirement is gone), and the output carries the input's own sparsity.

### `poulpy-ckks`: slot-kind metadata

- **Breaking:** `CKKSMeta` gains `slots: SlotsKind` (`Real` / `Complex`, defaulting to `Complex`) recording which subfield the slots are known to live in; every `CKKSMeta` literal must name it. Ops compose it with `SlotsKind::join`, keeping `Real` only when every operand is `Real`; `mul_i` and linear transformations yield `Complex`, and the conjugate folds restamp `Real`. Read with `CKKSInfos::slots`, set with `SetCKKSInfos::set_slots`.
- **Breaking:** the plaintext-assign add/sub methods on the default and OEP layers now require `SetCKKSInfos` on their destination, matching the api layer.

### `poulpy-ckks`: four-layer consolidation

- `CKKSInfos` is a supertrait of `LWEInfos` with provided `log_delta` / `log_budget` / `log_sparsity`; implementors supply only `fn meta()`. New bound-alias traits `CKKSCtBounds` and `CKKSAtkBounds<BE>` replace the spelled-out clusters.
- The add and sub Default/OEP families are generated from one body by `ckks_carry_verb_default!` / `ckks_carry_verb_oep!`; the tensor-multiply variants share `tensor_mul_core`; the multiply-then-accumulate composites share `mul_then_combine`; `impl_ckks_infos!` generates the metadata/view bundle.
- The DFT family joins Pattern A (default-bodied `DFTDefault` / `DFTMatrixDefault`, `impl_ckks_dft_defaults!` reduced to two marker impls, partial overrides now possible), and `ckks_new_dft_matrix` moves onto the scalar-generic `DFTMatrixImpl<BE, F>` / `DFTMatrixDefault` pair.
- `CKKSModuleAlloc` is fully default-bodied over `ModuleCoreAlloc` and gains `ckks_ciphertext_alloc_with_rank`. The `CKKSImpl` bundle also covers `DFTImpl` and `CKKSEvalModImpl`; bootstrapping is a no-OEP composition over `CKKSBootstrappingOps` and `BootstrappingKeys`.
- `UnnormalizedCKKSCiphertextRefMut` is sealed (public for OEP signatures, no public constructor). Shared layout validation helpers move to `layouts/validation.rs`.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512` / `poulpy-cpu-arm`

- Add `ckks_encoding.rs`, the CPU encoding implementation, behind `impl_ckks_encoding_fft64_f64!`, `impl_ckks_encoding_owned_for!` and `impl_ckks_encoding_owned!`. Every backend wires `f64` and `Quad`.
- **Breaking:** backend ring handles hold plan *sets* covering every power-of-two sub-dimension: `FFT64PlanSet` / `get_fft_plan(n)` replace `get_fft_table` / `get_ifft_table`, and `NttPlanSet` / `get_ntt_plan(n)` replace `get_ntt_table` / `get_intt_table`. Handles carry the new `ModulePlanCache`; the old cache names remain as aliases.
- Add `ckks_paco.rs` and `ckks_ship.rs` with the `impl_ckks_paco_coeff_encoding!` and `impl_ckks_ship_coeff_encoding!` macros, invoked by every backend.
- Add benchmarks binaries based on Criterion and the new `poulpy-bench` harness.
- Fix `poulpy-cpu-avx512` compiling with `enable-ckks,enable-avx512f` but without `enable-ifma`.

### `poulpy-core`: fused VMP and tensoring overrides

- Add the Core-owned `GGLWEProductDigitsStridedImpl` backend hook, including scratch sizing, which applies every gadget digit directly from an interleaved DFT input. The portable default materializes one digit at a time; accelerated backends can fuse the passes while preserving the reference digit-width schedule.
- **Breaking for backend implementations:** tensoring becomes an explicit `GLWETensoringImpl` opt-in through `impl_glwe_tensoring_default!` instead of a blanket implementation, allowing optimized backends to override the complete operation and its scratch bounds.
- The AVX-512 NTT backends specialize rank-one tensor apply and square at `n = 2^15` and `n = 2^16` with a direct three-product DFT path, avoiding the coefficient-domain diagonal cache and its associated scratch allocation; other shapes use the canonical all-rank implementation.

### `poulpy-cpu-avx` / `poulpy-cpu-avx512`: NTT, VMP and convolution optimization

- Pack `NTT3x42Ifma` transform-domain vectors and prepared convolution operands into two `u64` words per coefficient instead of three. The existing packed VMP representation is retained, reducing `VecZnxDft` and `CnvPVecL/R` storage and bandwidth by one third.
- Fuse multi-digit VMP on `NTT4x30Avx`, `NTT4x30Avx512` and `NTT3x42Ifma`: digit rows are read directly from their interleaved source, matrix data is streamed once per output block, and intermediate digit buffers are removed. The AVX accumulating save is also vectorized.
- Vectorize packed `NTT3x42Ifma` scalar-vector products, enlarge the cache-resident NTT base case, and fuse the forward NTT stages specialized for `n = 2^15` and `n = 2^16`.
- Fuse rank-one tensor convolution on the AVX-512 NTT backends, computing all three output columns while the prepared inputs remain resident.

### `poulpy-hal`: word genericity delivered

- **Breaking:** `VecZnx`, `ScalarZnx` and `MatZnx` lose the `= i64` default on their word parameter. Sizing follows, and `ZnxWord` gains `from_i64` so `FillUniform` and the secret samplers are word-generic. `layouts::encoding` stays bound to `ZnxWord = i64`.
- `Backend` gains provided sizing methods `size_of_znx_word`, `bytes_of_vec_znx`, `bytes_of_scalar_znx` and `bytes_of_mat_znx`, so sizing routes through the backend that owns the memory.
- Fixed: the `MatZnx` backend entry-view helpers computed offsets from the dense host formula while allocation goes through `Backend::bytes_of_mat_znx`; the stride now calls `B::bytes_of_vec_znx`. The same defect is fixed in the test suite's upload helpers and in `word_compat`, which did not assert matching `ZnxWord`.
- The generic test suites drop 11 `Backend<OwnedBuf = Vec<u8>>` bounds; `TestBackend` remains the single documented `ZnxWord = i64` pin.

### `poulpy-core`: backend-routed sizing

- **Breaking:** the cross-backend layout transfers (`ModuleTransfer::upload_*` / `download_*` and the `to_backend` wrappers) require both backends to share a coefficient word. Moving a value between backends whose limb axis differs is a re-encoding, left to a follow-up.
- Fixed: `lwe_bytes_of_from_infos` summed the body and mask sizes with no padding, under-counting the scratch alignment gap and surfacing as a scratch-exhaustion panic at small limb counts.
- Add `api::GLWEBytesOf<BE>` on `Module<BE>`, replacing host-pinned `Type::<Vec<u8>, i64>::bytes_of` calls at scratch-sizing sites.
- **Breaking:** `ModuleCoreAlloc` gains `type ZnxWord`, and `ModuleCoreCompressedAlloc` gains `type OwnedBuf` and `type ZnxWord`, returning `Self::OwnedBuf` across its constructors instead of `Vec<u8>`.
- The layout family is word-generic throughout; a handful of methods stay i64 by construction and say so.

### `poulpy-ckks`: word-generic containers

- **Breaking:** `CKKSCiphertext<D, W, S = Normalized>` and `CKKSPlaintext<D, W>` carry the coefficient word, the latter also dropping its `D = Vec<u8>` default. The new `CKKSCiphertextOwned<BE>` / `CKKSPlaintextOwned<BE>` aliases absorb the change at most call sites.
- Threading the two containers collapses 316 `Backend<ZnxWord = i64>` bounds to 10, all in the host float codec path.
- `CKKSConjugateImpl` states the `GLWEShift<BE>` bound it actually needs instead of pinning the word.
- Fixed: `delegates::eval_mod` sized backend scratch with the host `VecZnx::bytes_of` rather than `BE::bytes_of_vec_znx`.

### `poulpy-bin-fhe`: word-generic containers and host de-pinning

- **Breaking:** the storage types carry the coefficient word: `FheUint`, `FheUintPreparedDebug`, `LookupTable`, `BlindRotationKey` (both forms), `CircuitBootstrappingKey`, `BDDKey`, `GLWEBlindRetriever` and `LookupTableFactory`; `LookupTable` and `GLWEBlindRetriever` also drop their host defaults.
- 75 bounds state `Backend<OwnedBuf: HostDataMut + HostDataRef>` instead of `OwnedBuf = Vec<u8>`, and key allocation moves off host storage.
- What stays pinned is genuinely i64: the BDD and CGGI evaluation path writes message bits into i64 coefficients, so those items keep `ZnxWord = i64` where the requirement is created.
- Fixed: the compressed `CircuitBootstrappingKey` was generic over its buffer but held host key material, so a device instantiation would have compiled while silently holding it; and the CGGI blind rotation sized backend scratch with the host `VecZnx::bytes_of`.

### `poulpy-bench`: new generic benchmarking API

- Completely reworked the exisiting crate into a composable API for the backend crates to instantiate their benchmarks easily and with minimal boilerplate. The crate defines a generic runner for each backend operation, and a generic sweep driver that turns a table of runners plus a param sweep into Criterion groups. Backends can compose their own benchmark suites by selecting the operations they implement and the sweeps they want to run. See `poulpy-bench/README.md`.

### Build & Docs

- Add `docs/paco.md` and `docs/spec/paco_dft_convention.md`, and `docs/ship.md`; link them from `docs/README.md`.
- Update `docs/bootstrapping.md` for the constructor-validated plans, and add a Toolchain section to the `poulpy-ckks` README (nightly `#![feature(f128)]`, the x86_64 Linux GNU `libquadmath` fast path, the portable fallback elsewhere).
- Add native Intel macOS and Apple Silicon CI smoke coverage for the portable binary128 math surface and `Quad` CKKS encoding.

## [0.7.0] - 2026-07-09

Builds the full leveled-CKKS evaluation stack on the backend-generic core: BSGS polynomial evaluation, slot-domain linear transformations, the homomorphic DFT, EvalMod, and the complete bootstrapping pipeline. Adds the AArch64 NEON backend crate (`poulpy-cpu-arm`), a round of NTT/convolution/VMP performance work, and renames the multi-prime NTT families to `<primes>x<bits>` (entries below use the name current when the change landed).

### Renamed

- Rename the multi-prime NTT backend families to a `<primes>x<bits>` scheme: `NTT120` → `NTT4x30` and `NTT126` → `NTT3x42`, across the backend marker types, modules, CKKS test parameters and documentation. The `Q120` / `Q126` moduli and `Primes30` / `Primes42` prime sets keep their names.

### `poulpy-hal`

- Add scalar-automorphism and packed-matrix APIs: `ScalarZnxAutomorphismBackend`, `ScalarZnxAutomorphismAssignBackend`, `VecZnxTransposeBackend`, `VecZnxBigColWeightedSum`.
- Add reusable DFT-domain automorphism planning via `VecZnxDftAutomorphismPlan` and `VecZnxDftAutomorphism`.
- Add the accumulating convolution apply `Convolution::cnv_apply_dft_accumulate`, bit-identical to `cnv_apply_dft` followed by a DFT-domain add and reusing its scratch contract.
- Add the `VecZnxLshAddCoeffToCoeffBackend` / `VecZnxLshSubCoeffToCoeffBackend` hooks for left-shift-aligned coefficient accumulation.

### `poulpy-core`

- Add a scale-agnostic Baby-Step/Giant-Step polynomial-evaluation engine and the `GLWEPolynomialEvaluation` API. The engine owns only the combinatorial schedule and delegates every arithmetic operation, with all precision bookkeeping, to the scheme through `BSGSBabyOps` / `BSGSGiantOps`.
- Add the scheme-agnostic layouts `Polynomial`, `BSGSPolynomial`, `PowerBasis`, `Basis`, `Parity` and `SplitStrategy`: monomial and Chebyshev bases, interpolation, arbitrary intervals, Paterson-Stockmeyer decomposition under the `MinDepth` / `MinMult` strategies, and ahead-of-time accounting via `bsgs_consumed_bits` and `bsgs_eval_depth`.
- Factor the GLWE tensor product into `glwe_tensor_apply_loop` and add the hoisted-operand primitives `glwe_prepare_right` / `glwe_tensor_apply_prepared_right`.
- Allow GLWE plaintext add/sub alignment to left-shift when the encoded precision exceeds the ciphertext budget.
- Add the scheme-agnostic linear-transformation engine: `LinearTransformation` / `GiantStep` / `Diagonal`, the BSGS schedule types (`LinearTransformationLayout`, `Plan`, `Strategy`, `optimal_bsgs_giant_step`) and the `GLWELinearTransformations` trait. It evaluates `M·v = Σ_k diag_k ⊙ rot(v, k)` over raw GLWE and carries no scale notion.
- Add the convolution-domain prepared operand `PreparedDiagonal`, so the resident transform is `LinearTransformation<PreparedDiagonal>`, the same container as the streamed plaintext form. The left operand stays the separate `LinearTransformationBabySteps`.
- Add a streamed (unprepared-RHS) evaluation path preparing each diagonal through scratch. The per-giant product is dispatched by the `DiagonalProd` trait, so both paths share one driver behind a single `glwe_eval_linear_transformation_into` generic over `P`.
- Add the clear evaluator `Diagonals<T>` with the `Evaluate` / `DiagonalArithmetic` traits, used as the plaintext reference.
- Add packed LWE matrix support: `LWEMatrix`, `LWEMatrixLayout`, `LWEMatrixInfos`, `BackendLWEMatrix`, and the `lwe_matrix_alloc*` constructors.
- Add `GLWEExpandLWEMatrix` (GLWE into a matrix of LWE samples) and `LWEMatrixDecrypt`.
- Add the `GLWEMaskFill` / `LWEFillMask` traits for backend-generic mask generation from a `Source` or seed; compressed decompression uses them, and `GLWECompressed` exposes `data()` / `data_mut()`.
- Accumulate the `dsize > 1` keyswitch digits in place through `vmp_apply_dft_to_dft_accumulate`, dropping the per-digit scratch buffer.

### `poulpy-ckks`

- Add the scale-aware polynomial-evaluation bridge over the core engine: `PolynomialEvaluation`, the `EncodeBSGS` / `PowerBasisGen` extension traits, and `CKKSBSGSOps` dispatching onto the CKKS API so all scale math lives in the CKKS multiply ops.
- Add complex-coefficient evaluation via `ckks_eval_poly_complex_const_coeffs_from_power_basis`, taking a single `&ComplexBSGSPolynomial` and combining matched baby steps as `re + i·im`.
- Add the one-shot `ckks_eval_poly_real_const_coeffs` / `ckks_eval_poly_complex_const_coeffs`, which build the power basis internally.
- Add the hoisted `ct×ct` multiply: `CKKSMulOps::ckks_prepare_right` into the backend-resident `CKKSPreparedRight`, and `ckks_mul_prepared_assign`.
- Add backend-generic `eval_mod` (homomorphic `x mod 1`): `EvalModPlan` over `SinContinuous` / `CosContinuous` / `CosDiscrete` / `Exp`, with double-angle composition, optional arcsine, and a caller-selected BSGS split strategy. It runs at its own `f_mod_log_delta` through a budget-neutral scale round-trip, and `EvalModPlan::consumed_bits` charges the budget deterministically.
- Add the complex-exponential `Exp` variant, held by the new `EvalModBsgs::{Real, Complex}`.
- Add eval_mod OEP/delegate wiring (`CKKSEvalModImpl`, `CKKSEvalModOps`) and conformance tests for every variant.
- Generalize the fused multiply-add/plaintext paths over backend-owned layouts.
- Add conformance tests for CKKS polynomial evaluation.
- Add the CKKS linear-transformation API `LinearTransformationOps`. The CKKS layer owns the scale math and delegates evaluation to the core engine; the prepared-vs-streamed choice is which diagonal type `P` you pass, not a separate method. `ComplexDiagonals<T>` and `ckks_encode_linear_transformation_from_diagonals` build the encoded transform.
- Strengthen `assert_decrypt_precision` to run a ring-domain noise bound and the canonical-embedding per-slot assertion off one decryption.
- Add the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) via `DFTOps`: `ckks_new_dft_matrix` builds a factorized transform from a `DFTPlan` as the unprepared `DFTMatrix`, and `ckks_prepare_dft_matrix` promotes it to the resident `DFTMatrixPrepared`. Both forms share the unified linear-transformation evaluator, and the BSGS schedule is caller-supplied per factor.
- `DFTMatrix<BE, Dir, Fmt, R>` carries its direction (`Encode`/`Decode`) and output format (`Standard`/`Split`/`Repack`) as type-state, making a mismatch a compile error. Conformance tests compare each transform against an independent plaintext reference built from the dual encoding, not an `Encode∘Decode` round trip.
- Add budget/compaction maintenance for leveled pipelines: `SetCKKSInfos::compact_in_place` and `ckks_set_log_delta`. Every budget-consuming op compacts its result.
- Add CKKS bootstrapping (`CKKSBootstrappingOps`): the `ModUp → CoeffsToSlots → EvalMod → SlotsToCoeffs` pipeline. The only new primitive is ModUp (`ckks_mod_up_into`), a base-`2^base2k` digit shift rather than an RNS basis extension. `BootstrappingPlan` bundles the stage parameterization and budget accounting, and `BootstrappingContext::compile` lowers it once into resident matrices plus the encoded EvalMod.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512`

- Add the `bootstrap_trace` example: the end-to-end bootstrapping pipeline as a standalone binary for sampling profilers.
- Implement the new transpose, weighted-sum, automorphism and packed LWE matrix defaults, with AVX and AVX-512 overrides for the automorphism paths.
- Rework the FFT64 convolution applies into fused column kernels behind `Reim4Convolution::reim4_convolution_apply` / `_pairwise_apply`, tiling output limbs over a sliding window and staging outputs so each destination cache line is written once. 2.3-3.5x faster on `FFT64Avx512`/`FFT64Avx`.
- Interleave the `NTT120Avx512` VMP prepared-matrix prime planes per chunk so the apply streams the matrix sequentially: `vmp_apply_dft_to_dft` up to 3.4x faster, `glwe_keyswitch` at n = 2^15 from 41.5 ms to 24.7 ms.
- Switch the NTT-family prepared convolution operands to block-major rows with the right operand in reversed limb order, deleting the per-apply pack/gather passes and moving the left operand's reduction to prepare time. The applies tile four output limbs per pass and reduce once per output. At 32768x14: `NTT126Ifma` apply -36% / pairwise -62%, `NTT120Avx512` `mul_ct` -15%.
- Fuse the `NTT126Ifma` forward level-0 twist into the first butterfly level and run the last three levels as one radix-8 pass (~5% faster); add n = 2^12 cross-backend idft conformance tests.

### `poulpy-cpu-avx`

- Fuse NTT levels in the `NTT120Avx` by-level phase: the forward level-0 twist and the inverse final normalization fold into adjacent butterfly levels, and above `FUSE_MIN_N = 2^15` pairs of levels fuse into radix-4 passes. Bit-identical to the prior radix-2 passes; forward/inverse ~34%/~26% faster at n = 2^16.
- Batch the q120b → i128 CRT reconstruction four coefficients at a time, transposing each residue block so the weighted sum accumulates vertically. Bit-identical and ~15% faster on the kernel.
- Vectorize the `b_to_znx128_avx2` final CRT reduction as a planar 4-wide fold mirroring the AVX-512 path. Bit-identical to the prior finalize.

### `poulpy-cpu-avx512`

- Migrate the `NTT126Ifma` DFT-domain representation from a 4-lane array-of-structs to a planar 3-prime layout, removing the wasted lane and the hand-written assembly CRT kernel.
- Optimize the planar transforms for pass efficiency (masked sub-8 remainders, transposed radix-8 head/tail, fused upper-level pairs, folded normalization/untwist), roughly halving the raw NTT/iNTT cost at log n = 15.
- Vectorize the planar CRT-to-i128 reconstruction with AVX-512-IFMA base-2^52 limb accumulation.
- Replace the `NTT126Ifma` forward/inverse kernels with radix-2 Cooley-Tukey / Gentleman-Sande passes over the three planes, under a lazy reduction feeding the Harvey/Shoup multiply directly. Above `BASE_NTT_SIZE` each plane is split depth-first to stay cache-resident.
- Add a `lazy_output` mode to the `NTT126Ifma` forward NTT, opted into by the prepare paths whose consumers re-reduce; the public DFT contract keeps the full reduction.
- Vectorize the `NTT120Avx512` `b_to_znx128` final CRT reduction with a planar 8-wide fold. Byte-identical to the reference.
- Replace the scalar modular accumulate in the `NTT120Avx512` VMP output save with a lazy SIMD fold reproducing it byte-for-byte.
- Pack the `NTT3x42Ifma` prepared VMP matrix into two u64 words per coefficient instead of three, cutting the streamed bandwidth by a third.

### `poulpy-cpu-arm`

- Add the `poulpy-cpu-arm` NEON/ASIMD backend for AArch64, exposing `FFT64Neon` and `NTT120Neon`. Hand-written kernels cover every accelerated HAL family; the rest inherit the portable `poulpy-cpu-ref` defaults, and the core/CKKS wiring is inherited through the `impl_*_defaults` macros. Opt-in via `enable-neon`, `compile_error!`ing off AArch64. Integer operations are bit-identical to the reference and FFT operations match within ULP. A `neon` CI lane runs the suite natively.
- Add the canonical-x NEON convolution-apply kernel `NttMulBbc1ColX2::ntt_mul_bbc_tile4_x2`, the analog of the AVX2/AVX-512 tiled kernels.

### `poulpy-bench`

- Add the `ckks_poly_eval` benchmark (degree and split-strategy sweep, reporting baby-step size and budget consumption), later extended to `NTT120Avx` and `NTT126Ifma`.
- Add the `ckks_eval_mod` benchmark over the `EvalModType` variants.
- Enable the `poulpy-cpu-avx` CKKS implementations in the `ckks-bench` feature.
- Add the `cnv_apply_dft_accumulate` sweep and a `glwe_tensor_relinearize` benchmark.

### Build & Docs

- Add `docs/linear_transformation.md` and `docs/polynomial_evaluation.md`.
- Refresh the `ckks_poly2` example onto Chebyshev interpolation, `PowerBasis` and the BSGS evaluator.
- Overhaul `docs/`: add `README.md` (index), `backends.md`, `bootstrapping.md`, `grafting-vs-bivariate.md`, and the `docs/spec/` linear-transformation specification; refresh `getting-started.md` and trim the root README.

## [0.6.0] - 2026-05-18

Completes the migration from the host-oriented HAL/backend plumbing to backend-generic HAL and core layers, so backends own buffers, scratch, and transfer paths explicitly, and adds the `poulpy-cpu-avx512` crate with three accelerated backends (`FFT64Avx512`, `NTT120Avx512`, `NTT126Ifma`).

### `poulpy-hal`

- The layout family stores private shape snapshots instead of public mutable fields, with explicit getters (`shape()`, `n()`, `cols()`, `size()`, `max_size()`) and metadata-only `with_size` / `set_size`.
- `Module` becomes the canonical allocation entrypoint for the raw coefficient-domain layouts and `VecZnxBig`; the host-owned `alloc` constructors and the explicit-`n` allocator surface are removed, and host staging goes through `Module::<HostBytesBackend>::new(...)`.
- **Breaking:** HAL compute traits take backend-native borrows and scratch explicitly: `Scratch` → `ScratchArena<'_, BE>`, `*ToRef` / `*ToMut` → `*ToBackendRef<BE>` / `*ToBackendMut<BE>`, and trait names move to backend-explicit forms (`VecZnxAddIntoBackend`, …).
- Add backend-owned interop: `Backend::{OwnedBuf, BufRef, BufMut}`, `HostBackend` / `DeviceBackend`, `TransferFrom<From>`, per-layout view aliases and reborrow traits, the `ScalarZnxAlloc` / `VecZnxAlloc` / `MatZnxAlloc` allocators, and `api::reim::{NegacyclicFFT, NegacyclicFFTNew}`.
- Add `VmpApplyDftToDftAccumulate` (fused `res += a · pmat` with limb offset), replacing the scattered apply-then-fold in `gglwe_product_dft`.
- Fix the convolution API: the output shift is renamed `cnv_offset` and moved to the front of the apply calls.
- **Breaking:** `Convolution::cnv_by_const_apply` takes a backend `VecZnx` plus `(b_col, b_coeff)` selectors instead of a raw coefficient slice.
- Replace the monolithic `oep::HalImpl` with per-family OEP traits (`HalModuleImpl`, `HalVecZnxImpl`, `HalVecZnxBigImpl`, `HalVecZnxDftImpl`, `HalSvpImpl`, `HalVmpImpl`, `HalConvolutionImpl`) and per-family defaults, so a backend overrides only the families it owns.
- Make `WriterTo` for `MatZnx` / `VecZnx` emit the canonical logical byte length and error on short backing storage; fix `ScalarZnx::write_to` to emit the full `n * cols` span.
- **Breaking:** remove `ReaderFrom` / `WriterTo` and `from_bytes` for the prepared DFT layouts (`SvpPPol`, `VmpPMat`), whose alignment assumes a power-of-two degree.

### `poulpy-core`

- Fix #158: `VecZnxScalarProduct` now computes the element-wise Hadamard product into `VecZnxBig`; callers needing the inner sum follow up with `VecZnxBigInnerSumBackend`, as LWE encryption now does.
- Add rank>1 support to `GLWEExpandLWE::glwe_expand_lwe` (with a new `_tmp_bytes`) and to `SecretConversion::lwe_secret_from_glwe_secret`, which produces the concatenated LWE key.
- Add `ModuleCoreAlloc` / `ModuleCoreCompressedAlloc` and migrate every allocation site; the wrapper `alloc` constructors become crate-private.
- **Breaking:** core traits follow the HAL backend-native convention (`ScratchArena<'_, BE>`, `GLWEToBackendRef/Mut<BE>`, prepared-key `...ToBackendRef<BE>`), and no extension point assumes host-slice views.
- Add `api::ModuleTransfer` for typed upload/download of every core object across backends.
- Thread the corrected convolution-offset semantics through the multiply and tensoring paths, pass explicit effective-`k` into them, and mask partial bottom limbs instead of assuming full stored width.
- Fix tensoring noise blowup when the output operand was smaller than the input.
- Split the public API into `api` / `delegates` / `oep` layers, reorganize the operation modules to match, and re-export the top level for ergonomics.
- **Breaking:** backend-default wiring is exported per family (`impl_encryption_defaults_full!`, `impl_glwe_trace_defaults_full!`, …); add `impl_glwe_rotate_impl_from!` for backends forwarding rotation.
- Move backend conformance suites into `src/test_suite`.
- Standardize prepared allocations on `DeviceBuf`, and rename the Module allocation/prepare helpers to struct-first names (`gglwe_prepared_alloc`, `glwe_secret_prepare`).
- **Breaking:** remove `ReaderFrom` / `WriterTo` for `LWESecret` and `GLWESecret`; secret material should use seeds or application-level transfer.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`

- Refresh the FFT64/NTT120 references and glue for the private-shape layouts and the corrected `cnv_offset` API.
- Optimize NTT120 convolution on AVX by wiring the prep paths to backend kernels and restructuring the applies around prepacked x2 blocks.
- Add a row-prime-major prepared-matrix layout for the `NTT120Avx` VMP, streaming one prime plane at a time and reusing extracted input rows across the output-column loop.
- Reorganize backends around `hal_impl` / `hal_defaults` modules mirroring the per-family HAL surface, removing the legacy per-family modules.
- Export the FFT-table types the CKKS encoder needs (`FFT64ReimTable`, `FFT64AvxReimTable`), move the CKKS `poly2` example and backend tests into `poulpy-cpu-ref`, and add `poulpy-cpu-avx`'s opt-in `enable-ckks`.

### `poulpy-cpu-avx512`

- Add `FFT64Avx512`: f64 complex-FFT backend gated on `enable-avx512f`, mixing AVX-512F REIM butterflies with AVX2+FMA REIM4 kernels.
- Add `NTT120Avx512`: Q120 NTT over four ~30-bit primes, gated on `enable-avx512f`, with AVX-512F butterflies, row-prime-major VMP and convolution overrides, and a fused iNTT + Garner compaction hook.
- Add `NTT126Ifma`: Q126 NTT over three ~42-bit primes on AVX-512-IFMA, gated on `enable-ifma`. Implements every HAL family against IFMA kernels; the post-iNTT CRT-to-i128 reconstruction is a hand-written assembly kernel fusing IFMA Garner reduction with a BMI2/ADX carry chain.
- `enable-avx512f` / `enable-ifma` `compile_error!` when the matching target features are missing instead of emitting binaries that SIGILL; with neither feature the crate compiles to an empty shell.
- Opt-in `enable-ckks` mirrors `poulpy-cpu-avx`, running the full CKKS suite against the three backends.

### `poulpy-ckks`

- Implement a fully backend-generic leveled CKKS evaluator: every operation is generic over any `poulpy-hal` backend.
- Organize the interface into the same four layers as `poulpy-core` (`api` / `oep` / `delegates` / `default`), with backends opting into defaults via `impl_ckks_*_defaults!`.
- **Breaking:** collapse the plaintext family split into a unified `CKKSPlaintext<D>` plus `CKKSPlaintextVecHostCodec<F>` for host float encode/decode and `CKKSModuleAlloc` for module-first allocation.
- Add api trait families for copy, affine/composite helpers, and imaginary-unit operations.
- Add the normalization typestate `CKKSCiphertext<D, S = Normalized>` with `UnnormalizedCKKSCiphertext<D>` for fused linear operations before an explicit `normalize`, plus `CKKSCiphertextViewMut` for allocation-free in-place writes.
- Add `ScratchArenaTakeCKKS`, the backend-view aliases, and the `CKKSAddOpsUnnormalized` / `CKKSSubOpsUnnormalized` / `CKKSAffineOps` / `CKKSImagOps` / `CKKSCopyOps` traits.
- Move the conformance suite into `src/test_suite/`, wired into `poulpy-cpu-ref` via `ckks_backend_test_suite!`.
- Remove the concrete-backend dependencies from `poulpy-ckks`; backend crates opt in from their side.
- Preserve the historical `crate::leveled::api` paths as a re-export shim.
- **Breaking:** CKKS backend wiring moves to per-family traits and macros (`CKKSAddImpl`, `impl_ckks_add_defaults!`, …).
- Document unnormalized operations: signed-digit behavior, worst-case O(n) growth, Irwin-Hall O(√n) typical growth, and the `n ≤ 2^(63 − base2k)` bound against i64 overflow.

### `poulpy-bin-fhe`

- **Breaking:** bin-FHE traits follow the backend-owned surface (`ScratchArena<'_, BE>`, `...ToBackendRef/Mut<BE>`), and many entrypoints require `BE: Backend<OwnedBuf = Vec<u8>>` plus `ModuleCoreAlloc`.
- Move the public constructors to module-first forms (`FheUint::alloc`, `LookupTable::alloc`, `GLWEBlindRetriever::alloc`, `CircuitBootstrappingKey::alloc_from_infos`), add `LookupTable::to_backend`, and align the key/prepared layouts with the refactored core.
- Remove the unnecessary `BE: 'static` and redundant `ScratchArenaTakeCore` bounds.
- Add `ReaderFrom` / `WriterTo` for `CircuitBootstrappingKey` and `BDDKey<Vec<u8>>`, with stable sorted-Galois ATK serialization.
- **Note:** the crate is not yet backend-agnostic; it still depends unconditionally on `poulpy-cpu-ref` and exposes host bounds in several public APIs.

### `poulpy-bench`

- Update the suites for the new HAL/core APIs, remove the remaining layout-field assumptions, and add shared host-upload / `ModuleTransfer` staging helpers so fixtures can target any backend.
- Split the bench opt-ins by family (`hal-bench`, `core-bench`, `bin-fhe-bench`, `ckks-bench`) instead of one monolithic feature, and add `enable-avx512f` / `enable-ifma`.

### Build & Docs

- Refresh the READMEs and document the shared `api` / `oep` / `delegates` / `default` layering and backend-integration flow.
- Extend CI with CKKS-focused test steps in both AVX and portable configurations, and add an `avx512` lane that type-checks and lints the AVX-512F + IFMA configuration (tests need self-hosted silicon).
- Add a hugepage hint in the aligned allocator: on Linux, allocations ≥ 2 MB issue `madvise(MADV_HUGEPAGE)` (~5% on `ntt120-avx` at large rings), overridable via `POULPY_HUGEPAGE_MIN_BYTES`.
- Add acknowledgements for PZ, EF, and ENS in the root README.

### Fixes

- Avoid under-allocating scratch in the bin-FHE scheme tests via new tmp-bytes helpers.
- Make the AVX backend optional (`enable-avx`) so non-AVX machines build.

### Migration

HAL backend wiring moves from one `HalImpl` to per-family OEP traits, with `poulpy-cpu-ref` exposing per-family `hal_impl_*!` macros and `*Defaults` traits so a backend inherits the reference path for cold methods and overrides only the hot ones:

```rust
// before
unsafe impl HalImpl<FFT64Avx> for FFT64Avx { hal_impl_vec_znx!(); hal_impl_module_fft64!(); }

// after: one impl per family, overriding individual methods inline
unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx { poulpy_cpu_ref::hal_impl_vec_znx!(); }
unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx { poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefaults); }
```

Core and CKKS wiring likewise replaces `CoreImpl` + `impl_core_default_methods!` with per-family `impl_*_defaults_full!` (and `impl_ckks_*_defaults!`) macros, one call per family, overriding any method by writing it after the call.

## [0.5.0] - 2026-03-31

### `poulpy-bench` (new crate)

- Consolidate every benchmark suite into `poulpy-bench` under `bench_suite::{hal, core, schemes}`, removing the `bench_suite` modules from the other crates.
- Add the `standard` binary: one representative run across all layers at fixed parameters, for version-to-version regression tracking.
- Add JSON-configurable parameters via `POULPY_BENCH_PARAMS` (file path or inline). Every sweep range and layout constant is overridable, with built-in defaults for omitted fields: `hal.sweeps`, `cnv.sweeps`, `vmp.sweeps`, `svp_prepare.log_n`, and `core.{n, base2k, k, rank, dsize}`.
- Add the `run` field (bench binaries or Criterion function filters) and the `backends` field (`fft64-ref`, `ntt120-ref`, `fft64-avx`, `ntt120-avx`); listing an AVX backend enables `enable-avx` and the matching `RUSTFLAGS`.
- Replace the per-group `measurement_time` overrides with a shared `criterion_config()` (100 samples, 5 s).
- Add `examples/custom_params.json` and `examples/run_custom_params.sh`.

### `poulpy-hal`

- Remove `VmpApplyDftToDftAdd` / `SvpApplyDftToDftAdd` and merge the additive variant into `VmpApplyDftToDft` / `SvpApplyDftToDft` via a `limb_offset` parameter, with their OEP and delegate plumbing. Accumulating into a scattered output caused severe cache misses; a contiguous temporary folded with `VecZnxDftAddAssign` is ~2x faster.
- Add family defaults for `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat` and `convolution`, and portable `scratch` / `vec_znx` defaults in `HalImpl`, removing the legacy per-family OEP traits.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`

- Update the FFT64 and NTT120 `vmp_apply_dft_to_dft` to accept `limb_offset` directly, replacing the `_add` codepath.
- NTT120 AVX2: add `reduce_b_and_apply_crt`, fusing the CRT multiply into the Barrett reduction pass, roughly halving the instruction count of `compact_all_blocks`.
- Drop the legacy backend-specific VMP, convolution, scratch, `vec_znx`, `svp` and `vec_znx_dft` impl modules in favor of the HAL family defaults.

### `poulpy-core`

- Rewrite the external-product and keyswitching inner loops to write per-digit VMP results into a temporary before accumulating with `VecZnxDftAddAssign`, avoiding scattered-write cache thrashing.
- Add the `keyswitch::gglwe` bench module and the `keyswitch_glwe` benchmark on NTT120, replacing the FFT64-specific one.

## [0.4.4] - 2026-02-28

### `poulpy-hal`
- Add NTT120 reference primitives: primes, types, arithmetic, NTT butterfly, mat-vec, SVP, VMP, `VecZnxBig`, `VecZnxDft`, and convolution.
- Refactor byte size helpers: centralize scratch/layout size computations into `Module`.
- Consolidate FFT64 trait implementations to eliminate duplication between ref and AVX.

### `poulpy-cpu-ref`
- Add `NTT120Ref` backend: scalar Q120 NTT over CRT of four ~30-bit primes.
  - Full OEP coverage: `VecZnx`, `VecZnxBig`, `VecZnxDft`, `SvpPPol`, `VmpPMat`.
- Reorganize FFT64 sources into `fft64/` submodule.
- Add NTT120 benchmarks.

### `poulpy-cpu-avx`
- Add `NTT120Avx` backend: AVX2-accelerated NTT120.
  - AVX2 NTT butterfly with variable-shift accumulation.
  - AVX2 BBC mat-vec (`NttMulBbc`, x2, 2-column variants).
  - AVX2 arithmetic primitives: `b_from_znx64`, `c_from_b` (Barrett), `vec_mat1col_product_bbb`, `b_to_znx128` (hybrid AVX2/scalar CRT).
  - AVX2 `VecZnxBig` accumulation and normalization.
- Reorganize FFT64 sources into `fft64/` submodule.
- Add NTT120 benchmarks and unit tests for all AVX subroutines.

### `poulpy-core`
- Add NTT120 backend support across all operations: encryption, decryption, automorphisms, external products, keyswitching, and noise analysis.
- Extend test suite to cover `NTT120Ref` and `NTT120Avx`.

## [0.4.3] - 2026-01-16

- Fix [#131](https://github.com/poulpy-fhe/poulpy/issues/131)
- Fix [#130](https://github.com/poulpy-fhe/poulpy/issues/130)

## [0.4.2] - 2025-12-21

### `poulpy-core`

- Add the `GLWEMulPlain`, `GLWEMulConst` and `GLWETensoring` traits (apply, assign, and relinearize forms, each with `_tmp_bytes`), plus the `test_glwe_tensoring` method test.

### `poulpy-hal`

- Add the `CnvPVecL` / `CnvPVecR` structs with the `CnvPVecBytesOf` / `CnvPVecAlloc` traits, and the `Convolution` trait regrouping prepare-left/right, `cnv_apply_dft`, `cnv_pairwise_apply_dft` and `cnv_by_const_apply` (each with `_tmp_bytes`).
- Add the `Reim4Convolution*` and `i64*` convolution traits with their FFT64 reference implementations, rename `Reim4Extract1Blk` to `Reim4Extract1BlkContiguous`, and add `take_cnv_pvec_left` / `take_cnv_pvec_right` to `ScratchTakeBasic`, plus convolution tests and benchmarks.
- **Breaking:** the normalization API and OEP (`VecZnxNormalize`, `VecZnxBigNormalize`) take `res_offset: i64`, a positive or negative bit-shift applied before the mod-1 reduction, with inputs reordered for consistency. This completes cross-base2k normalization at arbitrary offset; correctness is ensured, optimality is not.

### `poulpy-cpu-ref`

- Implement the convolution OEP on `FFT64Ref`, with tests and benchmarks.

### `poulpy-cpu-avx`

- Implement the convolution OEP on `FFT64Avx` with AVX FFT64 kernels (`reim4_save_1blk_to_reim_contiguous_avx`, `reim4_convolution_1coeff_avx`, `reim4_convolution_2coeffs_avx`), with tests and benchmarks.

## [0.4.1] - 2025-11-21
- Default backend set to `poulpy-cpu-ref`, `poulpy-cpu-avx` is not anymore built and compiled by default.
- To build & use `poulpy-cpu-avx` user must use feature flag, see `poulpy-cpu-ref` and `poulpy-cpu-avx` READMEs.

## [0.4.0] - 2025-11-20

### Summary
- Full support for base2k operations.
- Many improvements to BDD arithmetic.
- Removal of **poulpy-backend** & spqlios backend.
- Addition of individual crates for each specific backend.
- Some minor bug fixes.

### `poulpy-hal`
- Add cross-base2k normalization

### `poulpy-core`
- Add full support for automatic cross-base2k operations & updated tests accordingly.
- Updated noise helper API.
- Fixed many tests that didn't assess noise correctly.
- Fixed decoding function to use arithmetic rounded division instead of arithmetic right shift.
- Fixed packing to clean values correctly.

### `poulpy-schemes`
- Renamed `tfhe` crate to `bin_fhe`.
- Improved support & API for BDD arithmetic, including multi-thread acceleration.
- Updated crate to support cross-base2k operations.
- Add additional operations, such as splice_u8, splice_u16 and sign extension.
- Add `GLWEBlindRetriever` and `GLWEBlindRetrieval`: a `GGSW`-based blind reversible retrieval (enables to instantiate encrypted ROM/RAM like object).
- Improved Cmux speed
- Added `sign` argument to GGSW-based blind rotation, which enables to choose the rotation direction of the test vector.

### `poulpy-cpu-ref`
- A new crate that provides the reference CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_ref**.

### `poulpy-cpu-avx`
- A new crate that provides an AVX/FMA accelerated CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_avx**.

## [0.3.2] - 2025-10-27

### `poulpy-hal`
- Improved convolution functionality

### `poulpy-core`
 - Rename `GLWEToLWESwitchingKey` to `GLWEToLWEKey`.
 - Rename `LWEToGLWESwitchingKey` to `LWEToGLWEKey`.
 - Add `GLWESecretTensor` which stores the flattened upper right of the tensor matrix of the pairs  `sk[i] * sk[j]`.
 - Add `GGLWEToGGSWKey`, `GGLWEToGGSWKeyPrepared`, `GGLWEToGGSWKeyCompressed`, which encrypts the full tensor matrix of all pairs `sk[i] * sk[j]`, with one `GGLWE` per row.
 - Update `GGLWEToGGSW` API to take `GGLWEToGGSWKey` instead of the `GLWETensorKey`
 - Add `GLWETensor`, the result of tensoring two `GLWE` of identical rank.
 - Changed `GLWETensorKey` to be an encryption of `GLWESecretTensor` (preliminary work for `GLWEFromGLWETensor`, a.k.a relinearization). 

### `poulpy-schemes`
 - Add `GLWEBlindRotation`, a `GGSW`-based blind rotation that evaluates `GLWE <- GLWE * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).
 - Add `GGSWBlindRotation`, a `GGSW`-based blind rotation that evaluates `GGSW <- (GGSW or ScalarZnx) * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).

## [0.3.1] - 2025-10-24

### `poulpy-hal`
 - Add bivariate convolution (X, Y) / (X^{N} + 1) with Y = 2^-K

### `poulpy-core`
 - Fix typo in the shared backend-view impl for `GLWEAutomorphismKey` that incorrectly required mutable data.

## [0.3.0] - 2025-10-23

- Fixed builds on MACOS

### Breaking changes
 - The changes to `poulpy-core` required to break some of the existing API. For example the API `prepare_alloc` has been removed and the trait `Prepare<...>` has been broken down for each different ciphertext type (e.g. GLWEPrepare). To achieve the same functionality, the user must allocated the prepared ciphertext, and then call prepare on it.

### `poulpy-hal`
 - Added cross-base2k normalization

### `poulpy-core`
 - Added functionality-based traits, which removes the need to import the low-levels traits of `poulpy-hal` and makes backend agnostic code much cleaner. For example instead of having to import each individual traits required for the encryption of a GLWE, only the trait `GLWEEncryptSk` is needed.

### `poulpy-schemes`
 - Added basic framework for binary decision circuit (BDD) arithmetic along with some operations.

## [0.2.0] - 2025-09-15

### Breaking changes
 - Updated the trait `FillUniform` to take `log_bound`.

### `poulpy-hal`
 - Added pure Rust reference code for `vec_znx` and `fft64` backend.
 - Added cross-backend generic test suite along with macros.
 - Added benchmark generic test suite.

### `poulpy-backend`
 - Added `FFTRef` backend, which provides an implementation relying on the reference code of `poulpy-hal`.
 - Added `FFTAvx` backend, which provides a pure Rust AVX/FMA accelerated implementation of `FFTRef` backend.
 - Added cross-backend tests between `FFTRef` and `FFTAvx`.
 - Added cross-backend tests between `FFTRef` and `FFT64Spqlios`.

### `poulpy-core`
 - Removed unsafe blocks.
 - Added tests suite for `FFTRef` and `FFTAvx` backends.

### Other
 - Fixed a few minor bugs.

## [0.1.0] - 2025-08-25
 - Initial release.
