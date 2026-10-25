# CHANGELOG

## [Unreleased]

### `poulpy-hal`

- Add `mat_znx_copy` (`MatZnxCopy`, `HalMatZnxImpl::mat_znx_copy`), the copy of one output column of every entry of a matrix, a `vec_znx_copy` on each entry by default.
  `mat_znx_normalize` takes a destination of any number of rows: its rows past the source's are zero, and the source's rows past it are not read.
- **Breaking:** `vec_znx_idft_normalize_consume` and `vec_znx_idft_normalize_consume_tmp_bytes` become `vec_znx_idft_normalize` and `vec_znx_idft_normalize_tmp_bytes`, the traits `VecZnxIdftNormalizeConsume` and `VecZnxIdftNormalizeConsumeTmpBytes` become `VecZnxIdftNormalize` and `VecZnxIdftNormalizeTmpBytes`, and the tests `test_vec_znx_idft_normalize_consume` and `test_vec_znx_idft_normalize_consume_derived` become `test_vec_znx_idft_normalize` and `test_vec_znx_idft_normalize_derived` ([#430](https://github.com/poulpy-fhe/poulpy/issues/430)).
  The operation still clobbers `a[a_col]`, as its `&mut` source and its contract state.
- **Breaking:** `cnv_apply_dft` and `cnv_apply_dft_add` take a Galois element `b_galois` after `b_col`, and `CnvDftAccTerm` has a `b_galois` field: the applies read their prepared right operand through `σ_{b_galois}`, so products by the automorphisms of one prepared operand store it once; `1` reads it as before.
  An even `b_galois` panics with "convolution: b_galois must be odd", the message of the new `check_galois`, and a sum checks every term before its first write.
  Out-of-tree backends that override `cnv_apply_dft_sum` must read each term's `b_galois`: an override that ignores it still compiles.
- **Breaking:** add `VecZnxGadgetDecompose` and `VecZnxGadgetDecomposeTmpBytes`, the gadget digit layout of a column for a gadget product whose digits have any width `k_digit` in bits ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  Each digit is the centered residue modulo `2^k_digit` of the input rounded at the gadget precision, written as `ceil(k_digit / base2k)` limbs aligned on its own bottom bit, the layout the interleaved-digit product reads.
  `HalVecZnxImpl` gains the required methods `vec_znx_gadget_decompose` and `vec_znx_gadget_decompose_tmp_bytes`, and `test_vec_znx_gadget_decompose` checks a backend against a reference and against the contract.
- **Breaking:** `VecZnxIdftApplyTmpA`, `HalVecZnxDftImpl::vec_znx_idft_apply_tmpa` and `test_vec_znx_idft_apply_tmpa` are removed; callers use `vec_znx_idft_apply` with `vec_znx_idft_apply_tmp_bytes` of scratch, which also removes the per-call allocation of the AVX-512 NTT4x30 and IFMA backends ([#339](https://github.com/poulpy-fhe/poulpy/issues/339)), and the default `vec_znx_idft_normalize_tmp_bytes` covers that scratch.
- The contract key `sparse` is renamed `embed`, the module `test_suite::sparse` becomes `test_suite::embed`, and the tests `test_vec_znx_sparse_add_sub`, `test_vec_znx_big_sparse_add_sub` and `test_convolution_sparse` become `test_vec_znx_embed_add_sub`, `test_vec_znx_big_embed_add_sub` and `test_convolution_embed`.
- **Breaking:** `VecZnx::encode_vec_i64_strided`, `encode_vec_i128_strided`, `decode_vec_i64_strided` and `decode_vec_i128_strided` are removed; the dense forms take one value per coefficient.
- **Breaking, behaviour:** normalization rounds ties to even instead of toward positive infinity (`rnd` in the HAL notation).
  Ties of a few dropped exact bits no longer push every coefficient the same way; outputs change only at exact ties.
- Add `MatZnxAddAssign::mat_znx_add_assign`, the entrywise sum of two `MatZnx` of one shape, limb by limb and not renormalized, in the new `api::mat_znx` family ([#380](https://github.com/poulpy-fhe/poulpy/issues/380)).
  **Breaking:** its OEP method sits in the new family `HalMatZnxImpl: HalVecZnxImpl`, which every backend implements; the default body, `oep::mat_znx_add_assign_derived`, is one `vec_znx_add_assign` per output column of every entry, so an empty `unsafe impl` suffices. `test_mat_znx_add_assign` runs in every backend suite.
- Add `MatZnxNormalize::mat_znx_normalize` and `MatZnxNormalizeAssign::mat_znx_normalize_assign`, with `MatZnxNormalizeTmpBytes`, the normalization of one output column of every entry of a `MatZnx`, as `vec_znx_normalize` and `vec_znx_normalize_assign` on each entry, which their `HalMatZnxImpl` default bodies are; `test_mat_znx_normalize` runs in every backend suite.
- **Breaking:** `ScalarZnx` implements `Debug` on host-readable storage only and prints its shape and the first 16 coefficients of each column, as `VecZnx` and `MatZnx` do, where it derived `Debug` and printed its whole buffer, and `CnvPVecL` and `CnvPVecR` implement `PartialEq` and `Eq` (same shape, same bytes), as `VmpPMat` does ([#450](https://github.com/poulpy-fhe/poulpy/issues/450)).
  `VecZnxDft`, `VecZnxBig`, `SvpPPol`, `VmpPMat`, `CnvPVecL` and `CnvPVecR` implement `Debug`, which prints the shape only.
  The test suites compare layouts with `assert_eq!` through these impls, and raw bytes only under a static message, so a failing assertion no longer prints whole buffers.
- `max_base2k_ntt` and `max_base2k_fft64` model the conjugate-invariant ring, read from the backend's `Ring` ([#435](https://github.com/poulpy-fhe/poulpy/issues/435)).
  The constant coefficient of a product has variance `4n - 3` there, twice the others', and a square's has mean `(2n - 1)` times the input variance, which comes off the NTT threshold `Q/2` and, summed coherently through the inverse DCT, adds `2u^2/3` of its square to the FFT64 error.
  FFT64 budgets each DCT as its cyclic FFT plus three stages. At `n = 2^16` and 128 bits, 32 products select 53 bits on NTT4x30 and 18 on FFT64, one square 52 and 17.
- **Breaking:** `MaxBase2k::max_base2k` and `Module::max_base2k` return `usize`, where they returned `Option<usize>`: a backend without a model does not implement `MaxBase2k`, and `HostBytesBackend` no longer does.
- The `# Degree` and `# Preconditions` docs of `poulpy_hal::api` state the coefficient-wise reduction exception and the column bound, and `backend_safety_contract.md` requires a release `assert!` to refuse an aliasing a kernel cannot handle.
- **Breaking:** `HalVecZnxMonomialImpl` requires `Backend<Ring = Standard>`, as `HalVecZnxCIImpl` does, so a conjugate-invariant backend can no longer implement monomial multiplication; out-of-tree backends implement it on their standard-ring types only.
- `scalar_znx_automorphism` and `scalar_znx_ci_embed` become derived operations with the OEP methods `HalVecZnxImpl::scalar_znx_automorphism` and `HalVecZnxCIImpl::scalar_znx_ci_embed`, whose default bodies, `oep::scalar_znx_automorphism_derived` and `oep::scalar_znx_ci_embed_derived`, run `vec_znx_automorphism` and `vec_znx_ci_embed` on the operands read as one-limb `VecZnx`, as the delegates did; the delegates now dispatch to these methods, and outputs are unchanged.
  `test_scalar_znx_automorphism_derived` and `test_scalar_znx_ci_embed_derived` pin the default bodies in every `derived` suite, the second on the standard ring only, and `test_scalar_znx` joins the AVX-512 and emulated test filters.
- **Breaking:** the rejection suite functions are renamed to `test_<op>_rejects_<case>`: `test_vec_znx_dft_step_zero_rejected` becomes `test_vec_znx_dft_rejects_zero_step`, `test_convolution_prepare_shape_rejected` becomes `test_cnv_prepare_rejects_mismatched_shape`, `test_convolution_by_const_degree_rejected` becomes `test_cnv_by_const_rejects_degree_mismatch` and `test_vmp_apply_dft_to_dft_shape_rejected` becomes `test_vmp_apply_dft_to_dft_rejects_column_mismatch`.
  A backend test list that names them registers the new names, and each now checks that a rejected call leaves its destination unchanged.
  Every suite scratch arena is `0xA5`-poisoned and sized by the tested operation's own `_tmp_bytes` alone, so an operation that reads scratch it did not write or exceeds its query fails; `test_convolution_by_const`, `test_convolution`, `test_convolution_pairwise`, `test_vmp_apply_dft`, `test_vmp_zero` and `test_mat_znx_normalize` require the `_tmp_bytes` traits of the setup calls they now size separately, and `test_vec_znx_big_normalize` no longer requires `VecZnxBigAutomorphismAssignTmpBytes`.
  No suite function requires `ScratchOwned<_>: ScratchOwnedAlloc<_>` any more.

### CPU backends

- **Breaking:** `impl_cpu_core_defaults!` registers the core trace on both rings, where each standard backend registered it apart: an out-of-tree backend calling it drops its own `impl_glwe_trace_derived_full!`, which now conflicts.
- `conjugate_invariant_core_test_suite!` also runs `test_glwe_trace_parity`.
- Every conjugate-invariant backend type has a `max_base2k` model: each backend implements `MaxBase2k` once for both rings, where the conjugate-invariant types returned `None`.
- The AVX-512 rank-one relinearization and the fused IFMA giant step refuse a key that does not cover their input, as the reference does.
- The native CKKS encapsulated modulus raise (portable, AVX2, AVX-512, and their Rayon variants) and the fused IFMA giant step leave a key with a row offset to the reference, which refuses it, and the AVX-512 rank-one relinearization refuses one, instead of using it as an offset-free key.
- Every CPU backend reads the right operand of the convolution applies through the Galois element in the pass that reads a sparse operand.
- **Breaking:** the `Reim4Convolution` and `Fft64RingArith` convolution methods take a `RightRead` in place of `b_log_gap`, built with `right_read_fft64` or `right_read_ntt`.
  `reim4_gather_sparse_block_portable` and `reim4_gather_sparse_block_sum_portable` become `reim4_gather_block_portable` and `reim4_gather_block_sum_portable`, which read through a `RightRead`.
  `convolution_apply_dft_portable`, `ntt4x30_cnv_apply_dft_portable`, their `_add` forms, the packed NTT4x30 drivers and `PackedNtt4x30Base::cnv_apply_dft` and `cnv_apply_dft_add` take `b_galois` after `b_col`.
- **Breaking:** `embed_log_gap_portable`, `RightRead::new`, `RightRead::embedded`, `right_read_fft64` and `right_read_ntt` take the backend's `MIN_DEGREE` as a last argument, in place of the literal floor 8: every caller passes its own backend's `BE::MIN_DEGREE`.
- **Behaviour:** the FFT64Avx512 convolution applies refuse a right operand below its `MIN_DEGREE` of 16 where they accepted degree 8, and the oracle backend refuses a right operand below `MIN_DEGREE` as the kernels do.
- Every CPU backend implements `vec_znx_gadget_decompose` with one portable kernel, a single limb-major pass with exact carries that needs no scratch, and the oracle backend with an independent big-integer implementation.
- **Breaking, behaviour:** the CPU backends no longer override the CKKS encapsulated ModUp: `NTT4x30Avx`, `NTT4x30Avx512`, `NTT3x42Ifma` and their Rayon variants install the reference body (`impl_ckks_encapsulated_mod_up_reference_full!`), as `NTT4x30Portable` and `NTT4x30PortableRayon` do.
  Their bodies skipped the limbs that the raise leaves at zero. A sparse-to-dense key with a row offset, the bootstrapping presets' key, leaves none: its switch reads the raised ciphertext at its own width, and no preset took them.
- The IFMA linear transformation keeps its fused giant-step rotation for automorphism keys of any digit width: each normalized mask column is laid out on the digit grid with `gadget_dft_apply` before the product, as in the reference lazy giant step.
  Keys whose digits fit in one limb still take the reference, as one-limb aligned keys do.
- The AVX-512 rank-one relinearization keeps its packed path for tensor keys of any digit width: digits off a limb boundary lay the tensor pairs out on the digit grid with `gadget_dft_apply`, and its scratch covers that operand.
- Add `FFT64PortableRayon` and `NTT4x30PortableRayon`, the Rayon variants of the portable backends, with their conjugate-invariant aliases `FFT64CIPortableRayon` and `NTT4x30CIPortableRayon`.
  They live in `poulpy-cpu-rayon`, which already depends on `poulpy-cpu-portable`, behind no feature for the HAL and behind `enable-core`, `enable-ckks`, `enable-bin-fhe` and `enable-mhe` for the scheme layers.
  `NTT4x30PortableRayon` runs the fused interleaved-digit product of `NTT4x30Portable` on per-worker scratch, and transforms the four planes of a limb as separate tasks from degree `2^13`.
  A vector of one limb is transformed serially.
- `poulpy-cpu-rayon` gains `impl_ntt4x30_rayon_backend!` and the trait `PackedNtt4x30Base`: a serial NTT4x30 backend with a packed `u32` transform domain describes its drivers through the trait, and the macro builds its Rayon variant, as `impl_fft64_rayon_backend!` does for FFT64.
- `NTT4x30NeonRayon` is built with `impl_ntt4x30_rayon_backend!` on the drivers of `NTT4x30Neon`, in place of its hand-written wiring.
  The split of a limb into plane tasks moves into the NEON limb transforms, which take the task executor, and the NEON convolution prepare takes the forward transform as a closure.
  Results are unchanged.
  The plane tasks now go through `RayonTaskExecutor::join`, so they follow its nesting budget and idle polling, and the backend items are implemented per ring instead of generically over `R: Ring`.
- The convolution applies of `PackedNtt4x30Base` take the base module, and `cnv_apply_dft_sum` has a default body that overwrites with the first term and adds the others.
  The element-wise operations of `impl_ntt4x30_rayon_backend!` run serially inside a nested Rayon region, as its other operations do.
- `poulpy-cpu-rayon` gains `impl_packed_ntt4x30_base!`, which implements `PackedNtt4x30Base` by forwarding each method to a driver module of the backend that carries the method's name and arguments.
  The AVX2 and AVX-512 NTT4x30 backends each provide such a module, with their own adaptations.
- `poulpy_cpu_portable::kernels` exposes, as hidden items, the `SendPtr` and the per-thread working buffer that the CPU backends share, and `poulpy_cpu_portable::ntt4x30::drivers::for_each_limb`.
  The NEON, AVX2, AVX-512 and IFMA backends and `poulpy-cpu-rayon` use them in place of their own copies.
- `NTT4x30AvxRayon` and `NTT4x30Avx512Rayon` are built with `impl_ntt4x30_rayon_backend!` on the drivers of `NTT4x30Avx` and `NTT4x30Avx512`, in place of their hand-written wiring.
  The element-wise transform-domain operations of both backends take the task executor and run one task per limb from degree `2^13`, their convolution prepares take the forward transform as a closure, and the AVX-512 convolution drivers are generic over the backend that tags their operands.
  A limb is transformed as one task: its residues are interleaved, so it has no planes to split.
  Results are unchanged, and the backend items are implemented per ring instead of generically over `R: Ring`.
  The convolution applies read the metadata of the module instead of building it per call.
  `vec_znx_dft_apply` and `svp_prepare` of `NTT4x30Avx` and `NTT4x30Avx512`, serial and Rayon, transform through a buffer each thread keeps, where they allocated one per call: the HAL gives these two operations no scratch.
  `poulpy-cpu-avx` no longer depends on `rayon` directly.
  **Breaking, API:** the two Rayon types no longer implement the q120 kernel traits of `poulpy_cpu_portable::kernels::ntt4x30` (`NttFromZnx64`, `NttToZnx128`, `NttAdd`, `NttMulBbc` and the like), which their serial backends keep.
- `poulpy_cpu_portable::ntt4x30::drivers` exposes, as a hidden item without stability promise, the packed drivers that the Rayon variant builds on.
- **Fix:** the portable `znx_add`, `znx_sub`, `znx_negate` kernels and their in-place forms, the power-of-two multiplications and the rotating automorphism wrap on overflow, as the oracle and the SIMD backends do. They panicked in builds with overflow checks.
- **Breaking, behaviour:** `NTT4x30Portable` stores the transform domain as four `u32` residues per coefficient (`DftWord = CrtWord<Primes30, u32>`), half the previous size, for `VecZnxDft`, `SvpPPol`, `VmpPMat` and both convolution operands.
  The layouts are those of the NEON backends: a transformed limb is one plane of canonical residues per prime, prepared operands are multiplied by `2^32`, the prepared matrix and the convolution operands hold residues centered around zero in the order their kernels read them.
  `NTT4x30Portable` and the NEON NTT4x30 backends are byte-compatible again for `VecZnxDft` and `SvpPPol`: the `VecZnxDftLayoutCompatible` and `SvpPPolLayoutCompatible` markers and the word-compatibility suite between them are back.
  The q120 kernels of `poulpy_cpu_portable::kernels::ntt4x30` are unchanged and stay the base of the AVX2 and AVX-512 backends.
- `NTT4x30Portable` runs a 32-bit NTT in place on the packed limb, on both rings, in plain scalar Rust: signed residues, a precomputed twiddle quotient, cache-blocked levels, a forward transform that reads the `i64` coefficients directly and an inverse transform that folds `1/n` and the CRT constant into its last level.
  Its loops have a fixed stride over one prime, and the two halves of a butterfly block go through a function boundary that tells the compiler they do not overlap, so that the target's vector instructions are used.
  The forward transform returns zero for a zero limb without running, and no transform or prepare allocates.
- The vector-matrix product and the convolutions of `NTT4x30Portable` share one inner product on signed accumulators, 24 products per Montgomery step, gather their inputs and stage their outputs in runs of 32 blocks of four coefficients, and take that staging buffer from scratch.
  `vmp_apply_dft_to_dft_add`, `cnv_apply_dft_add`, `cnv_apply_dft_sum` and `cnv_pairwise_apply_dft` accumulate in the same pass, `svp_apply_dft` transforms each limb into its destination, and `vec_znx_idft_normalize` writes the coefficients over the limb it transforms.
- `NTT4x30Portable` implements `GGLWEProductDigitsStridedImpl` with a fused kernel for up to 16 gadget digits, which returns the residues of the reference body and skips the rows of leading input limbs that are zero in every column.
- The middle step of the portable `i128` normalization takes the digit and the carry of the sum of the input and the incoming carry in one step.
  It returns the same values as the two-step form.
- The portable FFT64 passes load, combine and store four elements at a time, and the two merged layers of a pass run in one sweep.
  Every element goes through the same operations in the same order, so the transforms return the same bytes.
- `vmp_apply_dft_to_dft_add` of the FFT64 backends, serial and Rayon, accumulates each block as it is computed, where it filled a zeroed scratch vector and added it afterwards.
  Its scratch shrinks to the one of `vmp_apply_dft_to_dft`.
- **Fix:** the NTT convolution apply kernels of the AVX2, AVX-512 and NEON backends checked the size of the result but not its column, and wrote past the buffer for a column past the last one; they now panic.
- **Fix:** the convolution prepare of the NTT4x30 AVX2, AVX-512 and NEON backends gave each parallel task a mutable view of the whole column while the tasks of the other limbs of that column wrote to it; each task now writes only the rows of its limb.
- **Fix:** the FFT64 DFT kernels of the portable, AVX2, AVX-512 and NEON backends checked operand degrees but not columns.
  `vec_znx_dft_automorphism_add` wrote past the buffer for a result column past the last one, and the parallel `vec_znx_idft_apply` of the FFT64 Rayon backends read another limb for a source column past the last one.
  Every FFT64 DFT kernel now panics on an out-of-range column, also on vectors of no limb (#422).
- `poulpy_cpu_portable::hal_defaults::take_host_typed` is public, and the SIMD backends use it in place of their own copies.
- **Breaking:** the kernels of `vec_znx_idft_apply_tmpa` are removed with it: `vec_znx_idft_apply_tmpa_portable` and `ntt4x30_vec_znx_idft_apply_tmpa_portable` of `poulpy_cpu_portable::kernels`, and `PackedNtt4x30Base::idft_limb_tmpa` of `poulpy-cpu-rayon`.
- The AVX-512 backends run `glwe_tensor_apply_prepared` at rank 1 on the fused kernel of `glwe_tensor_apply`.
- `NTT3x42IfmaRayon` runs `glwe_eval_linear_transformation_sum_into` on the parallel evaluation of `glwe_eval_linear_transformation_into`.
  Both run the reference checks before resolving a key, and take the reference path for a destination below the module degree.
- **Breaking:** `SparseWord` becomes `MixedWord`, `sparse_log_gap_portable` becomes `embed_log_gap_portable`, `RightRead::sparse` becomes `RightRead::embedded`, and the conjugate-invariant test `conjugate_invariant_sparse_convolution` becomes `conjugate_invariant_embed_convolution`.
- Add `enable-blas` to `poulpy-cpu-portable`, `poulpy-cpu-rayon`, `poulpy-cpu-avx`, `poulpy-cpu-avx512`, `poulpy-cpu-arm` and `poulpy-cpu-oracle`: it implies the crate's `enable-ckks` and selects the `poulpy-blas` reference for their standard and conjugate-invariant backends, Rayon and IFMA ones included.
  The oracle runs `blas_backend_test_suite!` and parity between its families, every other backend type registers parity against the oracle of its family (`NTT3x42` types against the `NTT4x30` oracles), and every conjugate-invariant type, the oracle's included, runs `blas_conjugate_invariant_test_suite!`.
- Add the `blas_pipeline` example to `poulpy-cpu-portable` (`--features enable-blas`): it plans its keys from `BlasPlan` and `sum_galois_elements`, runs `blas_matmul`, `blas_transpose`, `blas_matmul_pt_right`, `blas_matmul_pt_left` and `blas_sum` on a batch of encrypted matrices and checks the result in the clear.
  It plans every layout from the output-layout queries and generates each key once, at the widest width it is looked up at.
- **Breaking:** CPU backends normalize through one blocked kernel, `ZnxNormalize`, and take no normalization scratch: `vec_znx_normalize_tmp_bytes` and `vec_znx_big_normalize_tmp_bytes` return 0.
  The traits `ZnxNormalize*Step*`, `ZnxNormalizeDigit`, `ZnxExtractDigitAddMul`, `I64NormalizeOps` and `I128NormalizeOps` with their kernel functions, and `I128NormalizePlan`, are removed.
  The `kernels::normalization` module, `AddOp`, `SubOp` and `AssignOp` of `kernels::ntt4x30::vec_znx_big`, `ntt4x30_vec_znx_big_normalize*_portable`, `kernels::fft64::vec_znx_big::vec_znx_big_normalize_tmp_bytes_portable` and `vec_znx_lsh_assign_carry_bytes_portable` are removed.
  `vec_znx_big_normalize_portable` moves from `kernels::fft64::vec_znx_big` to `kernels::vec_znx_big_normalize` and serves both big words.
  `poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par` is renamed `vec_znx_big_normalize_par`, the exported macro `rayon_forward_znx_const!` is removed, and the public normalize entries lose their `carry` parameter.
  The shared suites `test_suite::normalization::test_normalization_kernels` and `test_suite::normalization_i128::test_i128_normalize_fused` are replaced by `test_suite::znx_normalize::test_znx_normalize`.
  Out-of-tree CPU backends implement `ZnxNormalize`, whose default methods run the portable kernel.
- The normalization kernel runs on `std::simd` lanes on every CPU target, so no backend carries hand-written normalization lanes.
  Each backend picks its lane widths through the associated types of `ZnxNormalize`, portable widths by default; the AVX-512 backends fill zmm registers.
  `poulpy-cpu-portable` enables the nightly features `portable_simd`, `hint_prefetch` and `associated_type_defaults`.
- The portable `vec_znx_gadget_decompose` kernel bounds-checks its columns once and reads its limbs through raw pointers, where it called `at` and `at_mut` per limb and per chunk of coefficients ([#460](https://github.com/poulpy-fhe/poulpy/issues/460)). Results are unchanged.
- **Breaking:** the CPU backends, the oracle, `impl_cpu_core_defaults!` and `impl_cpu_ckks_defaults!` register through the renamed scheme opt-ins (`impl_<family>_reference_full!`, see the `poulpy-core`, `poulpy-ckks`, `poulpy-bin-fhe`, `poulpy-mhe` and `poulpy-blas` entries), each standard type invoking both halves of the split CKKS fold and DFT opt-ins.
  `impl_ckks_paco_coeff_encoding!`, `impl_ckks_ship_coeff_encoding!` and `impl_ckks_encoding!` implement the renamed OEP methods, and `impl_ckks_paco_coeff_encoding!` the context-taking `ckks_paco_coeff_encodings`; an out-of-tree backend calling them is unchanged, and one invoking a scheme opt-in directly renames the invocation.
- `impl_cpu_core_defaults!`, the oracle's `impl_oracle_core!` and the hand-listed IFMA Rayon type register `impl_decompression_reference_full!`, `impl_preparation_reference_full!` and `impl_secret_reference_full!`, so every CPU backend implements `DecompressionImpl`, `PreparationImpl` and `SecretImpl` with the reference and derived bodies.
- **Breaking:** every standard CPU backend (portable, AVX2, AVX-512, IFMA, NEON and their Rayon variants) and the oracle's standard types register `impl_ckks_ring_map_derived_full!`, `impl_ckks_paco_derived_full!` and `impl_ckks_ship_reference_full!`; `impl_cpu_ckks_defaults!` and the oracle's `impl_oracle_ckks!` register the five new both-ring derived families through `impl_ckks_reference_full!`, which `impl_oracle_ckks_without_eval_mod!` and the CKKS comparison adapters list directly. `impl_ckks_encoding!` implements the renamed encoding kernels.
  `impl_cpu_ckks_defaults!` registers the EvalMod reference on both rings, which the standard backends no longer do on their own, and the conjugate-invariant oracle backends register it too.
  An out-of-tree standard backend calling `impl_cpu_ckks_defaults!` adds the three standard-only opt-ins and removes its own `impl_ckks_eval_mod_reference_full!`.
- The CKKS override probes cover the new compositions: `ckks_overrides::composite::{add_many_enters_the_selected_additions, multiply_accumulate_enters_the_selected_addition_and_subtraction, affine_forms_enter_the_selected_plaintext_additions, approximations_enter_the_selected_map_and_polynomial_evaluation}`, `ckks_overrides::ring_map::ring_maps_enter_the_selected_core_maps` and `ckks_overrides::paco_ship::{paco_drivers_enter_the_selected_coefficient_encoding, ship_drivers_enter_the_selected_masking_and_mux}`, each comparing its outputs with the portable backend's bit for bit on shared keys and inputs; the add-many probe also runs the single-input copy on exactly its query, and the PaCo probe counts a slot-trace override whose extra workspace its query reports.
  `ckks_overrides::dft::conditional_dft_fallback_reenters_selected_evaluation` becomes `dft_format_wrapper_reenters_selected_evaluation`: the probe no longer overrides the repack wrapper, whose derived default enters the selected evaluation.
- The bin-fhe `register_backend!` of `poulpy-cpu-portable`, `poulpy-cpu-avx`, `poulpy-cpu-avx512` and `poulpy-cpu-arm` loses its `scheduling` argument, and `poulpy-cpu-rayon`'s no longer passes one: every type registers the plain `impl_bin_fhe_reference_full!`, and the Rayon types run the parallel blind rotation because their executor is parallel.
- `poulpy-cpu-oracle` gains `enable-mhe` (`dep:poulpy-mhe`, `enable-core`), which CI's portable lane enables: `impl_mhe_reference_full!` on `FFT64Oracle`, `NTT4x30Oracle` and `ControlledSamplingFFT64Oracle`, `mhe_backend_test_suite!` on the first two and `mhe_parity_test_suite!` of `NTT4x30Oracle` against `FFT64Oracle`.
  `FFT64Portable` and `NTT4x30Portable` run `mhe_parity_test_suite!` against `ControlledSamplingFFT64Oracle`, and `FFT64PortableRayon` and `NTT4x30PortableRayon` against their serial base, in place of `mhe_backend_test_suite!`, which runs on the oracle and the portable serial types only.
- The functional suites run on the oracle and the serial `poulpy-cpu-portable` types only; the other types run parity:
  - the AVX2, AVX-512, IFMA and NEON types and their Rayon variants no longer register `ckks_backend_test_suite!`, nor the AVX2, AVX-512 and NEON ones `bin_fhe_backend_test_suite!`;
  - `bin_fhe_backend_test_suite!` is registered on `NTT4x30Portable` (`tests::bin_fhe_ntt4x30`), `FFT64Oracle` (`tests::bin_fhe_fft64`) and `NTT4x30Oracle` (`tests::bin_fhe_ntt4x30`);
  - `blas_backend_test_suite!` is registered on `FFT64Portable` in `f64` and `NTT4x30Portable` in `f64` and `Quad` (`tests::blas_tests::blas_backend_<type>_<scalar>`).
- Every standard-ring CKKS type, portable, Rayon, AVX2, AVX-512, IFMA, NEON and oracle, registers `ckks_determinism_test_suite!` under `ckks_determinism_<type>` (`portable::tests::ckks::determinism_<family>` in `poulpy-cpu-rayon`), which the `determinism` filters of the native AVX-512 and macOS jobs select.
  The portable Rayon types gain the fixtures, and every type runs them in `f64` and `Quad`, where the FFT64 types, the AVX2 types and `NTT3x42IfmaRayon` ran `f64` only.

### `poulpy-core`

- Add `gglwe_copy` and `ggsw_copy` (`GGLWECopy`, `GGSWCopy`, `GGLWECopyImpl`, `GGSWCopyImpl`, `impl_gglwe_copy_reference_full!`, `impl_ggsw_copy_reference_full!`), the copy of the first rows of a GGLWE or a GGSW into one of the same gadget and as many or fewer digits, each row as `glwe_copy` copies it.
  A key never covers its own layout but covers such a copy, which the `_assign` key switches and automorphisms switch in place; an automorphism key's Galois element is not copied.
- `GLWESecretViewMut` implements `GetDistributionMut`, as the other secret views do, so a secret carved from scratch can be sampled.
- Add `ModuleCoreCompressedAlloc::ggsw_compressed_bytes_of_from_infos`, the backend-routed byte count of the buffer `ggsw_compressed_alloc_from_infos` allocates.
  **Breaking:** an out-of-tree implementor of `ModuleCoreCompressedAlloc` implements it.
- **Breaking, behaviour:** every gadget product with a GGLWE key decomposes its input exactly ([#451](https://github.com/poulpy-fhe/poulpy/issues/451)): `glwe_keyswitch`, `glwe_keyswitch_assign`, `glwe_keyswitch_dft`, the `glwe_automorphism` operations, `glwe_tensor_relinearize`, the one-pass products (`glwe_mul_relinearize`, `glwe_square_relinearize`, their assign, prepared and summed forms), the linear transformations, trace, packing, the LWE conversions and key switch, and `ggsw_expand_row` panic on an input wider than the key's `max_input_k`, where the product dropped the input digits past the key's rows.
  A caller drops those bits by narrowing the input first (`SetK::set_k`). The GGLWE and GGSW key switches and automorphisms switch rows of `dnum * k_digit + k_aux` bits, so their key covers that width, guard included.
  `layouts::assert_key_covers` is the check for a key passed directly, and `test_suite::noise::assert_refuses_uncovered_input` checks the refusal of an input one bit past the key in the keyswitch, automorphism and tensor noise tests and in the relinearization, one-pass product and linear transformation parity tests.
- **Breaking:** `GetAutomorphismKey::get_automorphism_key` and `GetTensorKey::get_tensor_key` return `CoreError::GGLWEKeyUse` for a key that does not cover the width asked for; a `GetTensorKey` implementor provides `lookup_tensor_key`, whose answer `get_tensor_key` checks, as `GetAutomorphismKey` does.
- **Breaking:** GGLWE keys take a row offset `k_offset` in bits: row `q` carries `2^-(k_offset + (q + 1) * k_digit)`, so a key switch with such a key lowers its input by `k_offset` bits, over the balanced lift of its digits.
  The result is `2^-k_offset (b + Σ aᵢ sᵢ)` with the integer part of that sum kept, as a modulus raise needs it.
  The key's precision is `k_offset + dnum * k_digit + k_aux`, `max_input_k` is `min(dnum * k_digit, k - k_offset - k_digit)`, and an offset of 0 gives every key as before, bit for bit.
  `GGLWEInfos` gains the required method `k_offset`; `GGLWELayout` and `GLWESwitchingKeyLayout` gain a `k_offset` field, and `GadgetProductOutputSizeParams` one too. `GLWEAutomorphismKeyLayout` has none, no automorphism taking a key with an offset, and its conversion from a `GGLWELayout` panics on one.
- **Breaking, format:** the serialized `GGLWE` and `GGLWECompressed`, and every key built on them, write `k_offset` as a `u32` after `k_digit`.
  Keys written before this change no longer read back.
- **Breaking:** `gadget::gadget_row_plaintext` takes `k_offset` after `k_digit`.
- **Breaking:** `GLWEKeyswitch` and `GLWEKeyswitchImpl` gain `glwe_keyswitch_dft` and `glwe_keyswitch_dft_tmp_bytes`: the gadget product of a key switch, its mask columns decomposed and multiplied by the key, which is the whole switch but for the body, left in the DFT domain.
  They gain `glwe_keyswitch_dft_add` and `glwe_keyswitch_dft_add_tmp_bytes` too, which add that product to an accumulator, so that several switches into one output are summed before a single inverse transform; `GLWEKeyswitchImpl` derives both from `glwe_keyswitch_dft` and the selected `vec_znx_dft_add_assign`.
  The caller finishes each column with `vec_znx_idft_normalize`, the sum of the bodies lowered by the key's offset as the addend of column 0.
  The input is at the key radix, and `impl_glwe_keyswitch_reference_full!` provides the reference of the product.
- `glwe_keyswitch` takes keys with a row offset, from any input radix, for an input of at most the key's `max_input_k` bits, at any output width.
  The other consumers of a GGLWE key in `poulpy-core` panic on one, through the new `layouts::assert_no_row_offset`: the assign and automorphism key switches, the hoisted baby rotation, the lazy giant step, `ggsw_expand_rows`, the LWE conversions and key switch, `gglwe_keyswitch`, `ggsw_keyswitch`, the tensor relinearization and `glwe_mul_relinearize`, and the encryption of tensor keys.
  `gglwe_keyswitch`, `glwe_automorphism_key_automorphism` and `gglwe_external_product` panic when the result's offset differs from the input's.
- `core_backend_test_suite!` gains `glwe_keyswitch_row_offset`, which checks an offset switch against the input lowered with `glwe_rsh` and switched by an offset-free key, `glwe_keyswitch_dft_add`, which checks a DFT sum of switches against the sum of single switches with the advertised scratch, and `glwe_keyswitch_row_offset_refusals`, which checks the refusals above.
  The row-placement test, the GGLWE and switching key encryption tests, plain and compressed, and the key-switch parity sweep cover keys with an offset.
- **Fix:** the one-pass products and squares (`glwe_mul_relinearize`, `glwe_square_relinearize`, their assign, prepared and summed forms) against a tensor key whose gadget covers only the operands' limbs.
  At an offset that is not a whole number of limbs, `d2 = a1·b1` spans one limb more than the operands and is relinearized `cnv_offset mod base2k` bits under the result: the key switch dropped the limb past the gadget, and that error and the key-switching error were both lifted by `2^(cnv_offset mod base2k)`.
  A product at its operands' width with a large remainder lost up to 26 bits of RMS precision on NTT backends, 29.5 with a dense secret (base2k 52, `N = 2^15`, a result 28 bits narrower, `cnv_offset mod base2k = 51`).
  When the offset is not a whole number of limbs and `d2` spans digits past the gadget, the one-pass path is kept only if its modeled variance exceeds the two-step path's by at most `2^-4` of it (`ONE_PASS_VARIANCE_SLACK`, about 0.04 bits of standard deviation); otherwise the columns are aligned to the result first and relinearized as a tensor by the backend's `glwe_tensor_relinearize`, which gives the two-step product bit for bit if and only if `cnv_offset >= min(k_a, k_b)`, as every CKKS product has it, and agrees with it up to rounding otherwise.
  A gate at two bits of standard deviation kept products that lost about 2 bits of RMS precision against the two-step path ([#461](https://github.com/poulpy-fhe/poulpy/issues/461)); the equal-variance gate loses at most 0.05 bits, and the tensor path costs about 21% for a product and 25% for a square.
  The model, `var_noise_glwe_mul_relinearize` in `reference::noise`, shares the operands' term between the paths and differs in the relinearization: `Var_ks + Var_tail` of `d2` at the tensor width plus the tensor rounding for the two-step path, `2^(2·(cnv_offset mod base2k))` times those of `d2` as the convolution leaves it for the one-pass path, where `Var_tail` is the residue past the gadget against `s²`.
  The secret's second moment and the key error come from the tensor key's noise metadata, and the operands' errors are floored at their rounding at their width, which vanishes once the product is lifted past it; a key without noise metadata, or with a secret of unknown law, takes the tensor path there.
  The model counts the rounding at the result and tensor widths as uniform over an ulp at every width, where `var_canonicalization` counts none at a whole number of limbs and a half ulp otherwise, so the decision does not flip with the alignment of `res_k`.
  The scratch of these products covers the backend's tensor path.
- **Breaking:** the gadget key geometry is computed from a digit width in bits, `k_digit`, instead of a digit size in limbs ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  `key_k` drops its `base2k` argument and returns `dnum * k_digit + k_aux`, and `key_size`, `key_work_size` and `GGLWELayout::dnum_for_input` take `k_digit`, the latter without `base2k`.
  `GGLWEInfos` and `GGSWInfos` gain `k_digit()`, which `k_digit_of` computes from a digit size, and the guard rule becomes `k_aux >= k_digit`.
  The noise models and the CKKS key validation read `k_digit`, and every value is unchanged for whole-limb digits.
- **Breaking:** gadget layouts store their digit width in bits ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  The `dsize` field of every key and GGSW layout, and the `dsize` argument of their allocators and `bytes_of`, become `k_digit: TorusPrecision`.
  `k_digit()` is the required method of `GGLWEInfos` and `GGSWInfos`, and `dsize()` is derived from it as `ceil(k_digit / base2k)`, the sub-limb count of the interleaved-digit product.
  Coarser views take a width: `with_dsize` and `at_dsize` become `with_k_digit` and `at_k_digit`, and `SetGGLWEInfos::set_dsize` becomes `set_k_digit`.
  A layout with `k_digit = dsize * base2k` gives the same keys, products and noise values as before.
- **Breaking, format:** serialized GGLWE and GGSW keys, compressed or not, store `k_digit` in place of `dsize`, so keys written before cannot be read back.
- `gadget::gadget_row_plaintext` writes the plaintext of gadget row `row` at bit `(row + 1) * k_digit`, on or off a limb boundary ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  Secret-key GGLWE and GGSW encryption, compressed or not, public-key GGSW encryption and the GGLWE and GGSW noise references place their rows through it, bit for bit as before when `k_digit = dsize * base2k`.
  The scratch of the GGLWE and GGSW noise references covers `vec_znx_normalize_tmp_bytes`.
- **Breaking:** the reference GGLWE and GGSW encryption traits require `BE: HalVecZnxImpl` in place of `VecZnxZero`, `VecZnxAddScalarAssign` and `VecZnxNormalizeAssign`.
- **Fix:** the external product, the off-limb digit layout and the key-switch body read only the limbs an operand's precision covers, `gadget::precision_limbs`, as their scratch queries count them.
  An operand whose buffer holds more limbs than its `k` made the external product transform them, past the scratch it advertised, and the off-limb layout and the key-switch body took them into the result.
  The external-product and key-switch parity suites also run such an operand.
- **Fix:** key switching and external products take an input of precision 0 with keys of any digit width, as with whole-limb digits: the digit layout of an input holding no limb is empty, and `precision_limbs` returns `None` for it.
  The key-switch and external-product parity suites also run such an input at every swept width.
- **Fix:** the LWE switching, GLWE-to-LWE and LWE-to-GLWE key allocators and size queries, host, compressed and prepared, and the reference LWE switching-key encryption check `k_digit == base2k` in place of `dsize() == 1`.
  A digit below a limb also has `dsize() == 1`, so such a layout passed and the host allocators built the key at `k_digit = base2k` while the queries sized the requested width.
- `reference::keyswitching::glwe::gadget_known_zero_prefix` gives the leading operand limbs that are zero for an input below `2^-(zero_bits + 1)`: the zero input limbs for whole-limb digits, and otherwise the top digits ending at least one bit above `zero_bits`, since the gadget rounding can carry one unit into the digit that ends at it.
- Add `From<Base2K> for TorusPrecision`, the width of a one-limb digit, and use it where a layout builds `k_digit = base2k`.
- The `transfer_into` asserts of GGLWE and GGSW, and the `Display` of their compressed forms, name `k_digit` where they compare or print it.
- `layouts::digits_aligned` and the provided `GGLWEInfos::digits_aligned` and `GGSWInfos::digits_aligned` tell whether the digits are whole limbs, so that a gadget product reads the input limbs as its operand.
  A view decides on the width it reads, `k_digit`, which sets the operand of its products.
- GGSW row expansion, and so GGSW key switching and automorphisms, takes tensor keys of any digit width: each mask column, normalized at the key radix, is laid out on the digit grid with `gadget_dft_apply`, and `ggsw_expand_rows_tmp_bytes` covers that operand.
  The GGSW key-switch noise tests, in place or not, and the gadget parity suite also run widths off a limb boundary.
- Key switching, automorphisms, traces and external products run with digits of any width ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  Keys whose digits are whole limbs read the input limbs as before, bit for bit.
  Other keys lay the input out on the digit grid with `vec_znx_gadget_decompose`, `ceil(k_digit / base2k)` sub-limbs per digit, and run the same interleaved product on that operand.
  `gadget::gadget_input_size`, `gadget_dft_apply` and `gadget_dft_apply_tmp_bytes` build that operand, and the key-switch and external-product scratch covers it.
- **Breaking:** `GLWEKeyswitchInternal` and `GLWEExternalProductDftReference` require `VecZnxGadgetDecompose` and `VecZnxGadgetDecomposeTmpBytes`.
- The key-switch, automorphism and external-product parity suites also sweep digit widths off a limb boundary.
- Relinearization, the one-pass multiply-and-relinearize and the linear transformations, hoisted baby steps and lazy giant steps included, run with tensor and automorphism key digits of any width, through the same operand layout as the key switch ([#188](https://github.com/poulpy-fhe/poulpy/issues/188)).
  The hoisted baby steps lay the input out once for the whole key set, and so take that route when every key has whole-limb digits or all share one digit width and digit count.
- The relinearization and linear-transformation parity suites also sweep digit widths off a limb boundary.
- **Breaking:** `ParityShapes` gains `unaligned_digits`, true in `Default`, which sweeps those widths (a sub-limb width and one bit past and short of a limb boundary) at rank 1 only; the Rayon parity suites of the AVX2, AVX-512 and NEON backends turn it off.
- `test_suite::noise::test_glwe_keyswitch_k_digit` and `test_glwe_external_product_k_digit`, registered in `core_backend_test_suite!`, run a key switch and an external product at every digit width from 2 bits to three limbs and check the output noise, and the gadget term alone, against the noise model to within one bit.
- The getting-started guide, the performance guide, the bootstrapping and linear-transformation docs, the CKKS README and the MHE contracts describe gadget digits by their width in bits, `k_digit`.
- **Breaking:** `GLWE`, `GLWEPlaintext` and `GLWECompressed` carry an optional, scheme-managed `GLWEPlaintextMeta` (`Scale`, `SlotsKind`, `log_sparsity`) through `GLWEPlaintextInfos` and `SetGLWEPlaintextInfos`, which also read and set each field; `SlotsKind` moves from `poulpy-ckks` and gains `Integer`.
- **Breaking, format:** serialized `GLWE` and `GLWECompressed` carry their plaintext metadata.
- `reference::keyswitching::glwe::gglwe_product_limbs` returns the `product_limbs` the reference key switch passes to `GGLWEProductDigitsStridedImpl`, so that backend overrides derive the same.
- The scratch of GLWE encryption, decryption, plaintext multiplication, tensoring, relinearization, linear transformations and GGSW noise covers `vec_znx_idft_apply_tmp_bytes`.
- **Breaking:** `LinearTransformationImpl` requires `glwe_eval_linear_transformation_sum_into` and `glwe_eval_linear_transformation_sum_tmp_bytes`; a hand-written impl forwards them to `glwe_eval_linear_transformation_sum_into_reference` and `glwe_eval_linear_transformation_sum_tmp_bytes_reference`.
- Add `GLWELinearTransformations::glwe_eval_linear_transformation_sum_into` and its `_tmp_bytes`: `Σ_t M_t(a_t)` over resident transforms that share one giant schedule, at the giant key-switch cost of one transform.
  It takes the baby caches `babies[t]` of the `a_t` and the transforms `lts[t]` as two slices of one length, so a caller passes its own without building pairs.
- `layouts` exports the checks the linear-transformation evaluations run before reading a key, `assert_linear_transformation_into` and `assert_linear_transformation_sum`, so that overrides fail as the reference does.
- **Breaking:** `GLWETensoringImpl` requires `glwe_mul_relinearize_sum`, `glwe_mul_prepared_relinearize_sum`, `glwe_tensor_apply_prepared` and their `_tmp_bytes`; a hand-written impl forwards them to the `GLWETensoringReference` methods of the same names with the `_reference` suffix.
- Add `GLWEMulRelinearizeSum::glwe_mul_relinearize_sum` and its `_tmp_bytes`: `relinearize(Σ_i a[i] * b[i])` in one pass, over two slices of one length of rank-1 operands, whose pairs share the first pair's layouts.
  Each pair's product columns are added to the first pair's in the transform domain, and only the key-switched column of the sum comes back to digits before the single final normalization, so the scratch does not grow with the number of pairs.
  One pair is `glwe_mul_relinearize`, bit for bit.
  The products of every pair accumulate before one inverse transform: pick `base2k` for `a.len()` times the products of one pair (`Module::max_base2k`), up to `2 * min(a_size, b_size)` per output limb in the cross column.
- Add `glwe_prepare_left`, the left-hand mirror of `glwe_prepare_right`, and the queries `glwe_prepare_left_tmp_bytes` and `glwe_prepare_right_tmp_bytes`.
- **Breaking, behaviour:** `glwe_prepare_right` panics unless `b_prep` has `ceil(b_k / base2k)` limbs, as `glwe_prepare_left` does, so that a prepared operand holds its ciphertext at width `b_k`.
- Add `GLWEPreparedTensoring`: `glwe_mul_prepared_relinearize_sum` and `glwe_tensor_apply_prepared`, with their `_tmp_bytes`, are `glwe_mul_relinearize_sum` and `glwe_tensor_apply` over operands prepared once, which transform no operand in the call.
  `glwe_mul_prepared_relinearize_sum` takes its left and right operands as two slices of one length, as `glwe_mul_relinearize_sum` does.
  Operands prepared from ciphertexts of width `k` on `ceil(k / base2k)` limbs, under either `PrepareHint`, give the unprepared products of those ciphertexts bit for bit.
  Both check every prepared operand against the degree of `res`, its rank and the given sizes before writing `res`, and the summed product checks that each given size is the limb count of its width.
  Their queries count only what the prepared products use, and one prepared pair needs no more than `glwe_mul_relinearize_tmp_bytes`.
- **Fix:** `glwe_eval_linear_transformation_into` checks every giant step before writing `res`, as `glwe_eval_linear_transformation_sum_into` does: on the per-giant fallback, a giant step after the first that had no diagonal, started with a diagonal of another width, or read a baby rotation missing from the cache panicked after `res` was written. The other diagonals are not checked: every diagonal of a transform shares one width, scale and degree, as the encoder produces them. Valid calls are unchanged.
- **Fix:** `glwe_tensor_relinearize_tmp_bytes` counts the operand's second conversion to the tensor key's radix where `glwe_tensor_relinearize` takes it, while the key-switched sum is held. It counted a conversion beside the key product instead, and only for a key of another radix than the output's, so with a key of another `base2k` than the operand's the relinearization could run out of the scratch its query returned and panic, as `ckks_mul_into` did at rank 2 with a key of radix 12 on radix-52 ciphertexts. The query changes only for keys of another radix than the operand's or the output's, and bits are unchanged.
- **Breaking:** `GLWEPlaintext::encode_vec_i64_strided`, `encode_vec_i128_strided`, `decode_vec_i64_strided` and `decode_vec_i128_strided` are removed with the gap placement they served; the dense forms take one value per coefficient.
- `GLWEMulPreparedPlainSum::glwe_mul_prepared_plain_sum` computes `res = Σ_t a[t] ⊛ σ_{galois[t]}(b[t])` over parallel slices of ciphertexts prepared as left operands, plaintexts prepared as right operands and odd Galois elements, at given limb counts and radix: one `cnv_apply_dft_sum` per column in slice order, then one normalization, canonical at `res.k()`.
  It transforms no operand, so prepared ciphertexts serve every output that shares them, and `glwe_mul_prepared_plain_sum_tmp_bytes` does not grow with the number of terms. It checks every operand against the degree and rank of `res`, every element odd, and its scratch, before writing `res`. One term through `1` is `glwe_mul_plain` of the sources bit for bit.
  Backends get it from `impl_operations_reference_full!`, or from `impl_glwe_mul_prepared_plain_sum_reference_full!` with the OEP family `GLWEMulPreparedPlainSumImpl` and its reference `GLWEMulPreparedPlainSumReference`.
- Add `GGLWELayout::max_input_k`, the widest input a key switches whole: what its digits cover, and no more than its precision less one digit.
- Add `From<GGLWELayout>` for `GLWEAutomorphismKeyLayout`, `GLWESwitchingKeyLayout` and `GLWETensorKeyLayout`. Each panics on a coarsened view.
- Add `GGLWECompressed::data` and `data_mut`, the stored bodies, as on `GGLWE`.
- **Fix:** `glwe_secret_tensor_prepare` takes every temporary from its scratch, where it allocated the transformed secret, the products and their normalization buffer, and wipes that scratch before returning, so no secret-derived buffer is freed unwiped ([#425](https://github.com/poulpy-fhe/poulpy/issues/425)).
  `glwe_secret_tensor_prepare_tmp_bytes` grows by those buffers, and the tensor key and GGLWE-to-GGSW key encryptions reuse that scratch for the encryption, so their queries count the larger of the two phases instead of their sum.
- **Fix:** `glwe_eval_linear_transformation_tmp_bytes` counts the scratch of the per-column and final BIG normalizations and of the final iDFT of the lazy giant-step path, and of the BIG normalization of the per-giant fallback ([#426](https://github.com/poulpy-fhe/poulpy/issues/426)).
  The lazy path counts its phases as sequential instead of summing them, so the query can shrink; outputs are unchanged.
- Add `glwe_switch_degree` (`GLWESwitchDegree`), the switch of a GLWE between ring degrees that divide one another, column by column, beside the conjugate-invariant maps: `GLWECIConversionImpl` gains it with a default body.
- Add `layouts::max_lazy_terms(base2k, coefficients)`, the most canonical digits a lazy sum holds before it must be normalized: the larger of the worst case `2^(63 - base2k)` and the average-case count whose Gaussian sum leaves the normalizer's domain `[-2^62, 2^62]` with probability at most `2^-128` over the coefficients, for independent centered uniform digits (`docs/base2k-failure-probability.md`).
- **Breaking:** `GLWETrace` traces over a window of levels in either direction, summed or normalized: `glwe_trace` and `glwe_trace_assign` take `levels: Range<usize>`, `reverse` and `normalize` in place of `skip`, which was `skip..log2(n)`, forward and normalized, and `glwe_trace_galois_elements` takes the levels and the direction.
  Level 0 is the conjugation, on the standard ring only, and level `i` the rotation by `2^(i-1)` slots, so a full trace ends at `log2(n)` on the standard ring and at `log2(n) + 1` on the conjugate-invariant ring, whose backends now implement `GLWETraceImpl`, part of `impl_core_reference_full!`.
  An out-of-tree backend calling `impl_core_reference_full!` drops its own `impl_glwe_trace_derived_full!`, which now conflicts, and a hand-written `GLWETraceImpl` takes the new parameters.
  `reverse` visits the levels from the last down, each by its inverse element, and a nonempty trace leaves its output canonical, summed or normalized.
  The test suite adds `test_glwe_trace_sum` and `test_glwe_trace_parity`, which takes over the trace cases of `test_trace_packing_parity`, runs on both rings and covers summed and reversed windows: register it beside `test_trace_packing_parity`.
- `docs/layering-contract.md` states the layering rules that `core-contracts.md` restated, and the `# Canonical flag` doc of `GLWE` names the exception of a documented canonical precondition.
- **Breaking:** `GLWERotateImpl` and `GLWEMulXpMinusOneImpl` require `Backend<Ring = Standard>`, and through them `GGSWRotateImpl`, `GLWEPackImpl` and `ConversionImpl`: these families compose the HAL monomial family, so out-of-tree backends implement them on standard-ring types only.
- **Breaking:** `EncryptionParityBackend` no longer includes `ConversionImpl` and stays ring-independent: `test_glwe_encryption_parity` drops its `GLWEExpandLWEMatrix` bound and its LWE-matrix decryption case, which moves to the new `test_lwe_matrix_decrypt_parity`, bounded on the standard ring and registered by `core_encryption_parity_test_suite!` as `lwe_matrix_decrypt`.
- **Breaking:** the helpers that code outside `reference` also calls move out of it with their signatures and messages, and `reference` no longer exports them: `assert_no_row_offset` and `assert_key_covers` (`reference::keyswitching::glwe`), `assert_linear_transformation_into` and `assert_linear_transformation_sum` (`reference::linear_transformation` and the crate root) and `DiagonalProd` (`reference::linear_transformation`) move to `poulpy_core::layouts`, `DiagonalProd` staying at the crate root.
  `gadget_input_size`, `precision_limbs`, `gadget_dft_apply`, `gadget_dft_apply_tmp_bytes`, `gglwe_product_output_size` and `gglwe_product_accumulation_output_size` (`reference::keyswitching::glwe`), `prod_dft_size`, `cnv_offset_to_limb_offset` and `normalize_input_limb_bound_with_offset` (`reference::operations`) and `gadget_row_plaintext` (`reference::encryption`) move to the new module `poulpy_core::gadget`, re-exported at the crate root.
  `BSGSOps` moves from `reference::polynomial_evaluation` to `poulpy_core::api`, still at the crate root.
  `poulpy_core::oep` no longer re-exports `GLWEAutomorphismReference`, `GGLWEAutomorphismReference`, `ConversionReference`, `GGSWConversionReference`, `DecryptionReference`, `GLWEExternalProductReference`, `GLWEKeyswitchReference` and `LWEKeyswitchReference`, which callers import from their `poulpy_core::reference` modules.
  Callers and backends update their imports; outputs are unchanged.
- **Breaking:** every opt-in macro is named after its OEP trait: `impl_glwe_tensoring_reference!` becomes `impl_glwe_tensoring_reference_full!`, `impl_gglwe_product_digits_strided_reference!` becomes `impl_gglwe_product_digits_strided_reference_full!` and `impl_glwe_packing_derived_full!` becomes `impl_glwe_pack_derived_full!` (`GLWEPackImpl`).
  Backends rename their invocations; `impl_core_reference_full!` invokes the new names.
- **Breaking:** decompression dispatches through the new OEP family `DecompressionImpl`, whose methods have the API names (`decompress_glwe`, `decompress_gglwe`, `decompress_ggsw`, `decompress_glwe_public_key`, `decompress_lwe`, `decompress_automorphism_key`, `decompress_glwe_switching_key`, `decompress_tensor_key`, `decompress_glwe_to_lwe_key`, `decompress_lwe_switching_key`, `decompress_lwe_to_glwe_key`, `decompress_gglwe_to_ggsw_key`).
  `decompress_glwe` and `decompress_lwe` are required, with reference bodies `decompress_glwe_reference` and `decompress_lwe_reference` (`reference::decompression::DecompressionReference`), and `impl_decompression_reference_full!` selects them; the other ten are derived defaults composing `decompress_glwe`, `decompress_gglwe` and `decompress_glwe_switching_key` through the selected backend, so an override of `decompress_glwe` reaches every gadget and key decompression. The bodies are those the traits provided, and outputs are unchanged.
  `GLWEDecompress`, `GGLWEDecompress`, `GGSWDecompress`, `GLWEPublicKeyDecompress`, `LWEDecompress`, `GLWEAutomorphismKeyDecompress`, `GLWESwitchingKeyDecompress`, `GLWETensorKeyDecompress`, `GLWEToLWESwitchingKeyDecompress`, `LWESwitchingKeyDecompress`, `LWEToGLWEKeyDecompress` and `GGLWEToGGSWKeyDecompress` move from `poulpy_core::layouts` to `poulpy_core::api` (and the crate root) and become API traits generic over the backend, `<BE>` in place of the `Backend` associated type, with no supertraits, implemented on `Module<BE>` for `BE: DecompressionImpl`.
  Callers bound `Module<BE>: GGSWDecompress<BE>` where they bounded `GGSWDecompress + GLWEDecompress<Backend = BE>`, and so for each trait; the calls and outputs are unchanged.
  `impl_core_reference_full!` invokes the new `impl_decompression_reference_full!`; out-of-tree backends that list the core families add it, and `EncryptionParityBackend` gains `DecompressionImpl`.
- **Breaking:** the preparations of the stored layouts dispatch through the new OEP family `PreparationImpl`, whose methods have the API names: `glwe_prepare`, `gglwe_prepare`, `ggsw_prepare`, `ggsw_zero`, `glwe_public_key_prepare`, `glwe_automorphism_key_prepare`, `glwe_switching_key_prepare`, `glwe_tensor_key_prepare`, `glwe_to_lwe_key_prepare`, `lwe_switching_key_prepare`, `lwe_to_glwe_key_prepare`, `gglwe_to_ggsw_key_prepare` and `glwe_ci_key_prepare`, and their `_tmp_bytes` queries (`ggsw_zero` takes none).
  The GLWE, GGLWE, GGSW, public-key and GGLWE-to-GGSW-key preparations and their queries are required, with reference bodies in `reference::preparation::PreparationReference`, and `impl_preparation_reference_full!` selects them; `ggsw_zero` is a derived default over the selected `vmp_zero`, `PreparationImpl` having `HalVmpImpl` as a supertrait, and the automorphism, switching, tensor, GLWE-to-LWE, LWE switching, LWE-to-GLWE and CI key preparations and their queries are derived defaults over `gglwe_prepare` and `glwe_switching_key_prepare`, so an override of `gglwe_prepare` reaches every key preparation, but an override of `gglwe_prepare_tmp_bytes` no longer affects `gglwe_to_ggsw_key_prepare_tmp_bytes`. The bodies are those the factories provided, and outputs are unchanged.
  `GLWEPreparedFactory`, `GGLWEPreparedFactory`, `GGSWPreparedFactory`, `GLWEPublicKeyPreparedFactory`, `GLWEAutomorphismKeyPreparedFactory`, `GLWESwitchingKeyPreparedFactory`, `GLWETensorKeyPreparedFactory`, `GLWEToLWEKeyPreparedFactory`, `LWESwitchingKeyPreparedFactory`, `LWEToGLWEKeyPreparedFactory`, `GGLWEToGGSWKeyPreparedFactory` and `GLWECIKeyPreparedFactory` move from `poulpy_core::layouts` and `poulpy_core::layouts::prepared` to `poulpy_core::api` (and the crate root), keep their names and allocation and byte-count methods, provided by the traits, and are implemented on `Module<BE>` for `BE: PreparationImpl`; they no longer have HAL operation supertraits (`VmpPrepare`, `VmpPrepareTmpBytes`, `VmpZero`, `VecZnxDftApply`, `GLWENormalize`, `GLWEBytesOf`), and `GGLWEToGGSWKeyPreparedFactory` gains `GGLWEPreparedFactory` as its supertrait.
  Callers import the factories from `poulpy_core::api` in place of `poulpy_core::layouts` or `poulpy_core::layouts::prepared`; callers that used a factory through the blanket implementation of another one bound that factory itself (`GLWECIKeyPreparedFactory` beside `GLWESwitchingKeyPreparedFactory`), and callers relying on a removed supertrait bound the HAL trait; the calls and outputs are unchanged.
  `impl_core_reference_full!` invokes the new `impl_preparation_reference_full!`; out-of-tree backends that list the core families add it, and `EncryptionParityBackend` gains `PreparationImpl`.
- **Breaking:** `GLWETensorKeyPreparedFactory::prepare_tensor_key` and `prepare_tensor_key_tmp_bytes` become `glwe_tensor_key_prepare` and `glwe_tensor_key_prepare_tmp_bytes`, with the same arguments, the object-first names of the other preparations: callers rename the calls.
- **Breaking:** sampling, preparation, tensoring and LWE/GLWE conversion of secrets dispatch through the new OEP family `SecretImpl`, with supertraits `SamplingImpl` and `HalVecZnxImpl`, whose methods have the API names: the six `glwe_secret_fill_*` and six `lwe_secret_fill_*` fills, `glwe_secret_prepare`, `glwe_secret_tensor_prepared_prepare`, `glwe_secret_tensor_prepare` and its `_tmp_bytes` query, `glwe_secret_from_lwe_secret`, and `lwe_secret_from_glwe_secret` and its `_tmp_bytes` query.
  The ternary and binary fills are derived defaults drawing each column, in column order, through the selected `scalar_znx_fill_distribution`, and the zero fills derived defaults over the selected `vec_znx_zero`; the preparations, the tensoring, the conversions and their queries are required, with reference bodies in `reference::secret::SecretReference`, and `impl_secret_reference_full!` selects them. The bodies are those the traits provided, and draws and outputs are unchanged.
- **Breaking, behaviour:** `SecretConversion::lwe_secret_from_glwe_secret` takes a `scratch` argument last and carves its rotated GLWE secret there in place of the heap, wiping it before returning; the new `SecretConversion::lwe_secret_from_glwe_secret_tmp_bytes` sizes it from the source secret's layout. Callers pass an arena of at least that size; outputs are unchanged.
  `GLWESecretSampling`, `LWESecretSampling`, `GLWESecretTensorFactory` and `SecretConversion` move from `poulpy_core::layouts`, and `GLWESecretPreparedFactory` and `GLWESecretTensorPreparedFactory` from `poulpy_core::layouts` and `poulpy_core::layouts::prepared`, to `poulpy_core::api` (and the crate root); they keep their names and are implemented on `Module<BE>` for `BE: SecretImpl`. The two prepared factories keep their allocation and byte-count methods, provided by the traits over `SvpPPolAlloc` and `SvpPPolBytesOf`, and `GLWESecretPreparedFactory` no longer has `SvpPrepare` as a supertrait.
  Callers import the traits from `poulpy_core::api`, and generic code that sampled or prepared a secret through the former blanket implementations bounds `Module<BE>` by the trait or `BE` by `SecretImpl`; the calls and outputs are unchanged.
- **Breaking:** `GLWESecretCIEmbed` moves from `poulpy_core::layouts` to `poulpy_core::api` and dispatches, with `GLWECIKeyEncryptSk`, through `GLWECIConversionImpl`, which gains `EncryptionImpl` as a supertrait.
  `glwe_secret_ci_embed` writes the embedding into a caller secret `res` of twice the source degree, `glwe_secret_ci_embed(&mut res, &src)`, in place of returning a new one, and panics before writing it on a degree or rank mismatch; callers allocate `res` (`glwe_secret_alloc_from_infos`).
  `glwe_secret_ci_embed` is required, with reference body `reference::ci_conversion::glwe_secret_ci_embed_reference`, which `impl_glwe_ci_conversion_reference_full!` selects; `glwe_ci_key_encrypt_sk_tmp_bytes`, `glwe_ci_embed_key_encrypt_sk` and `glwe_ci_trace_key_encrypt_sk` are derived defaults over it and the selected `glwe_switching_key_encrypt_sk`.
  Both traits are implemented on `Module<BE>` for `BE: GLWECIConversionImpl`, in place of any module with `HalVecZnxCIImpl` and the encryption API; outputs are unchanged.
- **Breaking, behaviour:** `glwe_ci_embed_key_encrypt_sk` and `glwe_ci_trace_key_encrypt_sk` carve the embedded conjugate-invariant secret from their scratch in place of the heap and wipe their whole budget before returning; `glwe_ci_key_encrypt_sk_tmp_bytes` grows by that secret, at the key degree and the larger of its input and output ranks. Callers sizing their arena from the query are unchanged; outputs are unchanged.
- **Behaviour:** `glwe_ci_embed_key_encrypt_sk` and `glwe_ci_trace_key_encrypt_sk` refuse, before `sk_ci` reaches the scratch, a conjugate-invariant secret of more than half the key degree or of another rank than its side of the key, a secret wider than the key or of another rank than its side, and, for the embed key, a secret of the `NONE` law, with static messages "conjugate-invariant key encryption: <reason>", where the nested embedding and switching-key encryption refused them after the embedding and skipped the wipe.
  `test_glwe_ci_embed_key_encrypt_sk_rejects_mismatched_secrets` and `test_glwe_ci_trace_key_encrypt_sk_rejects_mismatched_secrets`, registered in `core_backend_test_suite!`, fire each refusal on an empty and on a poisoned arena and check the key, the sources and the arena unchanged.
- The `# Safety` sections of `EncryptionImpl`, `DecryptionImpl` and `GLWECIConversionImpl` state that every method taking scratch (of `GLWECIConversionImpl`, the key encryptions) wipes the scratch its query reports before returning, as the reference and derived bodies do, and their methods are free in scratch use before that wipe.
- **Behaviour:** the default bodies of `glwe_public_key_generate` and `glwe_encrypt_pk_smudged` wipe their whole query before returning, where the normalizations after their last secret-reading call left data in the arena; `test_glwe_encrypt_pk_replay` checks both with `assert_wipes_scratch` on exactly their queries.
- **Behaviour:** the secret-key encryptions (`glwe_encrypt_sk`, `glwe_encrypt_zero_sk`, `glwe_encrypt_sk_with_mask`, `glwe_compressed_encrypt_sk`, `glwe_compressed_encrypt_zero_sk` and the GGLWE, GGSW, switching, automorphism, tensor, GGLWE-to-GGSW, GLWE-to-LWE, LWE-switching and LWE-to-GLWE key encryptions, compressed forms included) refuse a secret of the `NONE` distribution with "glwe secret distribution is NONE (have you prepared the key?)" before any draw or carve, where the GLWE and GGLWE bodies drew masks and the key bodies prepared or rotated the secret into the scratch first.
  `test_gglwe_encrypt_sk_rejects_unset_secret`, registered in `core_backend_test_suite!`, checks the key, both sources and the scratch unchanged.
- **Breaking:** `EncryptionImpl` gains `SecretImpl` as a supertrait, and `glwe_tensor_key_encrypt_sk`, `glwe_tensor_key_compressed_encrypt_sk`, their queries and the `GLWETensorKeyEncryptSk` and `GLWETensorKeyCompressedEncryptSk` delegates lose their `Module<Self>: GLWESecretPreparedFactory + GLWESecretTensorFactory` bound: their derived bodies prepare and tensor the secret through `SecretImpl`.
  `impl_core_reference_full!` invokes the new `impl_secret_reference_full!`; out-of-tree backends that list the core families add it, and `test_suite::noise::TestBackend` gains `SecretImpl`.
- **Breaking:** the core OEP families have as supertraits the families their default bodies dispatch through, in place of `Module<Self>` bounds on their methods: `EncryptionImpl` gains `SamplingImpl`, `HalVecZnxImpl` and `GLWENormalizeImpl`, `GLWEKeyswitchImpl` gains `HalVecZnxDftImpl` and `GLWECIConversionImpl` gains `HalVecZnxImpl`.
  `glwe_encrypt_pk_smudged`, `glwe_public_key_generate`, `ggsw_encrypt_pk` and their queries, `glwe_keyswitch_dft_add` and `glwe_switch_degree` lose their bounds on HAL and core operations, on the OEP method, the delegate and, for `glwe_keyswitch_dft_add`, the `GLWEKeyswitch` method: their default bodies call the selected HAL methods, and outputs are unchanged.
  The `GGLWEExternalProduct` and `GGSWExternalProduct` delegates drop their bound on `GLWEExternalProduct`, which their OEP families imply.
  `AutomorphismImpl` no longer has `GLWEKeyswitchImpl` as a supertrait, which only `GLWEAutomorphismReference` composes, bounding the module by `GLWEKeyswitch`.
  Out-of-tree backends implementing `EncryptionImpl`, `GLWEKeyswitchImpl` or `GLWECIConversionImpl` implement the new supertraits, as every CPU backend does; generic code that bounds `BE: AutomorphismImpl` and also switches keys bounds `BE: GLWEKeyswitchImpl`.
  `GLWEPackImpl` lists `AutomorphismImpl`, `GLWEShiftImpl` and `GLWENormalizeImpl` beside `GLWETraceImpl`, which already implied them.
- **Breaking:** `gadget::gadget_row_plaintext` takes `module: &Module<BE>` with `BE: HalVecZnxImpl` and runs the selected `vec_znx_normalize`, in place of any `module: &M` with `M: VecZnxNormalize<BE>`, so the derived public-key GGSW encryption writes its rows through it on a backend of any coefficient word.
  The reference GGLWE and GGSW encryptions, plain and compressed, and the `GGLWENoise` and `GGSWNoise` implementations bound `BE: HalVecZnxImpl` in place of `Module<BE>: VecZnxNormalize<BE>`; callers pass a `Module<BE>` and bound the backend instead of the module.
- OEP methods take their API method's generics and parameter names: `GLWEExternalProductImpl::glwe_external_product_tmp_bytes` takes `<R, A, B>` and `b_infos`, `glwe_external_product` `lhs` and `rhs`, `glwe_external_product_assign` `a`, `GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes` `<R, A, B>` and `PolynomialEvaluationImpl::glwe_eval_baby_step` `<Ops, V, P, A, G>`, and `GLWEExternalProductInternal::glwe_external_product_dft` spells its output `VecZnxDftBackendMut`; types and order are unchanged, so implementations and callers compile as they are.
- **Breaking:** `glwe_trace_assign_tmp_bytes` (`GLWETrace`, `GLWETraceImpl`) is removed; size `glwe_trace_assign` with `glwe_trace_tmp_bytes(res, res, key)`, which covers both forms. Backends drop their override of the removed method and fold its requirement into `glwe_trace_tmp_bytes`.
- Add `TransferInto` for `GLWEPublicKey`, `GLWEPublicKeyCompressed`, `GLWECompressed` and `GGLWECompressed`: each asserts its layout parameters (`base2k`, `k` or `k_aux`, `k_digit`, `k_offset`, the rank the container does not hold, the seed count) and the container shape before any write, then moves the coefficients and copies the seeds, distribution, noise estimate and plaintext metadata. `test_transfer_into_rejects_*` in `api::transfer` fire one check per layout.
- **Behaviour:** `GGLWE::transfer_into` panics with "transfer_into: GGLWE k_offset" on a destination of another row offset, before any write, where it moved the rows under the destination's offset; `test_transfer_into_rejects_gglwe_offset_mismatch` fires it, and `module_transfer_gglwe_roundtrip` moves a key with an offset.
- **Breaking:** the suite functions `test_glwe_encrypt_pk_imprecise_key`, `test_glwe_encrypt_pk_zero_ephemeral`, `test_glwe_encrypt_pk_nan_ephemeral` and `test_glwe_public_key_rank_zero` become `test_glwe_encrypt_pk_rejects_imprecise_key`, `test_glwe_encrypt_zero_pk_rejects_zero_ephemeral`, `test_glwe_encrypt_zero_pk_rejects_nan_ephemeral` and `test_glwe_public_key_alloc_rejects_rank_zero`, and `test_glwe_mul_plain_compact` becomes `test_glwe_mul_plain_plaintext_degree`; an out-of-tree backend registering them renames its entries.
  The refusal tests of `test_glwe_mul_relinearize_sum_*`, `test_glwe_mul_prepared_relinearize_sum_*`, `test_glwe_tensor_apply_prepared_rejects_mismatched_operands`, `test_glwe_prepare_left_rejects_size_of_another_width`, `test_glwe_prepare_linear_transformation_rhs_rejects_wide_diagonal`, `test_glwe_linear_transformation_*_rejects_*` and the three above that take a destination now catch the panic, compare its full message and check the destination is unchanged, so the registration drops its `#[should_panic]`; only `test_glwe_public_key_alloc_rejects_rank_zero` keeps one.
- **Behaviour:** the GGLWE encryption refusals of `res.rank_in()` against the plaintext columns and `res.rank_out()` against the secret rank, plain and compressed, panic with the static messages "res.rank_in() != pt.cols()" and "res.rank_out() != sk.rank()", where they printed both values; the other encryption refusals keep their messages and no longer print left and right values.
- **Behaviour:** refusal messages on secret paths no longer print operand values: the encryption and decryption rank, dimension and degree refusals, the tensor key encryption refusals and the `TransferInto` refusals of plaintexts and secrets panic with static messages, where the bare comparisons printed left and right values.

### `poulpy-ckks`

- Add the modulus raise that applies the first `CoeffsToSlots` factor on the raised ciphertext while it is small (`DFTPlan::with_first_factor_on_raise`, after ePrint 2025/1403): the factor consumes no modulus, and the raised width drops by one CoeffsToSlots scale.
  Each diagonal's term is rotated with no key and switched to the dense secret by a `GLWESwitchingKey` of its own, from the rotated raise secret, whose rows sit `k_offset` bits under the term's few limbs and end at the raise's working width (`BootstrappingKeysLayout::first_factor_key`, `BootstrappingLinearKeys::first_factor_key`); the switches are summed with `glwe_keyswitch_dft_add` and rounded once.
  The sparse-to-dense switch is not used, nor its key generated.
  `BootstrappingPreset::with_first_factor_on_raise` re-derives a preset with it; it is not combined with EvalRound+.
  A key set whose keys do not fit the raise, by radix, offset, input width or reach, is an error, as is a raise of a ciphertext wider than its keys switch.
  `ckks_bootstrap` and `ckks_functional_bootstrap` check the first-factor keys, and a sparse-to-dense key with a row offset, against the raise they run before any work, so a bootstrap at other widths than its keys were laid out for fails at entry; the raise itself checks every diagonal and key before its first switch.
  `generate_keys` carves each rotated raise secret from its scratch, which it wipes before returning.
  `DFTPlan::with_optimal_bsgs` keeps the one diagonal per giant step of a first factor left to the raise.
- **Breaking:** the sparse-to-dense key is optional, absent exactly when the first factor is applied on the raise: `EncapsulationKeysLayout::sparse_to_dense` is an `Option`, the second key of `BootstrappingKeySet::encapsulation_keys` and `BootstrappingKeysPrepared::encapsulation_keys` too, and `BootstrappingLinearKeys::encapsulation_keys` returns `Option<(&SwitchingKey, Option<&SwitchingKey>)>`; add the alias `EncapsulationKeys`. Key generation rejects a layout whose sparse-to-dense key disagrees with the recipe.
- Add S2C-first bootstrapping with EvalMod once on the conjugate-invariant ring instead of twice on the halves of the slots (`EvalModRing`, `BootstrappingPlan::with_eval_mod_ring`, `BootstrappingPreset::with_eval_mod_ring`), run by `CKKSBootstrappingCIOps::ckks_bootstrap_ci_into` on a context of `BootstrappingContext::compile_ci` (`EvalModCI`) and keys of `BootstrappingContext::generate_keys_ci` and `BootstrappingKeySet::prepare_ci`; `docs/bootstrapping.md` details the conversions.
  Key generation samples the CI secret in scratch, which it wipes before returning.
  The switching keys lift the dense secret to degree `2N` themselves.
  The bootstrap checks every key at entry, the conversion keys and the encapsulation keys included, and carves its CI ciphertexts from the CI scratch, which `ckks_bootstrap_ci_tmp_bytes` counts with the reference stages the bootstrap runs.
  It also checks both arenas against that query at entry for a key store that gives its layouts (`BootstrappingCIKeys::keys_layout`, which `BootstrappingKeysPrepared` gives).
  `compile_ci` runs `compile_ci_unprepared` then `BootstrappingContextUnprepared::prepare`, and generates the DFT matrices as at the bootstrap's degree `N`, not at the degree of its module; `compile_ci_unprepared_tmp_bytes`, `BootstrappingKeysLayout::generate_ci_tmp_bytes` and `BootstrappingKeysLayout::prepare_ci_tmp_bytes` size the arenas of the setup.
  Backends override it through `CKKSBootstrappingCIImpl`, whose associated type `CI` names the conjugate-invariant twin, and which `impl_ckks_bootstrapping_ci_reference_full!($be, $ci)` wires to the reference; `CKKSBootstrappingCIOps::CI` exposes the twin.
  The entry points take one standard module of degree at least `2N`, which serves the conversions and the stages of degree `N`, and the CI module of degree `N`.
- **Fix:** CKKS linear transformations, the first-factor raise and bootstrapping key generation read Galois elements at their operands' degree, not the module's, so a module of a larger degree serves them.
- Add the preset `n16_d35_k685_p19_s2c_ffr_ci` to `presets::bootstrapping::all`: `n16_d35_k733_p20_s2c` with the first factor on the raise and EvalMod on the CI ring, 19 bits of precision and 15 levels, for about 22% less time per bootstrap on the NEON Rayon backend.
  Measured on an Apple M5 Pro: `NTT4x30NeonRayon/ckks/ckks_bootstrapping/n16_d35_k685_p19_s2c_ffr_ci` takes 0.72 s, `NTT4x30NeonRayon/ckks/ckks_bootstrapping/n16_d35_k733_p20_s2c` 0.92 s.
- **Breaking:** `BootstrappingKeys` is split: `BootstrappingLinearKeys` holds the rotation, encapsulation and first-factor keys, `BootstrappingKeys` adds the tensor key of the standard ring, and `BootstrappingCIKeys<BE, BCI>` the tensor key on the CI ring and the conversion keys (`TraceKey`, `EmbedKey`, `conversion_keys()`), so `ckks_bootstrap_ci_into` takes any key store.
  A key store implements `BootstrappingLinearKeys` beside `BootstrappingKeys`; `CKKSModUpFirstFactorImpl::ckks_mod_up_first_factor` takes `BootstrappingLinearKeys`.
  `BootstrappingKeysPrepared` takes the backend EvalMod runs on as a last type parameter, the bootstrap's by default; it, `BootstrappingKeysLayout` and `BootstrappingKeySet` gain `conversion_keys`.
  `CKKSEvalModImpl` no longer requires the standard ring; the reference evaluates `ExpCmplx` through `CKKSEvalModComplexOps`, which the conjugate-invariant ring refuses.
- **Breaking:** `BootstrappingContext::eval_mod` returns an `Option`, `None` for EvalMod on the conjugate-invariant ring, whose context encodes no EvalMod on the standard ring; `BootstrappingContext::eval_mod_plan` gives the plan on either ring.
- `test_suite::presets::bootstrapping_presets_meet_precision` and `small_s2c_preset_meets_precision` require `CKKSBootstrappingCIOps`, and run the presets with EvalMod on the conjugate-invariant ring, on its twin `test_suite::presets::CIBackend`, at the fixture radix too.
- Add `test_suite::presets::ci_bootstrap_refuses_invalid_inputs`, and `conjugate_invariant_ckks_test_suite!` runs EvalMod on the conjugate-invariant ring, the refusal of `ExpCmplx` included.
- **Breaking:** `impl_ckks_bootstrapping_reference_full!` requires `CKKSModUpFirstFactorImpl` (`impl_ckks_mod_up_first_factor_reference_full!`) beside `CKKSEncapsulatedModUpImpl`; `BootstrappingKeysLayout`, `BootstrappingKeySet` and `BootstrappingKeysPrepared` gain the first-factor key fields; `BootstrappingContext::generate_keys` requires `VecZnxAutomorphism`.
- `presets::first_factor` exposes `first_factor_work`, `first_factor_small_bits` and `first_factor_diagonal_growth`, the widths a first-factor key is laid out from.
- The encapsulated ModUp takes a sparse-to-dense key with a row offset (`k_offset` of `poulpy-core`): its switch is the raise, from the raised ciphertext at its own width, integer part included, with no zero-padded copy and no transform of zero limbs.
  A raise whose shift exceeds the key's offset first shifts its input by the difference inside the key's window; one whose shift is smaller, or that the key's rows do not cover, is an error.
  Keys without an offset take the path they took before, bit for bit.
- `BootstrappingPlan::raised_window` gives the bits the raised ciphertext fills at the bottom of the raised width: EvalMod's scale for C2S-first, the ModUp modulus and the C2S guard bits for S2C-first.
- **Breaking, behaviour:** the bootstrapping presets' sparse-to-dense key has a row offset, its rows under the raised ciphertext and ending at the raised width, and whole-limb digits covering `raised_window`, fewest rows first.
  At `N = 2^16` it is one row of 104 bits instead of 7 rows of 208 (29 MiB instead of 231 for the key's coefficients), at `N = 2^15` two rows of 52 instead of 15 (17 MiB instead of 127.5), and its modulus is at most that of the key it replaces.
- `CKKSTestParams::ksk_offset_layout` gives such a key for the test suites, and `CKKSTestParams::first_factor_key_layout` the first-factor keys; the standard C2S-first and S2C-first bootstrapping tests and the encapsulated ModUp parity test use offset keys.
  The first-factor bootstrapping test also runs without sparse-secret encapsulation, and the functional bootstrapping test with the first factor on the raise.
- The ship mux returns an error on a key with a row offset.
- **Breaking:** the C2S-first logN16 preset `n16_d35_k600_p19_c2s` is replaced by `n16_d35_k636_p20_c2s`: three factors per transform, as the S2C-first preset (C2S at scale `2^51`, S2C at `2^35`), and log message ratio 6 for at least 20 bits of precision (20.8 measured, 23.0 on average), a 41-bit input, 595 usable bits (17 levels instead of 16) and a raised modulus of 1381 bits instead of 1427.
- **Breaking:** `n15_d35_k180_p18_c2s` runs CoeffsToSlots at scale `2^50` and SlotsToCoeffs at `2^29` (were `2^49` and `2^30`), at the same widths, and keeps its 18 bits of precision (18.4 measured, 20.3 on average) with a single tensor key.
- **Breaking:** the S2C-first logN16 preset `n16_d35_k720_p19_s2c` is replaced by `n16_d35_k733_p20_s2c`: three factors per transform (C2S `[(5, 4096), (5, 256), (5, 8)]` at scale `2^50`, S2C `[(4, 4), (5, 128), (6, 4096)]` at `2^30`) for at least 20 bits of precision (20.4 measured, 23.0 on average), a 138-bit input, 595 usable bits (17 levels instead of 16) and a raised modulus of 1353 bits instead of 1382.
- The FFT64 bootstrapping benchmarks (portable, AVX2, AVX-512) and preset precision tests (portable, AVX2, oracle) run the presets at `base2k = 18` instead of 19, and `docs/bootstrapping.md` and `docs/performance.md` follow.
  At logN16 the giant steps of CoeffsToSlots and SlotsToCoeffs accumulate several hundred key-switch products in the DFT domain before one inverse transform, which `f64` no longer represents exactly at `base2k = 19`: the bootstrap returned garbage there, and returns 19.85 bits at 18.
- `CKKSCiphertext` implements `WriterTo` and `ReaderFrom` in the format of its inner `GLWE`, so a flag-clear ciphertext is rejected with `InvalidInput`; the ring stays a type parameter.
  `CKKSModuleAlloc::ckks_linear_transformation_alloc_from_infos` allocates a transform of a `LinearTransformationLayout`'s schedule with a zero plaintext of the given infos at every diagonal, the receiver of an encoding.
- `encoding::ckks_encode_linear_transformation_into` encodes a diagonal map into a transform `ckks_linear_transformation_alloc_from_infos` allocated, overwriting every plaintext, and returns the errors of `ckks_encode_linear_transformation_from_diagonals` for a malformed map or strategy, and `InvalidOperand` for a transform of another schedule, before any write; `ckks_encode_linear_transformation_from_diagonals` allocates then encodes through it, with unchanged bits.
  `api::ckks_linear_transformation_output_k` and `ckks_linear_transformation_output_layout` give the width and layout of the result of `ckks_eval_linear_transformation_into`.
- **Breaking:** `LtDiagonalMeta` requires `lt_log_sparsity`, the sparsity a diagonal imposes on its product with a ciphertext of a given degree.
- **Fix:** `ckks_eval_linear_transformation_into` and `ckks_eval_linear_transformation_sum_into` label their result sparse only where the sources and the diagonals are, as `ckks_linear_transformation_output_layout` reports. The result kept the label of `dst`, which could claim a sparsity it lacks.
- **Fix:** `From<anyhow::Error> for CKKSError` recovers a `CKKSCompositionError` anywhere in the chain, as documented, including one held by a `CKKSError` behind a context, which came back as `Internal`.
- **Breaking:** `CKKSPreparedLeft` and `CKKSPreparedRight` follow poulpy-core's prepared lifecycle: `CKKSModuleAlloc::ckks_prepared_left_alloc_from_infos` (`_right_`) allocates one for a ciphertext layout under a `PrepareHint` and `ckks_prepared_left_bytes_of_from_infos` (`_right_`) sizes it, directly on `Module` with no OEP method, over the HAL `CnvPVecAlloc` and `CnvPVecBytesOf`, which are supertraits of `CKKSModuleAlloc`, and `ckks_prepare_left_into` (`_right_`) fills it with no allocation, returning `InvalidOperand` for a source of another layout before any write.
  `CKKSMulImpl` requires `ckks_prepare_left_into` and `ckks_prepare_right_into`, which `impl_ckks_mul_reference_full!` forwards to `CKKSMulReference`, and provides `ckks_prepare_left` and `ckks_prepare_right`, which allocate then prepare into, with unchanged bits.
  `CKKSMulReference::ckks_prepare_left_reference` and `ckks_prepare_right_reference` are removed.
- **Breaking:** `CKKSMeta` is removed for the wrapped GLWE's `GLWEPlaintextMeta` (`Scale::Log(log_delta)`, `slots`, `log_sparsity`), taken by `set_meta`, `CKKSLayout::meta` and `CoeffsMeta::meta`; `CKKSInfos` extends the core traits, which the prelude re-exports with `SetK`, and `SetCKKSInfos` is blanket-implemented over them (`set_k`, `slots`, `log_sparsity`, `set_slots` and `set_log_sparsity` move there), and a plaintext is dense at its own degree, so `ckks_encrypt_sk` and `ckks_{add,sub,mul}_pt_vec*` derive its sparsity as `log2(N / n)` ([#342](https://github.com/poulpy-fhe/poulpy/issues/342)).
- **Breaking:** the `impl_ckks_infos!` arms `inner_meta` and `self_meta` are renamed `ckks_inner` and `glwe_inner`.
- **Breaking:** `ckks_dot_product_ct` takes any ciphertext types, with the bounds of `ckks_mul_into`: its generic parameters change from `<Dst: Data, D: Data, E: Data, H>` over `CKKSCiphertext`s to `<Dst, A, B, H>`.
  Callers that pass ciphertexts keep compiling, and their bits are unchanged.
- **Breaking:** `test_suite::helpers::assert_mul_pt_output_meta` takes the destination's `k` before the call and expects what a ct × pt product stamps: `log_delta = a.log_delta` and `log_budget = a.log_budget - pt.log_delta`, less the bits past that `k`.
  It expected `max(log_delta)` and `a.log_budget - min(log_delta)`, wrong when the plaintext's scale exceeds the ciphertext's, and took the offset from the output, so an output narrower than requested passed.
  `ckks_backend_test_suite!` and `ckks_backend_rank2_test_suite!` run that case (`mul_pt_scale_above_ct`) for vectors and constants, `_into` and `_assign`.
- **Breaking:** `test_suite::helpers::assert_mul_ct_output_meta`, `assert_unary_output_meta` and `assert_binary_output_meta` take the destination's `k` before the call (`a.k()` for `_assign`), as `assert_mul_pt_output_meta` does, and charge the expected budget the bits by which the result exceeds that `k`.
  They took the offset from the output, so an output narrower than requested passed.
- **Breaking:** `CKKSDotProductOps` and `CKKSLinearTransformationOps` dispatch through two new OEP families, `CKKSDotProductImpl` and `CKKSLinearTransformationImpl`, which `CKKSImpl` bundles: a backend registers `impl_ckks_dot_product_reference_full!` and `impl_ckks_linear_transformation_reference_full!`, as `impl_cpu_ckks_defaults!` does, or hand-writes the family.
  Their reference algorithms are `CKKSDotProductReference` and `CKKSLinearTransformationReference`, which an override calls for the methods it keeps.
  The plaintext dot products, `ckks_eval_linear_transformation_assign` and the self-caching evaluations are derived defaults, with unchanged bits.
- **Breaking:** `CKKSMulOps::ckks_prepare_right` and `CKKSMulImpl::ckks_prepare_right` take a `PrepareHint`, `OneShot` for an operand of one product and `Reuse` for one of many, which may select the backend's representation of the operand but not the bits of the products that read it.
  `CKKSMulImpl` also requires `ckks_prepare_tmp_bytes`: `impl_ckks_mul_reference_full!` forwards it, and a hand-written impl forwards it to `CKKSMulReference::ckks_prepare_tmp_bytes_reference`.
  Polynomial evaluation prepares its giant steps under `Reuse`, as before, so its bits are unchanged.
- `test_suite::parity::helpers`, `test_suite::parity::keys` and `test_suite::parity::layout` are public under `test-utils`, so downstream crates build parity tests from the same fixtures.
- The scratch of the SHIP masking accumulation and mux-rotate covers `vec_znx_idft_apply_tmp_bytes`.
- **Fix:** the first-slot bias of `ckks_ci_trace`, and so of `ckks_unfold` of conjugate-invariant inputs, is gone: its halving is a normalization, which rounds ties to even.
  The halving rounded every tie of both columns up, about `2^-34` at slot 0 for a degree-`2^16` ciphertext at scale `2^54`, against `2^-43` elsewhere.
- The SHIP mux-rotate takes keys of any digit width: the rank-2 input shared by a group is laid out on the digit grid with `gadget_dft_apply`, and its scratch covers that operand.
  A group whose keys are not all whole-limb must share one digit width and count, and returns an error otherwise.
  The mux key generation takes the digit width in bits, and the mux test also runs widths below a limb, one bit past a limb and one bit short of two, and checks that a mixed group is rejected.
- `test_suite::parity::keys::key_layout_bits` builds a key layout of any digit width, and the encapsulated ModUp parity also sweeps widths off a limb boundary, with zero digit prefixes of none, one and several digits and a shift ending on a digit boundary.
- Add `CKKSLinearTransformationOps::ckks_eval_linear_transformation_sum_into` and its `_tmp_bytes`: `Σ_t M_t · src_t`, over the slices `srcs`, `babies` and `lts` of one length, for resident transforms that share one giant schedule, at the giant key-switch cost of one transform, with the single transform's metadata.
  Slices of different lengths, or a term that does not share the first one's schedule, diagonals, source or baby-cache layout, are `InvalidOperand`, returned before `dst` is written.
- **Breaking, behaviour:** `ckks_dot_product_ct` relinearizes once for pairs that share layouts, which changes its bits: with two pairs or more, `dst.base2k() <= 62`, one layout and scale for every `a[i]` and one for every `b[i]`, it relinearizes once into a canonical `dst`, with the tensor key at `ckks_mul_into`'s width.
  Whichever of its two circuits it takes, that path validates the lengths, the tensor key and the budget before writing `dst`, and a long sum is no longer rejected; other pairs keep the eager path and its bits.
  Where `ckks_mul_into` takes its one-pass product (rank 1, one `base2k` across `dst`, the operands and the tensor key, and a convolution offset of at least `base2k`, which operands one limb wide or more meet), the sum is poulpy-core's `glwe_mul_relinearize_sum`: the products of every pair accumulate in the transform domain, in slice order, in floating point on FFT backends, before each column's inverse transform, so pick `base2k` for `a.len()` times the products of one pair (`Module::max_base2k`); it takes three inverse transforms in all, where the two-pass sum takes `3n + 2` for `n` pairs.
  Otherwise, as at rank 2, it sums the tensor products at `max(a.k, b.k)` without normalizing, as other lazy additions, and normalizes the sum once, right before the relinearization reads it through a DFT.
  `ckks_dot_product_ct_tmp_bytes` budgets the path that relinearizes once as the larger of `glwe_mul_relinearize_sum_tmp_bytes` at `max(a.k, b.k)`, whatever the number of pairs, and two tensors at that width plus the largest of the apply, normalize and relinearize scratch; it used `2n` ciphertexts plus one tensor at `max(res.k, a.k, b.k)`.
- `ckks_add_many`, `ckks_dot_product_pt_vec`, `ckks_dot_product_pt_const` and `ckks_dot_product_ct` on its eager path no longer reject more than `2^(63 - base2k)` terms: they accumulate every term, and normalization is deferred to the next DFT that reads the sum, as for any lazy addition.
  Valid calls keep their bits.
- Add `CKKSCompositionError::InvalidOperand { op, reason }` for operands that do not fit an operation, as opposed to `InvalidPlan`'s plan literals.
- Add `CKKSPreparedLeft`, the left operand `CKKSMulOps::ckks_prepare_left` prepares as `ckks_prepare_right` prepares a `CKKSPreparedRight`, and `ckks_prepare_tmp_bytes`, the scratch of both, which `ckks_mul_tmp_bytes` covers.
  Both prepared types give their convolution operand through `cnv()`, and report the layout and metadata of the ciphertext they were prepared from, and no noise estimate, through `LWEInfos`, `GLWEInfos`, `GLWEPlaintextInfos` and `CKKSInfos`, so they are `CKKSCtBounds`.
  They implement `CnvPVecLToBackendRef` and `CnvPVecRToBackendRef`, so the core prepared products read them where they are.
- Add `CKKSDotProductOps::ckks_dot_product_prepared`: `ckks_dot_product_ct` of the ciphertexts its operands were prepared from, bit for bit, metadata and canonical flag included, under either hint and for any number of pairs, with no operand transformed in the call.
  It shares the body of `ckks_dot_product_ct`'s path that relinearizes once: where `ckks_mul_into` takes its one-pass product, the sum is poulpy-core's `glwe_mul_prepared_relinearize_sum`; otherwise each pair is one `glwe_tensor_apply_prepared`.
  It returns `PreparedOperandLayoutMismatch` for an operand of another degree, `base2k` or rank than `dst`, and `InvalidOperand` for a side whose operands do not share one layout and scale, or past `base2k` 62, since prepared pairs have no eager path; it checks the lengths, the operands, the tensor key and the budget, in that order, before writing `dst`.
  `ckks_dot_product_ct_tmp_bytes` of the prepared operands covers it: the query also takes the prepared core products' queries, which leaves its value unchanged on backends that keep their defaults.
- **Fix:** the product offset when the operand with the larger budget has the smaller scale.
  `ckks_mul_into`, `ckks_mul_assign`, `ckks_mul_prepared_assign` and the compositions built on them (dot products, `ckks_mul_add_ct_into`, `ckks_mul_sub_ct_into`, power bases) shifted the product by `max(k)` instead of `max(log_budget) + max(log_delta)`, and decoded too small by `2^min(budget gap, scale gap)`.
  The bits change only in that case, where results were wrong.
- **Fix:** `ckks_encode_linear_transformation_from_diagonals`, `EncodeBSGS::encode_bsgs_with`, `EncodeBSGS::encode_bsgs_folded_with` and `compile_eval_mod` return `CKKSCompositionError::InvalidOperand` for a `CoeffsMeta` with `k = 0`, before allocating anything, and `encode_bsgs`, the `ComplexPolynomial` encoders and the bootstrapping plans return it through them.
  At `log_delta = 0` the linear-transformation encoder panicked while encoding the first diagonal, and at `log_delta > 0` it returned `LimbReallocationShrinksBelowMetadata`; the BSGS and EvalMod encoders panicked while encoding the first baby step.
- **Breaking:** `CKKSModuleAlloc::ckks_pt_vec_alloc` takes the slot count first and allocates the plaintext at the degree that holds them, `2 * slots` on the standard ring and `slots` on the conjugate-invariant ring; `ckks_max_slots()` gives the ring degree, and a degree below the backend's `MIN_DEGREE` panics.
  `CKKSModuleAlloc::ckks_pt_vec_alloc_compact` is removed, as `ckks_pt_vec_alloc` covers it.
  The test suite's `compact` module is renamed `plaintext_slots`, with `test_plaintext_slots_add_sub_mul`, `test_linear_transformation_diagonal_slots` and `test_diagonal_encoder_slots`; `test_compact_plaintext_embedding` is removed.
- **Breaking, behaviour:** a plaintext is always dense: the encoders take exactly one value per coefficient of the plaintext, or `CKKSModuleInfos::ckks_slots_of_degree(n)` slots for a plaintext of degree `n`, and return an error for fewer instead of spreading them at a gap.
  Allocate a message of `m` slots with `ckks_pt_vec_alloc(m, ..)`; below the backend's `MIN_DEGREE`, repeat the slots up to it.
  Linear-transformation diagonals whose period sits below that floor are repeated over the plaintext's slots, so their encoded values stay the same up to float rounding.
  `encoding::ckks_encode_linear_transformation_tmp_bytes` gives the scratch the linear-transformation encoders need for such a period, and `blas_context_encode_tmp_bytes` and `blas_pt_operand_encode_tmp_bytes` return it.
- `ckks_decrypt` writes into a plaintext of degree `n` dividing the ciphertext's `N` when `log2(N / n) <= ct.log_sparsity`, every `(N / n)`-th coefficient of the decrypted polynomial, and returns `PlaintextDegreeMismatch` for any other degree before changing it; decoding fewer slots than a plaintext holds goes through it.
- `CKKSModuleInfos::ckks_slots_of_degree` gives the slots a plaintext of a given degree holds.
- The CKKS `determinism.txt` plaintext fixtures of the encoding test are recorded at the degree of their slots, and the one- and two-slot ones are gone; their decoded values are unchanged.
- `CKKSDotProductOps::ckks_dot_product_prepared_pt` and its `_tmp_bytes` sum the products of ciphertexts prepared by `ckks_prepare_left` by plaintexts prepared by `ckks_prepare_right`, each plaintext read through an odd Galois element (`ckks_galois_element(r)` for its slots rotated by `r`), through core's `glwe_mul_prepared_plain_sum`, at the metadata rule of `ckks_mul_pt_vec_into`.
  They check their operands as `ckks_dot_product_prepared` does, and return `InvalidOperand` for an even element, before writing `dst`; one term through `1` is `ckks_mul_pt_vec_into` bit for bit. Backends get them from `impl_ckks_dot_product_reference_full!`, over `CKKSDotProductReference`; they require `Module: GLWEMulPreparedPlainSum`, which every CPU backend implements.
- **Breaking:** `BootstrappingKeysLayout` gains `rotation_keys` (layouts of single rotation keys, by Galois element) and is no longer `Copy`.
- **Breaking:** the bootstrapping presets check every key against the modulus bounds of their secrets, 128-bit estimates documented with the presets (1707 and 345 bits at `N = 2^16` for weights 1024 and 32, 851 and 162 at `N = 2^15`), and give each key the whole-limb digit of fewest rows its use admits, guarded by at least one digit: rotation keys by CoeffsToSlots factor, and the tensor key by the widest product EvalMod relinearizes (a plan with the EvalRound+ bypass keeps one shape). Key sets generated for 0.9.0 presets do not match. `BootstrappingPreset::with_dsizes` still gives every high-modulus key one digit of whole limbs; add `BootstrappingPreset::key_k_digit`.
- **Breaking:** the bootstrap and functional-bootstrap entry points no longer require `BootstrappingKeys::TensorKey` to be `GLWETensorKeyPrepared`; the parallel paths require `TensorKey: Sync`.
- **Behaviour:** the bootstraps return an error when a rotation or tensor key is used at a precision above its `max_input_k`, where the key switch dropped the input limbs past the key's rows, for any `BootstrappingKeys` store.
  `ckks_bootstrap` checks every rotation key and the tensor key at the widths the working width `ct_out.k()` sets before any work, except for an EvalRound+ recipe, whose stages check them as they use them.
- **Behaviour:** every operation returns `MissingAutomorphismKey` or `MissingRelinearizationKey` for a key that does not cover its input, through the core key lookups, the ship mux an error on an input wider than a key's `max_input_k`, and the encapsulated modulus raise `InvalidOperand` on an encapsulation key that does not cover its input.
  `ckks_rotate_into` and `ckks_conjugate_rotate_into` look their key up at the width they switch, `min(src.k(), dst.k())`.
- **Breaking:** `CKKSCompositionError::MissingAutomorphismKey` and `MissingRelinearizationKey` carry the key source's error as `source`, which their message includes and `Error::source` returns, so a key too narrow for its use is no longer reported as absent; `CKKSError::source` returns the source of a composition error.
- Add `DFTPlan::galois_elements_per_factor`.
- The bootstrap scratch bounds cover every rotation key shape of the layout, taking one maximum per key list.
- The test suite adds `small_s2c_preset_meets_precision`, an S2C-first bootstrap on an insecure `N = 2^11` preset built by the preset machinery, which backends run by default.
- `ckks-contracts.md` links `docs/layering-contract.md` instead of restating it and no longer mentions the removed typestate.
- **Breaking:** every OEP method has its API method's name, without the `_impl` suffix (`CKKSAddImpl::ckks_add_into_impl` becomes `ckks_add_into`, and so for every method of the `CKKS*Impl` families, `DFTImpl`, `DFTMatrixImpl<F>` and `CKKSEncodingImpl<F>`).
  The rotation by a slot shift, `ckks_rotate_by_tmp_bytes_impl`, `ckks_rotate_by_into_impl` and `ckks_rotate_by_assign_impl`, becomes `ckks_rotate_tmp_bytes`, `ckks_rotate_into` and `ckks_rotate_assign`, and `ckks_conjugate_with_keys_assign_impl` becomes `ckks_conjugate_assign`.
  The hooks without an API twin that take one prepared automorphism key, `ckks_rotate_tmp_bytes_impl`, `ckks_rotate_into_impl`, `ckks_rotate_assign_impl`, `ckks_conjugate_into_impl` and `ckks_conjugate_assign_impl`, become `ckks_rotate_with_key_tmp_bytes`, `ckks_rotate_with_key_into`, `ckks_rotate_with_key_assign`, `ckks_conjugate_with_key_into` and `ckks_conjugate_with_key_assign`, and their references on `CKKSRotateReference` and `CKKSConjugateReference` take the same names with `_reference`.
  A backend implementing an `*Impl` by hand renames its methods.
- **Breaking:** `CKKSPaCoCoeffEncodingImpl::ckks_paco_coeff_encodings` takes the API's parameters `(ciphertext, context, scratch)`, the compiled `PaCoContext<Self, F>` in place of `(ct, plan, base2k)`, and `ckks_paco_coeff_encodings_tmp_bytes` takes the context in place of the plan; an implementation reads the plan and the radix through `context.plan()` and `context.base2k()`.
  `test_paco_encoding_parity` compiles one context per backend.
- **Breaking:** each opt-in macro covers one OEP trait and is named after it, `impl_<family>_reference_full!` with `<family>` the trait in snake case without `Impl`.
  `impl_ckks_<family>_reference!` becomes `impl_ckks_<family>_reference_full!` for add, sub, copy, neg, pow2, imag, conjugate, rotate, mul, dot_product, linear_transformation, encryption, polynomial_evaluation, complex_polynomial_evaluation, eval_mod, bootstrapping, bootstrapping_ci, encapsulated_mod_up and mod_up_first_factor, and `impl_ckks_plaintext_reference!` becomes `impl_ckks_plaintext_znx_reference_full!` (`CKKSPlaintextZnxImpl`).
  `impl_ckks_fold_reference!` splits into `impl_ckks_fold_layout_reference_full!` (`CKKSFoldLayoutImpl`) and `impl_ckks_fold_reference_full!` (`CKKSFoldImpl<R>`), and `impl_ckks_dft_reference!` into `impl_dft_reference_full!` (`DFTImpl`) and `impl_dft_matrix_reference_full!` (`DFTMatrixImpl<F>`); a backend invokes both halves.
- Add the umbrella `impl_ckks_reference_full!`, the opt-ins of every scalar-independent family that serves both rings (plaintext, copy, add, sub, encryption, neg, pow2, rotate, mul, dot product, linear transformation, polynomial evaluation, EvalMod).
  The standard-only families, the scalar-generic seams and the PaCo and SHIP coefficient encodings keep their own registration.
- **Breaking:** `CKKSMulImpl::ckks_mul_pt_vec_tmp_bytes` and `ckks_mul_pt_const_tmp_bytes` take the API's parameters `(res, a, b)`, `b: &P` with `P: CKKSInfos` and `res`, `a` bounded by `CKKSCtBounds`, in place of `(res, a, b_k: TorusPrecision)`, and so do `CKKSMulReference::ckks_mul_pt_vec_tmp_bytes_reference` and `ckks_mul_pt_const_tmp_bytes_reference`, which read `b.k()`.
  A backend overriding them takes the plaintext and reads its precision; callers pass the plaintext instead of `pt.k()`.
- **Breaking:** `test_suite::helpers::gen_ci_keys` and `test_suite::conjugate_invariant::test_conjugate_invariant_ring_map` bound the standard module by `GLWECIKeyPreparedFactory`, which needs `poulpy_core::oep::PreparationImpl` on the backend.
- **Breaking:** every CKKS API method dispatches through a same-named OEP method, the delegates being plain dispatches. New OEP families, every method a derived default unless named:
  - `CKKSAddManyImpl` (`CKKSAddManyOps`: `CKKSAddImpl + CKKSCopyImpl`), `CKKSMulAddImpl` (`CKKSMulAddOps`: `CKKSMulImpl + CKKSAddImpl`), `CKKSAffineImpl` (`CKKSAffineOps`: `CKKSMulImpl + CKKSAddImpl`), `CKKSMulSubImpl` (`CKKSMulSubOps`: `CKKSMulImpl + CKKSSubImpl`) and `CKKSApproximationImpl` (`CKKSApproximationOps`: `CKKSAddImpl + CKKSAffineImpl + CKKSCopyImpl + CKKSPow2Impl + CKKSPolynomialEvaluationImpl`, its query bounded by `Module<Self>: CKKSAllOpsTmpBytes<Self>`), selected by `impl_ckks_add_many_derived_full!`, `impl_ckks_mul_add_derived_full!`, `impl_ckks_affine_derived_full!`, `impl_ckks_mul_sub_derived_full!` and `impl_ckks_approximation_derived_full!`; they serve both rings, join the supertraits of `CKKSImpl` and are invoked by `impl_ckks_reference_full!`.
  - On the standard ring: `CKKSRingMapImpl` (`CKKSCIRingMapOps`: `GLWECIConversionImpl + GLWEKeyswitchImpl`, `impl_ckks_ring_map_derived_full!`); `CKKSPaCoImpl` (the drivers and queries of `CKKSPaCoOps`, and the slot folds `ckks_slot_trace_assign` and `ckks_slot_product_assign`, sized by `ckks_slot_trace_assign_tmp_bytes` and `ckks_slot_product_assign_tmp_bytes`, which the PaCo queries call and an override of a fold replaces with it, that replace the crate-private `PaCoSlotOps`: `CKKSPaCoCoeffEncodingImpl`, the CKKS mul, add, sub, conjugate, copy, rotate and linear-transformation families, and core `GLWERotateImpl`, `AutomorphismImpl`, `LinearTransformationImpl` and `GLWEKeyswitchImpl`, `impl_ckks_paco_derived_full!`); `CKKSShipImpl` (the drivers and query of `CKKSShipOps` derived, with the required hooks `ckks_ship_masking_accumulate` and `ckks_ship_mux_rotate` and their queries, whose references are the new `reference::ship::CKKSShipReference`: `CKKSShipCoeffEncodingImpl`, the CKKS mul, add, sub, imag and conjugate families, core `GLWEKeyswitchImpl` and `GLWEZeroImpl` and HAL `HalVecZnxDftImpl`, `impl_ckks_ship_reference_full!`).
  - The `CKKSPaCoOps` and `CKKSShipOps` delegates require `BE: CKKSPaCoImpl` or `BE: CKKSShipImpl` with `CKKSEncodingImpl<F>`, in place of the module bounds they carried.
  - The derived PaCo and SHIP drivers and `ckks_slot_product_assign` carve their call-local ciphertexts from scratch, which their queries count: the slot product's rotation; a PaCo branch's accumulator, temporary and shifted input, the sequential driver's branch output and the encapsulated input; SHIP's bottom ciphertext and the temporaries combining the product-tree roots. The parallel PaCo driver's branch outputs, which cross threads, and SHIP's leaves and product tree, whose count the plan sets, stay on the heap.
  An out-of-tree backend registers the opt-ins: `impl_ckks_reference_full!` covers the both-ring families, and a standard backend adds `impl_ckks_ring_map_derived_full!`, `impl_ckks_paco_derived_full!` and `impl_ckks_ship_reference_full!`; one listing the both-ring opt-ins by hand adds the five new ones.
  `test_suite::paco_ops::test_paco_slot_trace`, `test_paco_slot_product` and `paco_keys::test_paco_partial_pipeline` bound `BE: CKKSPaCoImpl` in place of `PaCoSlotOps`, and `ship::test_ship_mux_rotate` bounds `BE: CKKSShipImpl` in place of its HAL and core module bounds, calling the hooks.
- **Breaking:** `CKKSAddManyOps::ckks_add_many_tmp_bytes` and `CKKSAddManyImpl::ckks_add_many_tmp_bytes` take the layouts `(res, input)` of the destination and the inputs in place of `res_size`, and return the larger of the selected `ckks_add_tmp_bytes` at `res` and `ckks_copy_tmp_bytes` from `input` to `res`: the single-input sum is a copy, which the query left out. Callers pass the destination and an input.
- **Breaking:** the required `CKKSEncodingImpl<F>` kernels are renamed, and the API names are derived defaults that check the buffer and plaintext shapes and read the plans from the cache, the checks and the plan cache the delegate held: `ckks_encode_coeffs_into` becomes `ckks_quantize_coeffs_into`, `ckks_decode_coeffs_into` `ckks_dequantize_coeffs_into`, `ckks_slots_to_coeffs_assign(module, plans, values)` `ckks_slots_to_coeffs_assign_with_plan` and `ckks_coeffs_to_slots_assign(module, plans, values)` `ckks_coeffs_to_slots_assign_with_plan`.
  A backend implementing `CKKSEncodingImpl<F>` by hand renames these four methods; results are unchanged.
- **Breaking:** the PaCo and SHIP coefficient-encoding API methods no longer check the arena in the delegate: an implementation of `CKKSPaCoCoeffEncodingImpl::ckks_paco_coeff_encodings` or `CKKSShipCoeffEncodingImpl::ckks_ship_coeff_encodings` refuses an arena below its own query with `CKKSError::Internal`, as the in-tree ones, whose queries are zero, do. `ckks_ship_coeff_encodings` names its ciphertext `ciphertext`, as the API.
- **Breaking:** `ckks_dot_product_prepared_pt` and its query lose their `Module<Self>: GLWEMulPreparedPlainSum<Self>` bound on the `CKKSDotProductImpl` methods and the opt-in macro, and their `Self: GLWEMulPreparedPlainSum<BE>` bound on `CKKSDotProductOps`; only `CKKSDotProductReference` keeps it. Callers drop the bound, and a backend selecting the reference implements `poulpy_core::oep::GLWEMulPreparedPlainSumImpl`, as every CPU backend does.
- **Breaking:** `CKKSBootstrappingOps` no longer has `CKKSDFTOps` and `CKKSEvalModOps` as supertraits, and its delegate drops its `Module<BE>: CKKSDFTOps<BE> + CKKSEvalModOps<BE>` bound, which `CKKSBootstrappingReference` keeps. Generic code bounding `CKKSBootstrappingOps<BE>` that calls a DFT or EvalMod stage adds `CKKSDFTOps<BE> + CKKSEvalModOps<BE>`, as the `test_suite::bootstrapping`, `functional_bootstrapping` and `presets` functions now do.
- **Breaking:** `impl_ckks_fold_reference_full!` implements `CKKSFoldImpl<Standard>` and `CKKSFoldImpl<ConjugateInvariant>`, in place of a blanket `CKKSFoldImpl<R>` bounded by `CKKSFoldReference`; generic code naming `CKKSFoldImpl<R>` for its own `R` bounds each ring it uses.
- **Breaking:** `oep::defaults` is removed: the DFT format wrappers' derived bodies are crate-private, as every derived body. A backend overriding a format wrapper either composes the selected copy, `ckks_dft_evaluate_assign` and arithmetic itself or leaves the wrapper to its derived default.
- **Breaking:** the helpers a backend or a non-reference module shares with the reference bodies move out of `reference`, which imports them back: `reference::DftScalar` and `reference::dft::DftScalar` become `layouts::DftScalar`; `reference::dft::matrices` is removed and `gen_dft_matrices` and `gen_dft_matrices_blockwise` stay at `reference::dft` and `reference::gen_dft_matrices` as re-exports of the generators now in a crate-private module; `reference::ckks_encode_linear_transformation_from_diagonals`, `_into` and `_tmp_bytes` become `encoding::ckks_encode_linear_transformation_from_diagonals`, `_into` and `_tmp_bytes`, and the module `reference::linear_transformation_diagonals` is removed; `reference::first_factor::{first_factor_work, first_factor_small_bits, first_factor_diagonal_growth, FIRST_FACTOR_GUARD_BITS}` become `presets::first_factor::*`.
  The crate-private PaCo factor generators, `PaCoPsiTail` and the Galois and slot helpers (`reference::paco`) move to `layouts::paco::factors` and `layouts::slots`; behaviour and signatures are unchanged, and a caller only updates the import path.
- **Fix:** `ckks_eval_linear_transformation_tmp_bytes` and `ckks_eval_linear_transformation_streamed_tmp_bytes`, and through the latter the PaCo scratch queries, count the selected `ckks_copy` that copies the `_assign` and `_self_assign` results back from their scratch working copy, which a backend with a copy workspace did not get; the PaCo queries take that query at the wider of the branch and its factors in place of their own sizing.
- **Breaking:** `ckks_prepare_dft_matrix(res, dft, scratch)` on `CKKSDFTOps`, `DFTImpl` and `reference::dft` prepares into a caller-owned `DFTMatrixPrepared` with no allocation and returns `CKKSResult<()>`, refusing with `InvalidOperand`, before any write, a `res` of another factor count, factor schedule or diagonal degree, radix or width than `dft`'s, or a factor whose diagonals do not share one scale, width and slot kind.
  `CKKSModuleAlloc::ckks_dft_matrix_prepared_alloc_from_infos` and `ckks_dft_matrix_prepared_bytes_of_from_infos` allocate and size it from the unprepared matrix, and the allocating form becomes `ckks_prepare_dft_matrix_alloc(dft, scratch) -> CKKSResult<DFTMatrixPrepared<..>>`, a derived default of `DFTImpl`.
  Callers of the allocating `ckks_prepare_dft_matrix(dft, scratch)` call `ckks_prepare_dft_matrix_alloc` and handle its result, a backend overriding `DFTImpl::ckks_prepare_dft_matrix` takes `res` and returns `CKKSResult<()>`, and `BootstrappingContextUnprepared::prepare` returns `anyhow::Result<BootstrappingContext<BE, F>>`.
  `test_dft_prepare_into` pins the lifecycle: the bytes query against the buffers, prepare-into against the allocating form, a refill after poisoning in the same buffers, and the refusal of another schedule or non-uniform diagonals with the bytes and infos unchanged.
- **Breaking:** `ckks_prepare_linear_transformation_rhs` on `CKKSLinearTransformationOps`, `CKKSLinearTransformationImpl` and `CKKSLinearTransformationReference` returns `CKKSResult<()>` and refuses in every build, before any write, with `InvalidOperand`, a `prepared` allocated for another schedule or diagonal degree, radix or width, where core's assertions fired after earlier giant steps were written, and a transformation whose diagonals do not share one scale, width and slot kind, where it asserted it in debug builds only; `ckks_eval_linear_transformation_into` and `ckks_eval_linear_transformation_sum_into` refuse such diagonals with `InvalidOperand` in place of `CKKSError::Internal`.
  Callers handle the result, and an override returns it.
  `test_prepare_linear_transformation_rejects_foreign_layout` and `test_prepare_linear_transformation_rejects_mixed_diagonals` pin the refusals on an empty arena with the destination unchanged, and `test_linear_transformation_sum` expects `InvalidOperand` for a non-uniform term.
- **Breaking:** op-wide scalars lead the argument lists, and plans and keys follow the operands, on the API, OEP, delegate, reference and derived forms and the opt-in macro arms alike:
  - `ckks_rotate_into(dst, src, k, keys, scratch)` becomes `(k, dst, src, keys, scratch)`, `ckks_rotate_assign(dst, k, keys, scratch)` `(k, dst, keys, scratch)` and `ckks_conjugate_rotate_into(dst, src, k, keys, scratch)` `(k, dst, src, keys, scratch)`;
  - `ckks_mul_pow2_into(dst, src, bits, scratch)` and `ckks_div_pow2_into` become `(bits, dst, src, scratch)`, `ckks_mul_pow2_assign(dst, bits, scratch)` `(bits, dst, scratch)` and `ckks_div_pow2_assign(dst, bits)` `(bits, dst)`, with `CKKSPow2Reference`'s `_reference` forms;
  - `ckks_fold_layout(ct_in, degree, keys)` becomes `(degree, ct_in, keys)`, `ckks_fold_tmp_bytes(ct_out, ct_in, degree, keys)` `(degree, ct_out, ct_in, keys)`, `ckks_fold_count(ins, degree)` `(degree, ins)` and `ckks_unfold(outs, folded, outbound, automorphisms, scratch)` `(outs, folded, automorphisms, outbound, scratch)`, with `CKKSFoldLayoutReference`'s and `CKKSFoldReference`'s `_reference` forms;
  - `ckks_approximation_tmp_bytes(res, input, tsk, approximation)` becomes `(res, input, approximation, tsk)`, `ckks_all_ops_tmp_bytes(ct_infos, tsk_infos, pt_prec)` `(ct_infos, pt_prec, tsk_infos)` and `ckks_all_ops_with_atk_tmp_bytes(ct_infos, tsk_infos, atk_infos, pt_prec)` `(ct_infos, pt_prec, tsk_infos, atk_infos)`;
  - `ckks_eval_poly_real_const_coeffs_from_power_basis(res, poly, power_basis, tsk, scratch)` and its complex twin become `(res, power_basis, poly, tsk, scratch)`, with `PolynomialEvaluationReference`'s `_reference` forms;
  - `ckks_ship_coeff_encodings_tmp_bytes(plan, base2k, complex)` becomes `(base2k, complex, plan)` and `ckks_ship_coeff_encodings(ciphertext, plan, base2k, complex, scratch)` `(base2k, complex, ciphertext, plan, scratch)`.
  Generic parameter lists are unchanged. Callers reorder their arguments by hand: a call in the old order does not compile, except where two arguments share a type, and a backend overriding these OEP methods reorders its signatures.
- **Breaking:** the OEP methods name and order their generics as their API twins: `CKKSEncryptionImpl::ckks_encrypt_sk<Dct, S, Pt>` becomes `<Dct, Dpt, S>`, `ckks_decrypt<S, Dct, Pt>` `<Dpt, Dct, S>`, `CKKSPlaintextZnxImpl::ckks_extract_pt<Dst, Src>` `<D, S>`, and the tensor-key generic `T` of `CKKSMulImpl::ckks_mul_into`, `ckks_mul_assign`, `ckks_mul_prepared_assign`, `ckks_square_into` and `ckks_square_assign` becomes `H`.
  A backend implementing `CKKSEncryptionImpl` by hand reorders the generics of `ckks_encrypt_sk` and `ckks_decrypt` with their bounds, and a turbofish call of either reorders its type arguments.
- **Breaking:** the determinism fixtures leave `ckks_backend_test_suite!`, which loses `encoding_determinism`, for `ckks_determinism_test_suite!(mod <name>, backend = <type>)`, whose `encoding_determinism_f64` and `encoding_determinism_quad` run them in both precisions.
  `test_suite::determinism::test_encoding_determinism<BE, F, E>(params, module, host_module)` over `TestContextBackend<Ring = Standard>` becomes `test_encoding_determinism::<BE, F>()` over `Backend<ZnxWord = i64, Ring = Standard>`, which builds its own module of degree 4096.
  An out-of-tree backend registers `ckks_determinism_test_suite!` for each standard-ring type, and a direct caller drops the encoder parameter and the arguments.
- **Breaking:** the CKKS rejection suite functions are renamed `test_<op>_rejects_<case>`, and an out-of-tree backend registering them renames its entries: `test_add_pt_vec_alignment_error` to `test_add_pt_vec_assign_rejects_alignment`, `test_add_pt_vec_base2k_mismatch_error` to `test_add_pt_vec_assign_rejects_base2k_mismatch`, `test_encode_linear_transformation_zero_k_error` to `test_encode_linear_transformation_from_diagonals_rejects_zero_k`, `test_encode_bsgs_zero_k_error` to `test_encode_bsgs_with_rejects_zero_k`, `test_mul_prepared_layout_mismatch_error` to `test_mul_prepared_assign_rejects_layout_mismatch`, `test_mul_ct_explicit_metadata_error` to `test_mul_into_rejects_explicit_metadata`, `test_div_pow2_assign_explicit_error` to `test_div_pow2_assign_rejects_insufficient_capacity`, `test_rotate_assign_missing_key_error` to `test_rotate_assign_rejects_missing_key`, `test_encryption_degree_mismatch_error` to `test_encrypt_sk_rejects_degree_mismatch`, `test_decrypt_extract_output_hom_rem_too_large` to `test_decrypt_rejects_output_hom_rem_too_large`, `test_decrypt_extract_base2k_mismatch_error` to `test_decrypt_rejects_base2k_mismatch`, `test_decrypt_extract_degree_mismatch_error` to `test_decrypt_rejects_degree_mismatch` and `test_repeated_square_exhausts_capacity` to `test_square_into_rejects_exhausted_capacity`.
- **Breaking:** `encoding::ship_coeff_encodings_host(module, ct, plan, base2k, complex)` becomes `(module, base2k, complex, ct, plan)` and `encoding::paco_coeff_encodings_host(module, ct, plan, base2k)` `(module, base2k, ct, plan)`, the order of their API methods, and `poulpy_cpu_portable::ckks_ship::ship_coeff_encodings_staged` and `ckks_paco::paco_coeff_encodings_staged` take the same order; callers reorder their arguments by hand.
- **Behaviour:** `ShipSecretSpec` implements `Debug` by hand and prints the length of its support only, where it derived `Debug` and printed the support.
- `ckks_determinism_test_suite!(mod <name>, backend = <type>, f32)` also registers `encoding_determinism_f32`, which the portable and oracle `FFT64` types register in place of their hand-written test.
- **Behaviour:** the `op` of CKKS errors and of their messages names the invoked public method, where it named an internal label: `conjugate` and `conjugate_assign` become `ckks_conjugate_into` and `ckks_conjugate_assign`, `rotate` and `rotate_assign` `ckks_rotate_into` and `ckks_rotate_assign`, `div_pow2` and `div_pow2_assign` `ckks_div_pow2_into` and `ckks_div_pow2_assign`, `mul_pow2` `ckks_mul_pow2_into`, `neg` `ckks_neg_into`, `mul_i` and `div_i` `ckks_mul_i_into` and `ckks_div_i_into`, `copy` `ckks_copy`, `add` and `sub` `ckks_add_into` and `ckks_sub_into` (`ckks_add_pt_vec`, `ckks_add_pt_const` and their `sub` forms for the plaintext forms), `mul` (and `power_basis`) the multiplication invoked: `ckks_mul_into`, `ckks_mul_assign`, `ckks_square_into`, `ckks_square_assign`, `ckks_mul_pt_vec_into`, `ckks_mul_pt_vec_assign`, `ckks_mul_pt_const_into` or `ckks_mul_pt_const_assign`, `mul_prepared` `ckks_mul_prepared_assign`, `linear_transformation` `ckks_eval_linear_transformation_into` or `ckks_eval_linear_transformation_sum_into` (`ckks_prepare_linear_transformation_baby_steps` when preparing baby steps), the products of a dot product `ckks_dot_product_*`, `paco_slot_trace` `ckks_slot_trace_assign`, `ckks_paco_bootstrap` `ckks_paco_bootstrap_into`, `ckks_paco_bootstrap_direct_into`, `ckks_paco_bootstrap_parallel_into` or `ckks_paco_bootstrap_parallel_direct_into` (`_tmp_bytes` forms for the queries), `ckks_ship_bootstrap` `ckks_ship_bootstrap_into`, `ckks_ship_bootstrap_complex_into` or `ckks_ship_bootstrap_tmp_bytes`, `ship_mux_rotate` and `ship_masking_accumulate` `ckks_ship_bootstrap_into`, and `ckks_mod_up_first_factor` and `ckks_encapsulated_mod_up` `ckks_bootstrap_mod_up`. The shared helpers take the label from their caller, so a refusal inside `ckks_square_into`'s product no longer reads `ckks_mul`.
- **Behaviour:** refusal messages on secret paths no longer print operand values: `ShipSecretSpec::from_support`, `ShipSecretSpec::fill_glwe_secret`, the `PaCoSecretSpec` validation and `fill_glwe_secret` return errors with static messages, where they printed support indices, signs, shifts, lengths and degrees.
- Add `CKKSDFTMatrixOps::ckks_new_dft_matrix_tmp_bytes`, the scratch of `ckks_new_dft_matrix`, and `BootstrappingContext::compile_unprepared_tmp_bytes`, the scratch of `compile_unprepared`.
  `ckks_new_dft_matrix_tmp_bytes` is the largest slot encoding of one factor's diagonals, the factors being encoded one after the other on one scratch; `compile_unprepared_tmp_bytes` is the largest of the queries of the DFT matrices and of the slot encoding EvalMod stages its coefficients through.
- **Breaking:** `DFTMatrixImpl<F>` gains the required method `ckks_new_dft_matrix_tmp_bytes`; a backend registering `impl_dft_matrix_reference_full!` is unaffected, and one implementing the trait by hand returns a bound that covers its `ckks_new_dft_matrix`, as `reference::dft::ckks_new_dft_matrix_tmp_bytes` does.
- Add `test_suite::parity::helpers::poisoned_scratch(bytes)`, an arena of exactly `bytes` poisoned bytes for a driver that keeps one across calls; `test_suite::presets::BootstrappingPresetRun` runs on one.

### `poulpy-bin-fhe`

- Add `FheUint::sext_tmp_bytes`, `splice_u8_tmp_bytes`, `splice_u16_tmp_bytes`, `get_bit_glwe_tmp_bytes` and `zero_byte_tmp_bytes`, the scratch queries of the operations that took scratch without one; the `fheuint` tests run each call on exactly its query.
- **Behaviour:** `FheUint::get_bit_lwe` narrows its input to the `max_input_k` of each switching key before the switch, and the circuit bootstrapping traces at the precision its automorphism keys cover, at most the blind rotation's; the circuit bootstrapping panics on a tensor key that does not cover the output GGSW.
- **Breaking:** `FheUintPrepared` and `FheUintPreparedDebug` allocators take `k_digit` in place of `dsize`, and the blind rotation keys report the `k_digit` their GGSWs store, `base2k` as built, as their layout does.
- The scalar-to-GGSW blind rotation reference places its gadget rows through `gadget_row_plaintext`, requires `BE: HalVecZnxImpl` in place of `VecZnxAddScalarAssign` and `VecZnxNormalizeAssign`, and its scratch covers `vec_znx_normalize_tmp_bytes`.
- `bin-fhe-contracts.md` links `docs/layering-contract.md` instead of restating it.
- **Breaking:** `BlindRotationExecuteImpl`, `LookupTableFactoryImpl`, `CircuitBootstrappingExecuteImpl`, `GLWEBlindRotationImpl` (and through it `GGSWBlindRotationImpl`), `ExecuteBDDCircuit1WTo1WImpl`, `ExecuteBDDCircuit2WTo1WImpl` and `FheUintPrepareImpl` require `Backend<Ring = Standard>`: they compose standard-only rotations, so out-of-tree backends implement them on standard-ring types only.
- **Breaking:** each opt-in macro is named after its OEP trait, without the crate prefix, and the group umbrellas follow: `impl_bin_fhe_blind_rotation_execute_reference!` becomes `impl_blind_rotation_execute_reference_full!`, `impl_bin_fhe_blind_rotation_key_encrypt_reference!` `impl_blind_rotation_key_encrypt_sk_reference_full!`, `impl_bin_fhe_blind_rotation_key_compressed_reference!` `impl_blind_rotation_key_compressed_encrypt_sk_reference_full!`, `impl_bin_fhe_blind_rotation_key_prepared_reference!` `impl_blind_rotation_key_prepared_reference_full!`, `impl_bin_fhe_blind_rotation_mod_switch_reference!` `impl_blind_rotation_mod_switch_reference_full!`, `impl_bin_fhe_blind_rotation_key_decompress_reference!` `impl_blind_rotation_key_decompress_reference_full!` and `impl_bin_fhe_lookup_table_reference!` `impl_lookup_table_factory_reference_full!`.
  `impl_bin_fhe_circuit_bootstrapping_{execute,key_encrypt_sk,key_prepared}_reference!` become `impl_circuit_bootstrapping_{execute,key_encrypt_sk,key_prepared}_reference_full!`.
  `impl_bin_fhe_{execute_bdd_circuit,execute_bdd_circuit_1w_to_1w,execute_bdd_circuit_2w_to_1w,cswap,cmux,ggsw_blind_rotation,glwe_blind_rotation,glwe_blind_selection,fhe_uint_prepared_encrypt_sk,fhe_uint_prepare,bdd_key_encrypt_sk}_reference!` become `impl_{..}_reference_full!` with the same stems, `impl_bin_fhe_bdd_key_prepared_reference!` becomes `impl_bdd_key_prepared_factory_reference_full!` (`BDDKeyPreparedFactoryImpl`) and `impl_bin_fhe_glwe_blind_retrieval_derived!` `impl_glwe_blind_retrieval_derived_full!`.
  The group umbrellas `impl_bin_fhe_blind_rotation_reference!`, `impl_bin_fhe_circuit_bootstrapping_reference!` and `impl_bin_fhe_bdd_reference!` become `impl_blind_rotation_reference_full!`, `impl_circuit_bootstrapping_reference_full!` and `impl_bdd_reference_full!`; `impl_bin_fhe_reference_full!` keeps its name.
- **Breaking:** `reference::blind_rotation::blind_rotation_key_decompress_reference` and `test_blind_rotation_key_lifecycle` bound `Module<BE>: GGSWDecompress<BE>`, the backend-generic core decompression API, so a backend registering `impl_blind_rotation_key_decompress_reference_full!` implements `poulpy_core::oep::DecompressionImpl`.
- **Breaking:** the backend-generic test bodies move to the crate-level `test_suite`: `poulpy_bin_fhe::bdd_arithmetic::test_suite` becomes `poulpy_bin_fhe::test_suite::bdd_arithmetic`, `blind_rotation::test_suite` becomes `test_suite::blind_rotation` and `circuit_bootstrapping::test_suite` becomes `test_suite::circuit_bootstrapping`. `bin_fhe_backend_test_suite!` is unchanged; a backend crate that names the bodies directly updates the path.
- **Breaking:** the modules `poulpy_bin_fhe::api::blind_rotation`, `api::circuit_bootstrapping`, `api::bdd`, `oep::blind_rotation`, `oep::circuit_bootstrapping` and `oep::bdd` are private, so `poulpy_bin_fhe::api::<family>::X` becomes `poulpy_bin_fhe::api::X` and `poulpy_bin_fhe::oep::<family>::X` becomes `poulpy_bin_fhe::oep::X`. Callers and backends update the paths only.
- **Breaking:** prepared-key allocation and sizing are provided methods of the factory traits, direct on the module with no OEP method, named object-first: `blind_rotation_key_prepared_alloc` becomes `blind_rotation_key_prepared_alloc_from_infos`, `alloc_bdd_key_from_infos` becomes `bdd_key_prepared_alloc_from_infos`, and `circuit_bootstrapping_key_prepared_alloc_from_infos` keeps its name.
  Add their byte counts `blind_rotation_key_prepared_bytes_of_from_infos`, `circuit_bootstrapping_key_prepared_bytes_of_from_infos` and `bdd_key_prepared_bytes_of_from_infos`.
  Each requires the module to implement the core prepared factories it composes, `GGSWPreparedFactory` for the blind-rotation key, plus `BlindRotationKeyPreparedFactory`, `GGLWEToGGSWKeyPreparedFactory` and `GLWEAutomorphismKeyPreparedFactory` for the circuit-bootstrapping key, plus `CircuitBootstrappingKeyPreparedFactory`, `GLWESwitchingKeyPreparedFactory` and `GLWEToLWEKeyPreparedFactory` for the BDD key, as `BlindRotationKeyPrepared::alloc`, `CircuitBootstrappingKeyPrepared::alloc_from_infos` and `BDDKeyPrepared::alloc_from_infos` do: generic callers add these bounds.
  `BlindRotationKeyPreparedImpl`, `CircuitBootstrappingKeyPreparedImpl` and `BDDKeyPreparedFactoryImpl` lose their allocators, as do their opt-in macros, and `reference::blind_rotation::blind_rotation_key_prepared_alloc_ref`, `reference::circuit_bootstrapping::circuit_bootstrapping_key_prepared_alloc_from_infos_reference` and `reference::bdd::alloc_bdd_key_from_infos_reference` are removed: hand-written impls drop the method.
- **Breaking:** `BlindRotationKeyCompressedFactoryImpl`, its opt-in `impl_bin_fhe_blind_rotation_key_compressed_factory_reference!` and `reference::blind_rotation::blind_rotation_key_compressed_alloc_ref` are removed, and every `Module<BE>` implements `BlindRotationKeyCompressedFactory<CGGI, BE>`, whose `blind_rotation_key_compressed_alloc` becomes `blind_rotation_key_compressed_alloc_from_infos` beside the new `blind_rotation_key_compressed_bytes_of_from_infos`: backends drop the impl or registration, and callers rename the call.
- **Breaking, behaviour:** the key allocators and byte counts panic in every build with a static `"<method>: key degree differs from the module degree"` when a component of their infos has another degree than the module, where the compressed allocator checked only in debug builds and the prepared ones accepted a smaller degree.
- **Breaking:** `prepare_blind_rotation_key` becomes `blind_rotation_key_prepare`, and `prepare_bdd_key` and `prepare_bdd_key_tmp_bytes` become `bdd_key_prepare` and `bdd_key_prepare_tmp_bytes`, with the same arguments, in the API traits, `BlindRotationKeyPreparedImpl`, `BDDKeyPreparedFactoryImpl` and the references `blind_rotation_key_prepare_reference`, `bdd_key_prepare_reference` and `bdd_key_prepare_tmp_bytes_reference`: callers and hand-written impls rename them.
- **Breaking:** `test_blind_rotation_parity`, `test_circuit_bootstrapping_parity`, `test_lifecycle_parity`, `test_blind_rotation`, `test_circuit_bootstrapping_to_constant`, `test_circuit_bootstrapping_to_exponent` and `TestContext::new` bound the module by the factories of the prepared keys they allocate, and `test_blind_rotation_key_serialization` drops its `BlindRotationKeyCompressedFactoryImpl<CGGI>` bound.
  The parity bodies and `test_blind_rotation_key_lifecycle` check each key allocator's byte count against its buffers and its degree refusal.
- **Breaking:** the backend's executor picks the blind-rotation schedule: `reference::blind_rotation::blind_rotation_execute_reference` and `blind_rotation_execute_tmp_bytes_reference` run the products of a block in parallel, reserving one worker scratch per product, exactly when `BE::TaskExecutor::IS_PARALLEL`, and add their contributions in index order, so every executor and pool width returns the same bytes.
  The `scheduling = parallel` arms of `impl_bin_fhe_reference_full!` and `impl_blind_rotation_reference_full!`, the opt-in `impl_bin_fhe_blind_rotation_execute_parallel!` and the references `blind_rotation_execute_parallel` and `blind_rotation_execute_tmp_bytes_parallel` are removed: a backend drops the `scheduling = parallel` argument, a hand-written impl calls the `_reference` references, and both select the parallel schedule through a parallel executor.
  Each product becomes its own contribution in place, through a one-column temporary carved from its worker scratch after the product: the serial schedule needs no more than before, the temporary now sharing the product's scratch, and the parallel one holds one product of `rank + 1` columns per block index, a one-column temporary per worker and one accumulator, where it held two products per index.
  Both references also bound `Module<BE>: VecZnxDftSubNegateAssign<BE>`: generic callers add the bound.
- **Breaking, behaviour:** the FFT64 serial block blind rotation (block size above one, extension factor one) adds each product `v` as one term, `acc + (X^{a_i}·v - v)`, where it computed `(acc + X^{a_i}·v) - v`, so its transform-domain rounding differs from 0.9.0.
  Integer outputs change only where that rounding crosses a coefficient's rounding boundary: at the parity test parameters, the 0.9.0 Rayon twins already ran the new order and matched their serial bases bit for bit. No test fixture changes.
  The NTT backends compute exactly and return the same bytes, and the extended blind rotation (extension factor above one) keeps the old order.
- **Breaking:** `glwe_blind_rotation_assign_tmp_bytes` and `ggsw_blind_rotation_assign_tmp_bytes` (`GLWEBlindRotation`, `GGSWBlindRotation`, their OEP methods and the reference `glwe_blind_rotation_assign_tmp_bytes_reference`) are removed; size the in-place forms with `glwe_blind_rotation_tmp_bytes(res, res, k)` and `ggsw_to_ggsw_blind_rotation_tmp_bytes(res, res, k)`, which cover both forms. Backends drop their override of the removed methods.
- **Breaking:** op-wide scalars lead the argument lists, and plans and keys follow the operands, on the API, OEP, delegate, reference and derived forms and the opt-in macro arms alike:
  - `glwe_blind_rotation(res, a, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch)`, `ggsw_blind_rotation` and `scalar_to_ggsw_blind_rotation(res, test_vector, fhe_uint, ..)` become `(sign, bit_rsh, bit_mask, bit_lsh, res, a, fhe_uint, scratch)`, and `glwe_blind_rotation_assign(res, value, sign, bit_rsh, bit_mask, bit_lsh, scratch)` and `ggsw_blind_rotation_assign` `(sign, bit_rsh, bit_mask, bit_lsh, res, value, scratch)`;
  - `glwe_blind_selection(res, a, fhe_uint, bit_rsh, bit_mask, scratch)` becomes `(bit_rsh, bit_mask, res, a, fhe_uint, scratch)`, and `glwe_blind_retrieval_statefull(res, bits, bit_rsh, bit_mask, scratch)` and `glwe_blind_retrieval_statefull_rev` `(bit_rsh, bit_mask, res, bits, scratch)`;
  - `execute_bdd_circuit_1w_to_1w(out, circuit, a, key, scratch)` becomes `(out, a, circuit, key, scratch)` and `execute_bdd_circuit_2w_to_1w(out, circuit, a, b, key, scratch)` `(out, a, b, circuit, key, scratch)`, their `_multi_thread` forms taking `threads` first; their queries `(circuit, res_infos, ggsw_infos, key)` become `(res_infos, ggsw_infos, circuit, key)` and the `_multi_thread_tmp_bytes` ones `(threads, res_infos, ggsw_infos, circuit, key)`; `execute_bdd_circuit_tmp_bytes(res_infos, state_size, ggsw_infos)` becomes `(state_size, res_infos, ggsw_infos)` and `execute_bdd_circuit_tmp_bytes_for(res_infos, circuit, ggsw_infos)` `(res_infos, ggsw_infos, circuit)`;
  - `circuit_bootstrapping_execute_to_constant(res, lwe, key, log_domain, extension_factor, scratch)` becomes `(log_domain, extension_factor, res, lwe, key, scratch)`, `circuit_bootstrapping_execute_to_exponent(log_gap_out, res, lwe, key, log_domain, extension_factor, scratch)` `(log_gap_out, log_domain, extension_factor, res, lwe, key, scratch)`, `circuit_bootstrapping_prepare_to_constant(res_infos, key, log_domain, extension_factor)` `(log_domain, extension_factor, res_infos, key)`, `circuit_bootstrapping_prepare_to_exponent(log_gap_out, res_infos, key, log_domain, extension_factor)` `(log_gap_out, log_domain, extension_factor, res_infos, key)` and `circuit_bootstrapping_execute_prepared(res, lwe, key, plan, scratch)` `(res, lwe, plan, key, scratch)`;
  - the `CircuitBootstrappingKeyPrepared` convenience methods follow after `module`: `prepare_to_constant(module, log_domain, extension_factor, res_infos)`, `prepare_to_exponent(module, log_gap_out, log_domain, extension_factor, res_infos)`, `execute_to_constant(module, log_domain, extension_factor, res, lwe, scratch)` and `execute_to_exponent(module, log_gap_out, log_domain, extension_factor, res, lwe, scratch)`;
  - `blind_rotation_mod_switch(modulus, res, lwe, direction)` and `blind_rotation::mod_switch_2n` become `(modulus, direction, res, lwe)`, and `lookup_table_set(res, f, k)` `(k, res, f)`;
  Generic parameter lists are unchanged. Callers move the scalars as one block in their order, since swapped `bit_rsh`, `bit_mask`, `bit_lsh`, `log_domain` and `extension_factor` still compile, and a backend overriding these OEP methods reorders its signatures.
- **Breaking:** the blind-rotation key encryptions take the LWE secret first, as the circuit-bootstrapping and BDD key encryptions do: `blind_rotation_key_encrypt_sk(res, sk_glwe, sk_lwe, source_xe, source_xa, scratch)` becomes `(res, sk_lwe, sk_glwe, source_xe, source_xa, scratch)` and `blind_rotation_key_compressed_encrypt_sk(res, sk_glwe, sk_lwe, seed_xa, source_xe, scratch)` `(res, sk_lwe, sk_glwe, seed_xa, source_xe, scratch)`, on the API, OEP, delegate and opt-in forms, the references `blind_rotation_key_encrypt_sk_reference` and `blind_rotation_key_compressed_encrypt_sk_reference` and `BlindRotationKey::encrypt_sk(module, sk_lwe, sk_glwe, ..)`, whose generic `S0` now names the LWE secret and `S1` the GLWE secret.
  `BlindRotationKeyCompressedEncryptSk<B, BRA>` becomes `BlindRotationKeyCompressedEncryptSk<BRA, BE>`, and `BlindRotationKeyEncryptSkImpl::blind_rotation_key_encrypt_sk_tmp_bytes` bounds `A: GGSWInfos` in a where clause, as its API twin.
  Callers swap the two secrets (their types differ, so a missed call does not compile), a bound on `BlindRotationKeyCompressedEncryptSk<BE, CGGI>` becomes `<CGGI, BE>`, and a hand-written impl reorders its signature.
- **Breaking:** bin-fhe references end in `_reference`, never `_ref`: `blind_rotation_execute_ref`, `blind_rotation_execute_tmp_bytes_ref`, `blind_rotation_key_encrypt_sk_ref`, `blind_rotation_key_compressed_encrypt_sk_ref`, `blind_rotation_key_decompress_ref`, `blind_rotation_key_prepare_ref`, `lookup_table_set_ref` and `lookup_table_rotate_ref`, with their `_tmp_bytes_ref` queries, become the same names ending `_reference` in `reference::blind_rotation`: backends calling them from hand-written impls rename the call.
- **Breaking:** `reference::blind_rotation::mod_switch_2n_ref` is removed and the host-staging helper lives at `blind_rotation::mod_switch_2n` with the same arguments: callers and backends rename the path.
- **Breaking:** a container's convenience methods take `module` right after `self`, then the API method's remaining arguments in its order: `<op>_multi_thread(threads, module, ..)` and `<op>_multi_thread_tmp_bytes(threads, module, ..)` of `define_bdd_2w_to_1w_trait!` and `<op>_multi_thread(threads, module, ..)` of `define_bdd_1w_to_1w_trait!` become `(module, threads, ..)`, `FheUintPrepared::prepare_custom_multi_thread(threads, module, ..)` `(module, threads, ..)`, and `LookupTable::set(module, f, k)` `(module, k, f)`: callers reorder the arguments, which a missed call may not catch for the equally typed ones.
- **Behaviour:** refusal messages on secret paths no longer print operand values: the blind rotation key and circuit bootstrapping key generation, the `FheUint` encryption and decryption and the prepared `FheUint` encryption panic with static messages, where the bare comparisons printed left and right values.

### `poulpy-mhe`

- **Breaking:** the `MHEModuleAlloc` share allocators take `k_digit` in place of `dsize`.
- GGSW and tensor key share generation place their gadget rows through `gadget_row_plaintext`, and the tensor key share scratch covers `vec_znx_normalize_tmp_bytes`.
- **Fix:** the fresh noise estimate of a collective GGSW bounds a digit by `2^(B-1)` when `k_digit` is not a whole number of limbs, where the digit layout balances digits at their own width, and keeps `2^(B-1) * (1 - 2^-B)/(1 - 2^-b)` for whole-limb digits.
  Below a limb the old factor fell under 1 and underestimated the switching noise, and `mhe-contracts.md` gave the same formula.
  `test_ggsw_share_digit_widths` checks the estimate and the measured noise at widths below a limb, one bit past a limb and one bit short of two.
- The aggregations of `GGLWEPatCompressed`, `GGLWEPat` and the public key shares add a share with one `mat_znx_add_assign` over the whole matrix, where they summed one entry at a time ([#380](https://github.com/poulpy-fhe/poulpy/issues/380)).
  The public key share aggregation compares the seed vectors once and no longer calls `glwe_pat_compressed_aggregate_assign` per entry.
  The finalizations of `GGLWEPatCompressed` and `GGLWEPat` normalize the whole matrix with `mat_znx_normalize_assign` (the bodies) and `mat_znx_normalize` (one call per output column), and their scratch queries are `mat_znx_normalize_tmp_bytes`.
  The public key share finalization normalizes the bodies with one `mat_znx_normalize` and fills each mask from its seed, where it called `glwe_pat_compressed_finalize` per entry; its scratch query is `mat_znx_normalize_tmp_bytes`.
  **Breaking:** the supertraits of `GLWEPublicKeyMHEProtocolImpl` are `HalMatZnxImpl` and `poulpy_core::oep::EncryptionImpl`, where it had `GLWEPatCompressedImpl`.
- **Fix:** `mhe_ggsw_share_finalize` checks the key's `max_input_k` against the share's precision before any write, instead of `dnum * k_digit`, which a key with `k_aux < k_digit` exceeds.
- `mhe-contracts.md` links `docs/layering-contract.md` instead of restating it and states that generation normalizes fresh shares.
- **Breaking:** each OEP trait has its own opt-in macro, named after it: `impl_mhe_pat_reference!` splits into `impl_glwe_pat_compressed_reference_full!`, `impl_gglwe_pat_compressed_reference_full!` and `impl_gglwe_pat_reference_full!`, `impl_mhe_evaluation_key_reference!` into `impl_glwe_switching_key_mhe_protocol_reference_full!` and `impl_glwe_automorphism_key_mhe_protocol_reference_full!`, `impl_mhe_keyswitch_reference!` into `impl_glwe_private_keyswitch_mhe_protocol_reference_full!` and `impl_glwe_public_keyswitch_mhe_protocol_reference_full!`, and `impl_mhe_sharing_reference!` into `impl_glwe_enc_to_share_mhe_protocol_reference_full!` and `impl_glwe_share_to_enc_mhe_protocol_reference_full!`.
  `impl_mhe_ggsw_reference!`, `impl_mhe_public_key_reference!` and `impl_mhe_tensor_key_reference!` become `impl_ggsw_mhe_protocol_reference_full!`, `impl_glwe_public_key_mhe_protocol_reference_full!` and `impl_glwe_tensor_key_mhe_protocol_reference_full!`; `impl_mhe_reference_full!` invokes the twelve.
- **Breaking:** `GGLWEPatCompressedReference` is implemented on `Module<BE>` bounded by `GGLWEDecompress<BE>`, the backend-generic core decompression API, so a backend registering `impl_gglwe_pat_compressed_reference_full!` implements `poulpy_core::oep::DecompressionImpl`.
- **Breaking:** `GLWETensorKeyMHEProtocolReference` and `GGSWMHEProtocolReference` are implemented on `Module<BE>` for `BE: HalVecZnxImpl`, which `gadget_row_plaintext` requires, and the tensor key reference no longer bounds the module by `VecZnxNormalize`.
- **Breaking:** the `api` family modules `evaluation_key`, `ggsw`, `keyswitch`, `pat`, `public_key`, `sharing` and `tensor_key` are private, so `poulpy_mhe::api::<family>::X` becomes `poulpy_mhe::api::X`. Callers update the paths only.
- **Breaking, behaviour:** every `MHEModuleAlloc::*_alloc_from_infos` panics with `"invalid layout: degree differs from the module's"` unless `infos.n()` is the module's degree, including `glwe_private_keyswitch_share_alloc_from_infos` and `glwe_enc_to_share_share_alloc_from_infos`, which did not read it. `MHEModuleAlloc` gains the `ModuleN` supertrait, which every `Module<BE>` implements; build the module at the layout's degree. `test_mhe_module_alloc_rejects_foreign_degree`, registered in `mhe_backend_test_suite!`, fires the refusal of each of the twelve allocators.
- **Breaking, behaviour:** `mhe_glwe_private_keyswitch_share_gen`, `mhe_glwe_public_keyswitch_share_gen` and `mhe_glwe_enc_to_share_share_gen` panic with `"invalid share: mask not canonical"` on a flag-clear mask, before any draw or write; normalize the ciphertext first. `test_mhe_glwe_private_keyswitch_share_gen_rejects_flag_clear_mask`, `test_mhe_glwe_public_keyswitch_share_gen_rejects_flag_clear_mask` and `test_mhe_glwe_enc_to_share_share_gen_rejects_flag_clear_mask` pin it.
- **Breaking, behaviour:** `mhe_ggsw_share_finalize` refuses a key narrower than the share with `poulpy_core::layouts::assert_key_covers`, panicking with `"mhe_ggsw_share_finalize: the input is wider than the key's max_input_k"` where it said `"invalid finalization: key does not cover the GGSW precision"`; narrow the share's precision or use a wider key. `test_mhe_ggsw_share_finalize_rejects_key_precision` pins the message.
- **Breaking:** the mhe test suite renames its 24 refusal tests (backend test suite functions in `poulpy_mhe::test_suite`, registered by `mhe_backend_test_suite!`): `test_pat_aggregate_seed_mismatch` to `test_glwe_pat_compressed_aggregate_assign_rejects_seed_mismatch`, `test_gglwe_pat_compressed_aggregate_seed_mismatch` to `test_gglwe_pat_compressed_aggregate_assign_rejects_seed_mismatch`, `test_pat_aggregate_layout_mismatch` to `test_glwe_pat_compressed_aggregate_assign_rejects_layout_mismatch`, `test_pat_finalize_layout_mismatch` to `test_glwe_pat_compressed_finalize_rejects_layout_mismatch`, `test_glwe_public_key_finalize_dist_none` to `test_mhe_glwe_public_key_share_finalize_rejects_dist_none`, `test_glwe_public_key_aggregate_rank_mismatch` to `test_mhe_glwe_public_key_share_aggregate_rejects_rank_mismatch`, `test_glwe_public_key_aggregate_dist_mismatch` to `test_mhe_glwe_public_key_share_aggregate_rejects_dist_mismatch`, `test_glwe_public_key_finalize_shared_seed` to `test_mhe_glwe_public_key_share_finalize_rejects_shared_seed`, `test_glwe_public_key_gen_secret_none` to `test_mhe_glwe_public_key_share_gen_rejects_secret_none`, `test_ggsw_share_seed_mismatch` to `test_mhe_ggsw_share_aggregate_rejects_seed_mismatch`, `test_ggsw_share_ephemeral_rank_mismatch` to `test_mhe_ggsw_share_gen_rejects_ephemeral_rank_mismatch`, `test_ggsw_share_key_rank_mismatch` to `test_mhe_ggsw_share_finalize_rejects_key_rank_mismatch`, `test_glwe_switching_key_degree_mismatch` to `test_mhe_glwe_switching_key_share_aggregate_rejects_degree_mismatch`, `test_glwe_switching_key_out_degree_mismatch` to `test_mhe_glwe_switching_key_share_aggregate_rejects_out_degree_mismatch`, `test_glwe_automorphism_key_p_mismatch` to `test_mhe_glwe_automorphism_key_share_aggregate_rejects_p_mismatch`, `test_glwe_private_keyswitch_aggregate_layout_mismatch` to `test_mhe_glwe_private_keyswitch_share_aggregate_rejects_layout_mismatch`, `test_glwe_public_keyswitch_finalize_layout_mismatch` to `test_mhe_glwe_public_keyswitch_share_finalize_rejects_layout_mismatch`, `test_glwe_public_keyswitch_pk_precision` to `test_mhe_glwe_public_keyswitch_share_gen_rejects_pk_precision`, `test_glwe_share_to_enc_precision` to `test_mhe_glwe_share_to_enc_share_gen_rejects_precision`, `test_glwe_enc_to_share_precision` to `test_mhe_glwe_enc_to_share_share_gen_rejects_precision`, `test_glwe_enc_to_share_finalize_layout_mismatch` to `test_mhe_glwe_enc_to_share_share_finalize_rejects_layout_mismatch`, `test_glwe_tensor_key_pk_precision` to `test_mhe_glwe_tensor_key_share_gen_rejects_pk_precision`, `test_glwe_tensor_key_secret_degree` to `test_mhe_glwe_tensor_key_share_gen_rejects_secret_degree`, `test_ggsw_share_key_precision_mismatch` to `test_mhe_ggsw_share_finalize_rejects_key_precision`. Each is a plain `#[test]` that pins the full message with `assert_panics_with`, then checks the output against a pre-call copy and every private `Source` is undrawn; callers that run the suite by name update the names.
- The OEP traits document each method (the reference or derived body it reproduces, seeds, degrees, Galois element, checks and `_tmp_bytes` bound), and their `# Safety` sections keep the shared obligations, among them the scratch wipe of share generation and of the encryption-to-shares finalization, and point to `poulpy-hal/docs/backend_safety_contract.md` for memory, layout, synchrony and aliasing. `aggregate_metadata` and `aggregate_common_key_metadata` live in a crate-private module outside `reference`, which `oep::derived` no longer imports.
- Add `mhe_parity_test_suite!(mod <name>, backend_ref = R, backend_test = T)` over `test_suite::parity`: `test_pat_parity` (aggregation and finalization of the three PAT types), `test_glwe_public_key_parity`, `test_glwe_switching_key_parity`, `test_glwe_automorphism_key_parity`, `test_glwe_tensor_key_parity`, `test_ggsw_parity`, `test_glwe_private_keyswitch_parity`, `test_glwe_public_keyswitch_parity`, `test_glwe_enc_to_share_parity` and `test_glwe_share_to_enc_parity`.
  Each builds its fixtures on `R` and transfers them to `T`, runs every party's generation from the same private streams, then aggregation and finalization, on each backend, and compares every output bit for bit with its layout, noise estimate, canonical flag, seeds, distribution, degrees or Galois element and the next draw of each stream, every call on exactly its own query's bytes poisoned with `0xA5`; `T` is bounded by API and transfer traits only.
  The macro runs each test inside `poulpy_core`'s `with_backend_samples` on `T`, so a `ControlledSampling*` comparison backend copies `T`'s draws. `mhe-contracts.md` and the `oep` docs state that an override passes it.
- `GLWEPatCompressed`, `GGLWEPatCompressed` and `GGLWEPat` implement `poulpy_core::TransferInto`, forwarding to their core layout, which the parity suite moves its fixtures through.
- **Behaviour:** refusal messages on secret paths no longer print operand values: the aggregation of shares with different component ranks panics with the static message "invalid aggregation: component ranks differ" only, where it also printed left and right values.

### `poulpy-blas`

- New crate: backend-agnostic encrypted matrix arithmetic on `poulpy-ckks`, on the standard and conjugate-invariant rings. A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block, and every operation acts on the batch of matrices the slots hold (`BlasCiphertext`, `BlasLayout`, `BlasModuleAlloc`, `layouts::grid::{split_into_blocks, join_blocks}`).
- `BlasMatMulOps` adds `blas_matmul` (the JKLS product of two encrypted grids), `blas_matmul_prepare_left` and `blas_matmul_prepare_right` (under a `PrepareHint`), `blas_matmul_prepared` (the same product from prepared operands of any hints) and `blas_matmul_pt_right` and `blas_matmul_pt_left` (by public matrices on the right and on the left), each with its `_tmp_bytes` query.
- `BlasMatIntraOps` adds `blas_transpose`, `blas_rotate_rows` and `blas_rotate_cols`, and `BlasMatInterOps` adds `blas_rotate` and `blas_sum` across the batch, each with its `_tmp_bytes` query.
- The operations take any `BlasGrid` (`BlasGridMut` for `res`) and `Sync` key stores, rotation keys as `keys` and the tensor key as `tsk`, document their output metadata, and leave `res` untouched on a rejected call (`CKKSCompositionError::InvalidOperand`, a missing key or a `poulpy-ckks` budget error).
- Operation-wide scalars lead every argument list, before `res`: `blas_rotate(r, res, a, keys, scratch)`, `blas_rotate_rows(k, res, a, ctx, keys, scratch)` and `blas_rotate_cols`, the validators and key-use queries behind them (after `module`), the output-layout queries of both shifts, `blas_map_blocks_tmp_bytes(sources, dst, src, lts, keys)` and `blas_jkls_block_tmp_bytes(terms, res, lhs, rhs, tsk)`.
- `BlasMatMulOperandPreparedFactory` allocates the hot and cold prepared operands `BlasMatMulLeft<BE>` and `BlasMatMulRight<BE>` (`layouts::BlasMatMulOperand`) from the infos `blas_matmul_left_operand_layout` and `blas_matmul_right_operand_layout` give, and `blas_matmul_left_term_layout` and `blas_matmul_right_term_layout` give the layout of a cold operand's terms.
- A backend implements the `oep` families `BlasMatInterImpl`, `BlasMatIntraImpl` (the map hook `blas_map_blocks`, with the transpose and both shifts provided over it by crate-private bodies, which an override of a provided method reproduces), `BlasMatMulImpl` (with `blas_matmul` provided over two hot preparations and the prepared product), `BlasJklsImpl` (one JKLS output block, over terms prepared by `ckks_prepare_left` and `ckks_prepare_right`) and `BlasEncodingImpl` (the encodings of public operands and contexts and their preparations, the encoding scalar `F` a method parameter bounded by `Self: CKKSEncodingImpl<F>`).
- Every implementation runs its operation's validator, a public `layouts::validate_*`, before any write and stamps the batch record of `res` after `Ok`, and the API dispatches without a check.
  A validator rejects a rotation or tensor key that does not cover the width it is used at, as a missing key.
  `blas_rotate` looks its key up at the narrower of `a`'s and `res`'s widths, the width `ckks_rotate_into` switches at, and `blas_rotate_key_uses` takes `res`.
- An implementation of `BlasMatMulImpl` reads and fills the operands' storage through `BlasMatMulOperand::{blocks, terms}`, their `_mut` forms and `set_prepared`.
- `poulpy-blas/docs/blas-contracts.md` states the rules every family shares: batches, key sets and widths, validation and error order, scratch and parallelism.
- `BlasPlan` fixes the block dimension and lists the rotation keys of each operation without a module, context or public matrix (`transpose_galois_elements`, `matmul_galois_elements`, …), and `sum_galois_elements` those of `blas_sum` for a grid's batch from the cyclotomic order alone. A `BlasContext` holds the plan's maps, allocated from a `BlasContextLayout`, encoded once at a caller-chosen precision with `blas_context_encode_into` and prepared with `blas_context_prepare` into a `blas_context_prepared_alloc_from_infos`; a `BlasPtOperand<BE, Right>` (the default side) or `BlasPtOperand<BE, Left>` does the same for public matrices on the right and on the left (`blas_pt_operand_alloc`, `blas_pt_operand_encode_into`, `blas_pt_operand_prepared_alloc_from_infos`, `blas_pt_operand_prepare`), matrix slot `s` reading matrix `s mod count` for a `count` that divides the batch. A `CoeffsMeta` with `k = 0` panics at allocation.
  The encodings are `BlasEncodingOps<BE, F>` of `api`, at poulpy-ckks's encoding scalar `F`, a trait parameter that a call names when its arguments do not give it (`BlasEncodingOps::<BE, f64>::blas_context_encode_into(&module, &mut ctx, scratch)`), and the preparations are on `BlasPtOperandPreparedFactory` and `BlasContextPreparedFactory` of `api`, whose allocations are direct on the module.
- `BlasPtAlgorithm` on `BlasPlan`: a public operand allocated from a plan encodes the diagonal products (default) or the JKLS ones, whose operand holds its first map (`BlasPtOperand::first`), at the same rotation keys.
- `BlasPtAlgorithm::JklsStored` and `BlasPtAlgorithm::JklsGalois` put JKLS's masks on the ciphertext: the operand's first map, sigma (tau on the left), then the context's masks of its results, phi^k on the right and the column halves of psi^k on the left (`maps::rows_left`, `maps::rows_right`, `BlasContext::rows_left`, `BlasContext::rows_right`, encoded only when the context's plan selects either path), then one sum of products by prepared public plaintexts, poulpy-ckks's `ckks_dot_product_prepared_pt`, three scales.
  `JklsStored` stores `d` plaintexts per block on the right and `2d - 1` on the left (`maps::slot_product`, `maps::tau_values`, `maps::sigma_values`), `JklsGalois` one, read through the Galois elements of their rotations; a prepared operand holds them as poulpy-ckks's `CKKSPreparedRight` (`BlasPtOperand::block_plaintexts`), which its preparation fills by `ckks_prepare_right_into`.
  On the right both take one public matrix, and `validate_blas_matmul_pt_right` rejects more with `InvalidOperand`, as `validate_blas_matmul_pt_left` rejects a context without the masks. Each path is its own circuit.
- `blas_matmul_pt_right`, `blas_matmul_pt_left`, their `_tmp_bytes`, OEP and `_reference` forms, their validators, `blas_matmul_pt_right_key_uses`, `blas_matmul_pt_left_key_uses`, `blas_matmul_pt_right_output_layout`, `blas_matmul_pt_left_output_layout`, `blas_matmul_pt_right_max_base2k` and `blas_matmul_pt_left_max_base2k` take the context, which only the masked paths read; `matmul_pt_right_galois_elements` and `matmul_pt_left_galois_elements` add the masks' keys under those paths.
- `BlasCiphertext` implements `WriterTo` and `ReaderFrom`: a header of its dimensions and block layout, its batch record, then its blocks; a grid whose blocks differ in layout, or hold a flag-clear block, is `InvalidInput` with nothing written, and a read into a grid whose blocks differ in layout is `InvalidInput`, into a grid of another layout `InvalidData`, before any block is read.
- `impl_blas_reference_full!` selects every reference implementation, and `impl_blas_mat_inter_reference_full!`, `impl_blas_mat_intra_reference_full!`, `impl_blas_mat_mul_reference_full!`, `impl_blas_jkls_reference_full!` and `impl_blas_encoding_reference_full!` one family each, over the public reference traits of `reference` (`BlasMatInterReference`, `BlasMatIntraReference`, `BlasMatMulReference`, `BlasJklsReference` for the JKLS kernel and `BlasEncodingReference`).
- `blas_matmul` and `blas_matmul_tmp_bytes` are required `BlasMatMulImpl` methods, which `impl_blas_mat_mul_reference_full!` forwards to `BlasMatMulReference::blas_matmul_reference` and `blas_matmul_tmp_bytes_reference`: the reference prepared product of two hot operands on the grids read in place, without the preparations' copies.
- Add key-use queries in `layouts`, re-exported at the crate root (`blas_matmul_key_uses`, `blas_matmul_prepare_left_key_uses`, `blas_matmul_prepare_right_key_uses`, `blas_matmul_prepared_key_uses`, `blas_matmul_pt_right_key_uses`, `blas_matmul_pt_left_key_uses`, `blas_transpose_key_uses`, `blas_rotate_rows_key_uses`, `blas_rotate_cols_key_uses`, `blas_rotate_key_uses`, `blas_sum_key_uses`) returning `BlasKeyUses`, the `(Galois element, width)` pairs and the tensor key width an operation looks up.
  Each validator enumerates its lookups with the function its query collects them with.
  Add output-layout queries, in `layouts` and re-exported at the crate root as well (`blas_matmul_output_layout`, `blas_matmul_prepared_output_layout`, `blas_matmul_pt_right_output_layout`, `blas_matmul_pt_left_output_layout`, `blas_transpose_output_layout`, `blas_rotate_rows_output_layout`, `blas_rotate_cols_output_layout`, `blas_rotate_output_layout`, `blas_sum_output_layout`) returning the `BlasLayout` an operation stamps on `res`.
- `blas_matmul_pt_left` on a replicated input by a `Diagonal` operand of one public matrix on more than one matrix slot takes `d`-diagonal block maps (`BlasPtOperand::block_replicated`, `maps::left_mul_rep`, scheduled at `left_mul`'s giant step), 22 key switches per block at `d = 128` against 30 for the `2d - 1` maps of the other inputs, with the same key set.
  Such operands hold `(3d - 1) d^2` slot values per block, against `(2d - 1) d^2`.
- `blas_backend_test_suite!`, `blas_conjugate_invariant_test_suite!`, `blas_parity_test_suite!` and `blas_max_base2k_test_suite!` register the BLAS test suites for a backend under `test-utils`, the last for a backend that implements `MaxBase2k`, at 32 slots (`n = 32` on the conjugate-invariant ring).
  The parity suite also runs a `concurrency` test.
- A grid records its batch (`BlasGridInfos::mat_slots`), which `BlasCiphertext::from_blocks` reads from its blocks' label, and a product of two batches broadcasts the smaller.
  Each output block of a product of encrypted grids is one JKLS sum relinearized once through `ckks_dot_product_prepared`, over terms converted once to the transform domain (summed in the transform domain where poulpy-ckks takes its one-pass product, on rank 1, under the radix rule `blas_matmul` states).
  The products by public matrices under `Diagonal` and `Jkls` share their giant steps across the inner dimension through `ckks_eval_linear_transformation_sum_into`.
  The references run their blocks one after another: concurrency stays inside the backend's kernels, within one ciphertext operation.
- Prepared operands are hot or cold, after the `PrepareHint` of their preparation.
  Hot, `OneShot`, for one product: a relabel of the input, whose live limbs the reference copies, whose terms the prepared product evaluates under its own context, whatever the context or scale the operand was prepared under, converting each once; its budget error comes from the product.
  Cold, `Reuse`, for many: every term, sigma (tau) and its shifts, converted once by `ckks_prepare_left` (`ckks_prepare_right`).
  Every mix of operands prepared under the product's context gives the same bits, and `blas_matmul` is the prepared product of two hot operands.
  The reference products make every term once and keep it only while it serves: a hot side's terms come a block row of `a` or a block column of `b` at a time, and of two hot sides the one whose terms, with one row or column of the other's, take fewer limbs is held while the other streams, so `blas_matmul`, which reads its grids in place, holds one side's terms and one row or column of the other's, whatever the extent of the streamed side.
  An operand reports its hint and its input's block layout through `BlasMatMulOperandInfos`, and `blas_matmul_left_operand_layout` and `blas_matmul_right_operand_layout` give those infos before preparing, a `BlasMatMulOperandLayout`, for the prepared product's query, which takes the context and both key layouts.
- On a right factor whose label proves one matrix, the encrypted products (`blas_matmul`, `blas_matmul_prepare_right` and `blas_matmul_prepared`) take tau with `d` diagonals and psi^k as rotations at no scale (`replicated`, `BlasContext::{tau_rep, rotate_rows_rep}`), so its terms are one map scale wider.
- Add `blas_matmul_max_base2k`, `blas_matmul_pt_right_max_base2k` and `blas_matmul_pt_left_max_base2k` in `layouts`, re-exported at the crate root: the largest `base2k`, at most 62, that `Module::max_base2k` allows for every lazy sum of a product, counted at each radix from the grids' layouts, the context and the public operand, 0 if none does.
  The counts are those `blas_matmul` and `blas_matmul_pt_right` document.
- `validate_blas_sum` rejects with `InvalidOperand` a `base2k` whose digits cannot hold the `log2(mat_slots) + 1` lazy terms the sum adds within `2^62`, as the products reject one above 62.
- Calls with distinct `res` and `scratch` may run concurrently over a shared module, context, public operands, prepared operands and key stores, and an override keeps no state across calls.
- `blas-contracts.md` links `docs/layering-contract.md` instead of restating it.
- OEP methods have their API method's name and the hooks are `blas_map_blocks`, `blas_map_blocks_tmp_bytes`, `blas_jkls_block` and `blas_jkls_block_tmp_bytes`; each opt-in macro is named after its OEP trait (`impl_blas_mat_inter_reference_full!`, `impl_blas_mat_intra_reference_full!`, `impl_blas_mat_mul_reference_full!`, `impl_blas_jkls_reference_full!`, `impl_blas_encoding_reference_full!`).
- `blas_pt_operand_prepare` and `blas_context_prepare` panic, naming the operation, on a map whose diagonals do not share one scale, width and slot kind, which `ckks_prepare_linear_transformation_rhs` refuses, and `BlasMatMulOperandPreparedFactory` allocates and sizes its terms through `CKKSModuleAlloc`.

### `poulpy-cpu-arm`

- Add the full-size bootstrapping preset precision test for `NTT4x30NeonRayon`.

### `poulpy-bench`

- **Breaking:** `bench_ckks_bootstrapping` requires `CKKSBootstrappingCIOps`, and runs the presets with EvalMod on the conjugate-invariant ring.
- **Breaking:** `ckks_rotate_ops` and `runner_ckks_rotate_into` bound the module by `GLWETensorKeyPreparedFactory`, the factory of the key they allocate.
- **Breaking:** `blind_rotate_ops`, `circuit_bootstrapping_ops`, `bin_fhe_standard_ops`, `bench_binfhe`, `runner_blind_rotate` and `runner_circuit_bootstrapping` bound the module by the factories of the prepared keys they allocate (`GGSWPreparedFactory`, `BlindRotationKeyPreparedFactory`, `GGLWEToGGSWKeyPreparedFactory`, `GLWEAutomorphismKeyPreparedFactory`).

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
- **Breaking:** the `oep` traits lose their backend type parameter (`CKKSEncodingImpl<F>` and `DFTMatrixImpl<F>` keep the scalar); the blanket `*Impl` implementations, `DFTDefault` and `DFTMatrixDefault` are removed, so backends register every family (new `impl_ckks_eval_mod_reference_full!`, `impl_ckks_polynomial_evaluation_reference!`, `impl_ckks_bootstrapping_reference!`).
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
