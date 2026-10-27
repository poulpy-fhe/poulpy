//! Tests of what no upstream suite reaches: the two crate-internal products
//! against their reference compositions, and views the suites do not build.

use once_cell::sync::Lazy;
use poulpy_core::reference::operations::GLWETensoringReference;
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftCopy,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes, VmpPMatAlloc, VmpPrepare,
        VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, DataView, MatZnx, Module, PrepareHint, Ring, ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxDftOwned,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxShape, VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxView,
    },
    test_suite::{download_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

type BE<R> = crate::NTT4x30Gpu<super::Runtime, R>;
type Gpu = BE<poulpy_hal::layouts::Standard>;
type GpuCI = BE<poulpy_hal::layouts::ConjugateInvariant>;

const N: usize = 64;
static MODULE: Lazy<Module<Gpu>> = Lazy::new(|| Module::<Gpu>::new(N as u64));
static MODULE_CI: Lazy<Module<GpuCI>> = Lazy::new(|| Module::<GpuCI>::new(N as u64));

/// `count` signed 50-bit coefficients.
fn coefficients(count: usize, seed: &mut u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 8);
    for _ in 0..count {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        bytes.extend_from_slice(&((*seed as i64) >> 14).to_le_bytes());
    }
    bytes
}

fn vector<R: Ring>(n: usize, cols: usize, size: usize, seed: &mut u64) -> VecZnx<<BE<R> as Backend>::OwnedBuf, i64> {
    VecZnx::from_data(BE::<R>::from_host_bytes(&coefficients(n * cols * size, seed)), n, cols, size)
}

/// The transform of a random vector: unrelated content for a destination.
fn transformed<R: Ring>(module: &Module<BE<R>>, n: usize, cols: usize, size: usize, seed: &mut u64) -> VecZnxDftOwned<BE<R>> {
    let source = vector::<R>(n, cols, size, seed);
    let mut dft = module.vec_znx_dft_alloc(n, cols, size);
    for col in 0..cols {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut dft.to_backend_mut(),
            col,
            &vec_znx_backend_ref::<BE<R>>(&source),
            col,
        );
    }
    dft
}

#[allow(clippy::too_many_arguments)]
fn digits_case<R: Ring>(
    module: &Module<BE<R>>,
    n: usize,
    a_size: usize,
    dsize: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    key_size: usize,
    res_size: usize,
    product_limbs: usize,
    // Every coefficient of the operands is `-1`: every residue is the largest one.
    extreme: bool,
) {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ ((a_size * 131 + dsize * 17 + rows * 7 + res_size) as u64);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, key_size).max(
        poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_reference(
            module, res_size, cols_in, a_size, dsize, rows, cols_in, cols_out, key_size,
        ),
    ));
    let fill = |count: usize, seed: &mut u64| {
        if extreme {
            (-1i64).to_le_bytes().repeat(count)
        } else {
            coefficients(count, seed)
        }
    };
    let a = {
        let source = VecZnx::from_data(
            BE::<R>::from_host_bytes(&fill(n * cols_in * a_size, &mut seed)),
            n,
            cols_in,
            a_size,
        );
        let mut dft = module.vec_znx_dft_alloc(n, cols_in, a_size);
        for col in 0..cols_in {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut dft.to_backend_mut(),
                col,
                &vec_znx_backend_ref::<BE<R>>(&source),
                col,
            );
        }
        dft
    };
    let matrix = MatZnx::from_data(
        BE::<R>::from_host_bytes(&fill(n * rows * cols_in * cols_out * key_size, &mut seed)),
        n,
        rows,
        cols_in,
        cols_out,
        key_size,
    );
    let mut pmat = module.vmp_pmat_alloc(n, rows, cols_in, cols_out, key_size, PrepareHint::Reuse);
    module.vmp_prepare(
        &mut pmat.to_backend_mut(),
        &poulpy_hal::layouts::MatZnxToBackendRef::<BE<R>>::to_backend_ref(&matrix),
        &mut scratch.arena(),
    );

    // Both destinations start from the same unrelated content.
    let mut destination_seed = seed;
    let mut want = transformed(module, n, cols_out, res_size, &mut destination_seed);
    let mut have = transformed(module, n, cols_out, res_size, &mut seed);
    poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_reference(
        module,
        &mut want.to_backend_mut(),
        &a.to_backend_ref(),
        dsize,
        product_limbs,
        &pmat.to_backend_ref(),
        &mut scratch.arena(),
    );
    crate::hal_impl::gglwe_product_digits_strided(
        &mut have.to_backend_mut(),
        &a.to_backend_ref(),
        dsize,
        product_limbs,
        &pmat.to_backend_ref(),
    );
    assert!(
        want == have,
        "digit product differs from its reference: a_size={a_size} dsize={dsize} rows={rows} cols_in={cols_in} \
         cols_out={cols_out} key_size={key_size} res_size={res_size} product_limbs={product_limbs}"
    );

    // The rotated product against its composition: the product above over
    // `product_size` limbs, the body added to column 0, then the
    // automorphism, stored or accumulated.
    let narrow = res_size.saturating_sub(1).max(1);
    for (accumulate, body_size, product_size) in [
        (false, res_size, res_size),
        (true, res_size + 2, res_size),
        (true, res_size.div_ceil(2), res_size),
        (false, res_size, narrow),
        (true, res_size + 2, narrow),
    ] {
        let body = transformed(module, n, cols_out, body_size, &mut seed);
        let plan = module.vec_znx_dft_automorphism_plan(n, 5);
        let mut accumulator_seed = seed;
        let mut rotated_want = transformed(module, n, cols_out, res_size, &mut accumulator_seed);
        let mut rotated_have = transformed(module, n, cols_out, res_size, &mut seed);
        let mut sum = module.vec_znx_dft_alloc(n, cols_out, product_size);
        for col in 0..cols_out {
            module.vec_znx_dft_copy(1, 0, &mut sum.to_backend_mut(), col, &want.to_backend_ref(), col);
        }
        module.vec_znx_dft_add_assign(&mut sum.to_backend_mut(), 0, &body.to_backend_ref(), 0);
        for col in 0..cols_out {
            if accumulate {
                module.vec_znx_dft_automorphism_add_with_plan(
                    &plan,
                    &mut rotated_want.to_backend_mut(),
                    col,
                    &sum.to_backend_ref(),
                    col,
                    &mut scratch.arena(),
                );
            } else {
                module.vec_znx_dft_automorphism_with_plan(
                    &plan,
                    &mut rotated_want.to_backend_mut(),
                    col,
                    &sum.to_backend_ref(),
                    col,
                );
            }
        }
        crate::hal_impl::gglwe_product_fused(
            &mut rotated_have.to_backend_mut(),
            product_size,
            &a.to_backend_ref(),
            dsize,
            product_limbs,
            &pmat.to_backend_ref(),
            [Some((&body.to_backend_ref(), 0)), None],
            Some(&plan),
            accumulate,
        );
        assert!(
            rotated_want == rotated_have,
            "rotated digit product differs from its composition: a_size={a_size} dsize={dsize} rows={rows} \
             cols_in={cols_in} cols_out={cols_out} key_size={key_size} res_size={res_size} \
             product_limbs={product_limbs} accumulate={accumulate} body_size={body_size} product_size={product_size}"
        );
    }
}

#[test]
fn gglwe_product_digits_strided_matches_reference() {
    // (a_size, dsize, rows, cols_in, cols_out, key_size, res_size, product_limbs)
    let cases = [
        // The bootstrapping benchmark's key switch.
        (27, 4, 7, 1, 2, 33, 30, 2),
        (27, 4, 7, 1, 2, 33, 30, 1),
        (27, 4, 7, 1, 2, 33, 33, 4),
        // One digit: the plain product.
        (5, 1, 5, 1, 2, 6, 6, 1),
        (9, 1, 4, 2, 3, 5, 7, 1),
        // Source shorter than the rows, and shorter than one digit.
        (5, 2, 4, 2, 2, 8, 8, 1),
        (2, 3, 4, 1, 2, 6, 5, 1),
        // Source longer than the rows consume; result wider and narrower than the matrix.
        (20, 3, 4, 2, 3, 6, 9, 2),
        (11, 3, 4, 3, 1, 9, 2, 0),
        // More than one fold per result limb.
        (40, 2, 20, 2, 2, 5, 4, 1),
    ];
    for (a_size, dsize, rows, cols_in, cols_out, key_size, res_size, product_limbs) in cases {
        digits_case(
            &MODULE,
            16,
            a_size,
            dsize,
            rows,
            cols_in,
            cols_out,
            key_size,
            res_size,
            product_limbs,
            false,
        );
    }
    digits_case(&MODULE, N, 27, 4, 7, 1, 2, 33, 30, 2, false);
    digits_case(&MODULE_CI, 16, 27, 4, 7, 1, 2, 33, 30, 2, false);
}

/// The product that runs the first inverse launch of its result itself,
/// completed column by column, against the product followed by the whole
/// inverse transform and the normalization: the same coefficients, without
/// and with addends and a body.
#[test]
fn product_with_fused_inverse_matches_product_then_inverse() {
    // (a_size, dsize, rows, key_size, res_size, product_limbs)
    let cases = [
        (27, 6, 5, 33, 30, 3),
        (27, 4, 7, 33, 33, 2),
        (14, 7, 2, 20, 16, 3),
        (5, 1, 5, 6, 6, 1),
        (2, 3, 4, 6, 5, 1),
        (11, 3, 4, 9, 2, 0),
        (40, 2, 20, 5, 9, 1),
    ];
    for log_n in [13usize, 16] {
        let n = 1usize << log_n;
        let module = Module::<Gpu>::new(n as u64);
        let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 24);
        for (index, &(a_size, dsize, rows, key_size, res_size, product_limbs)) in cases.iter().enumerate() {
            // The full degree only on the two bootstrapping shapes.
            if log_n == 16 && index != 0 && index != 2 {
                continue;
            }
            let mut seed = 0x51ce_f00d_0bad_cafeu64 ^ ((a_size * 131 + dsize * 17 + rows * 7 + res_size + log_n * 1009) as u64);
            let a = transformed(&module, n, 1, a_size, &mut seed);
            let matrix = MatZnx::from_data(
                Gpu::from_host_bytes(&coefficients(n * rows * 2 * key_size, &mut seed)),
                n,
                rows,
                1,
                2,
                key_size,
            );
            let mut pmat = module.vmp_pmat_alloc(n, rows, 1, 2, key_size, PrepareHint::Reuse);
            module.vmp_prepare(
                &mut pmat.to_backend_mut(),
                &poulpy_hal::layouts::MatZnxToBackendRef::<Gpu>::to_backend_ref(&matrix),
                &mut scratch.arena(),
            );
            for extras in [false, true] {
                let addend_size = res_size.max(2) - 1;
                let addend = transformed(&module, n, 2, addend_size, &mut seed);
                let body = vector::<poulpy_hal::layouts::Standard>(n, 1, res_size, &mut seed);
                let addend_ref = addend.to_backend_ref();
                let body_ref = vec_znx_backend_ref::<Gpu>(&body);
                let addends = if extras {
                    [Some((&addend_ref, 1)), Some((&addend_ref, 0))]
                } else {
                    [None, None]
                };
                let mut want_dft = transformed(&module, n, 2, res_size, &mut seed);
                let mut have_dft = transformed(&module, n, 2, res_size, &mut seed);
                let mut want = vector::<poulpy_hal::layouts::Standard>(n, 2, res_size, &mut seed);
                let mut have = vector::<poulpy_hal::layouts::Standard>(n, 2, res_size, &mut seed);
                crate::hal_impl::gglwe_product_digits_strided(
                    &mut want_dft.to_backend_mut(),
                    &a.to_backend_ref(),
                    dsize,
                    product_limbs,
                    &pmat.to_backend_ref(),
                );
                for (col, addend) in addends.iter().enumerate() {
                    if let Some((addend, addend_col)) = addend {
                        module.vec_znx_dft_add_assign(&mut want_dft.to_backend_mut(), col, *addend, *addend_col);
                    }
                }
                assert!(crate::hal_impl::gglwe_product_digits_strided_inverse(
                    &module,
                    &mut have_dft.to_backend_mut(),
                    &a.to_backend_ref(),
                    dsize,
                    product_limbs,
                    &pmat.to_backend_ref(),
                    addends,
                ));
                for col in 0..2 {
                    let body = if extras && col == 0 { Some((&body_ref, 0)) } else { None };
                    module.vec_znx_idft_normalize(
                        &mut vec_znx_backend_mut::<Gpu>(&mut want),
                        52,
                        res_size * 52,
                        0,
                        col,
                        &mut want_dft.to_backend_mut(),
                        col,
                        52,
                        body,
                        &mut scratch.arena(),
                    );
                    crate::hal_impl::idft_tail_normalize(
                        &module,
                        &mut vec_znx_backend_mut::<Gpu>(&mut have),
                        52,
                        res_size * 52,
                        0,
                        col,
                        &mut have_dft.to_backend_mut(),
                        col,
                        52,
                        body,
                    );
                }
                assert!(
                    download_vec_znx::<Gpu>(&want) == download_vec_znx::<Gpu>(&have),
                    "fused inverse differs: log_n={log_n} a_size={a_size} dsize={dsize} rows={rows} key_size={key_size} \
                     res_size={res_size} product_limbs={product_limbs} extras={extras}"
                );
            }
        }
    }
}

/// Accumulators at their largest: every product is `(q - 1)^2`. With seven
/// digits of nine rows and only some of them reaching a result limb, a limb
/// accumulates eighteen products across digits whose own folds are skipped.
#[test]
fn gglwe_product_digits_strided_folds_across_skipped_digits() {
    digits_case(&MODULE, 8, 63, 7, 9, 1, 2, 11, 11, 3, true);
    digits_case(&MODULE, 8, 27, 4, 7, 1, 2, 33, 30, 2, true);
    digits_case(&MODULE, 8, 27, 4, 7, 1, 2, 33, 30, 1, true);
    digits_case(&MODULE, 8, 30, 5, 6, 2, 2, 12, 9, 2, true);
}

// The reference is core's, registered on the device with `enable-core`.
#[cfg(feature = "enable-core")]
#[allow(clippy::too_many_arguments)]
fn mul_columns_case<R: Ring>(
    module: &Module<BE<R>>,
    n: usize,
    b_n: usize,
    a_size: usize,
    b_size: usize,
    diag_size: usize,
    cross_size: usize,
    offset: usize,
) where
    Module<BE<R>>: GLWETensoringReference<BE<R>>,
{
    let mut seed = 0x0fed_cba9_8765_4321u64 ^ ((a_size * 31 + b_size * 7 + diag_size * 3 + cross_size + offset) as u64);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(
        [
            module.cnv_prepare_left_tmp_bytes(a_size, a_size),
            module.cnv_prepare_right_tmp_bytes(b_size, b_size),
            module.cnv_apply_dft_tmp_bytes(offset, diag_size, a_size, b_size),
            module.cnv_pairwise_apply_dft_tmp_bytes(offset, cross_size, a_size, b_size),
        ]
        .into_iter()
        .max()
        .unwrap_or(0),
    );
    let (a, b) = (vector::<R>(n, 2, a_size, &mut seed), vector::<R>(b_n, 2, b_size, &mut seed));
    let mut left = module.cnv_pvec_left_alloc(n, 2, a_size, PrepareHint::Reuse);
    let mut right = module.cnv_pvec_right_alloc(b_n, 2, b_size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
        &vec_znx_backend_ref::<BE<R>>(&a),
        &mut scratch.arena(),
    );
    module.cnv_prepare_right(
        &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
        &vec_znx_backend_ref::<BE<R>>(&b),
        &mut scratch.arena(),
    );
    let left = poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(&left);
    let right = poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(&right);

    let mut diag_seed = seed;
    let mut diag_want = transformed(module, n, 2, diag_size, &mut diag_seed);
    let mut diag_have = transformed(module, n, 2, diag_size, &mut seed);
    let mut cross_seed = seed;
    let mut cross_want = transformed(module, n, 1, cross_size, &mut cross_seed);
    let mut cross_have = transformed(module, n, 1, cross_size, &mut seed);

    module.glwe_mul_columns_reference(
        offset,
        &mut diag_want.to_backend_mut(),
        &mut cross_want.to_backend_mut(),
        &left,
        &right,
        &mut scratch.arena(),
    );
    crate::hal_impl::glwe_mul_columns(
        module,
        offset,
        &mut diag_have.to_backend_mut(),
        &mut cross_have.to_backend_mut(),
        &left,
        &right,
        false,
    );
    let shape = format!("n={n} b_n={b_n} a_size={a_size} b_size={b_size} diag={diag_size} cross={cross_size} offset={offset}");
    assert!(
        diag_want == diag_have,
        "diagonal products differ from their reference: {shape}"
    );
    assert!(cross_want == cross_have, "cross product differs from its reference: {shape}");

    // The square of `a`: the reference on both prepared forms of `a`, the
    // fused columns from the left form alone.
    let mut right = module.cnv_pvec_right_alloc(n, 2, a_size, PrepareHint::Reuse);
    module.cnv_prepare_right(
        &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
        &vec_znx_backend_ref::<BE<R>>(&a),
        &mut scratch.arena(),
    );
    let right = poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(&right);
    let mut diag_seed = seed;
    let mut diag_want = transformed(module, n, 2, diag_size, &mut diag_seed);
    let mut diag_have = transformed(module, n, 2, diag_size, &mut seed);
    let mut cross_seed = seed;
    let mut cross_want = transformed(module, n, 1, cross_size, &mut cross_seed);
    let mut cross_have = transformed(module, n, 1, cross_size, &mut seed);
    {
        use poulpy_hal::{api::VecZnxDftSubAssign, layouts::vec_znx_dft_backend_ref_from_mut};
        let mut diag = diag_want.to_backend_mut();
        let mut cross = cross_want.to_backend_mut();
        for i in 0..2 {
            module.cnv_apply_dft(offset, &mut diag, i, &left, i, &right, i, 1, &mut scratch.arena());
        }
        module.cnv_pairwise_apply_dft(offset, &mut cross, 0, &left, &right, 0, 1, &mut scratch.arena());
        let diag_ref = vec_znx_dft_backend_ref_from_mut(&diag);
        module.vec_znx_dft_sub_assign(&mut cross, 0, &diag_ref, 0);
        module.vec_znx_dft_sub_assign(&mut cross, 0, &diag_ref, 1);
    }
    crate::hal_impl::glwe_mul_columns(
        module,
        offset,
        &mut diag_have.to_backend_mut(),
        &mut cross_have.to_backend_mut(),
        &left,
        &right,
        true,
    );
    assert!(diag_want == diag_have, "squares differ from their reference: {shape}");
    assert!(
        cross_want == cross_have,
        "cross column of a square differs from its reference: {shape}"
    );
}

#[cfg(feature = "enable-core")]
#[test]
fn glwe_mul_columns_matches_reference() {
    // (n, b_n, a_size, b_size, diag_size, cross_size, offset)
    let cases = [
        (16, 16, 23, 23, 24, 24, 23),
        (16, 16, 23, 23, 24, 24, 0),
        (16, 16, 5, 3, 7, 7, 1),
        // Columns of different widths, either way.
        (16, 16, 4, 6, 9, 5, 0),
        (16, 16, 4, 6, 3, 9, 2),
        // More terms than one fold block, and an offset past every term.
        (16, 16, 40, 37, 20, 20, 30),
        (16, 16, 3, 3, 4, 4, 9),
        // A right operand of smaller degree.
        (N, 16, 6, 5, 8, 8, 3),
    ];
    for (n, b_n, a_size, b_size, diag_size, cross_size, offset) in cases {
        mul_columns_case(&MODULE, n, b_n, a_size, b_size, diag_size, cross_size, offset);
    }
    mul_columns_case(&MODULE_CI, N, 16, 6, 5, 8, 8, 3);
}

/// The fused tensor product refuses what its reference composition refuses
/// before writing either destination: here a right operand below `MIN_DEGREE`.
#[test]
fn glwe_mul_columns_rejects_a_right_operand_below_the_minimum_degree() {
    let module = &*MODULE;
    let (n, size) = (N, 3);
    let mut seed = 0x0dd0_0dd0_0dd0_0dd0u64;
    let a = vector::<poulpy_hal::layouts::Standard>(n, 2, size, &mut seed);
    let mut left = module.cnv_pvec_left_alloc(n, 2, size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
        &vec_znx_backend_ref::<Gpu>(&a),
        &mut ScratchOwned::<Gpu>::alloc(module.cnv_prepare_left_tmp_bytes(size, size)).arena(),
    );
    let narrow: poulpy_hal::layouts::CnvPVecROwned<Gpu> =
        poulpy_hal::layouts::CnvPVecR::alloc(Gpu::MIN_DEGREE / 2, 2, size, PrepareHint::Reuse);
    let mut diag = transformed(module, n, 2, size, &mut seed);
    let mut cross = transformed(module, n, 1, size, &mut seed);
    let before = (Gpu::to_host_bytes(diag.data()), Gpu::to_host_bytes(cross.data()));
    let refusal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::hal_impl::glwe_mul_columns(
            module,
            0,
            &mut diag.to_backend_mut(),
            &mut cross.to_backend_mut(),
            &poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(&left),
            &poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(&narrow),
            false,
        )
    }))
    .expect_err("glwe_mul_columns accepted a right operand below MIN_DEGREE");
    let message = refusal
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| refusal.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert_eq!(
        message,
        "convolution: right operand degree does not embed into the destination's"
    );
    assert!(
        before == (Gpu::to_host_bytes(diag.data()), Gpu::to_host_bytes(cross.data())),
        "glwe_mul_columns wrote a destination before refusing"
    );
}

/// The reconstruction that clears its temporary gives the values of the plain
/// one, limbs past the temporary's included, and leaves every residue it read
/// zero: over the conjugate-invariant ring, a unit clears the residues of a
/// coefficient and of its reflection only once it has read both.
fn clearing_reconstruction_case<R: Ring>(module: &Module<BE<R>>) {
    let (n, size) = (N, 3);
    let (kept, cleared) = (
        transformed(module, n, 1, size, &mut 7),
        transformed(module, n, 1, size, &mut 7),
    );
    let dft = |dft: &VecZnxDftOwned<BE<R>>| crate::ntt4x30::DftColumn {
        binding: dft.data().view().binding().expect("vector holds storage"),
        base: 0,
        stride: (4 * n) as u32,
        n,
        size,
    };
    let before = coefficients(2 * n * (size + 1), &mut 11);
    let big = || {
        poulpy_hal::layouts::VecZnxBig::<<BE<R> as Backend>::OwnedBuf, i128, BE<R>>::from_data(
            BE::<R>::from_host_bytes(&before),
            n,
            1,
            size + 1,
        )
    };
    let (want, have) = (big(), big());
    let column = |big: &poulpy_hal::layouts::VecZnxBig<<BE<R> as Backend>::OwnedBuf, i128, BE<R>>| crate::znx::Column {
        binding: big.data().view().binding().expect("vector holds storage"),
        base: 0,
        stride: (2 * n) as u32,
        words: 2,
        n,
        size: size + 1,
    };
    let tables = crate::module::handle(module).tables();
    crate::ntt4x30::to_big::<super::Runtime>(tables, &column(&want), &dft(&kept));
    crate::ntt4x30::to_big_clearing::<super::Runtime>(tables, &column(&have), &dft(&cleared));
    assert!(
        BE::<R>::to_host_bytes(want.data()) == BE::<R>::to_host_bytes(have.data()),
        "the clearing reconstruction differs from the plain one"
    );
    assert!(
        BE::<R>::to_host_bytes(cleared.data()).iter().all(|&byte| byte == 0),
        "the clearing reconstruction left a residue in its source"
    );
}

#[test]
fn clearing_reconstruction_matches_and_clears() {
    clearing_reconstruction_case(&MODULE);
    clearing_reconstruction_case(&MODULE_CI);
}

/// The copy back of an in-place permutation moves the temporary into the
/// destination and leaves it zero, for `i64` and `i128` words.
#[test]
fn move_from_temp_copies_and_clears() {
    for words in [1u32, 2] {
        let (n, size) = (N, 3);
        let mut seed = 0x0a0b_0c0d_0e0f_1011u64 ^ words as u64;
        let (source, before) = (
            coefficients(n * size * words as usize, &mut seed),
            coefficients(n * size * words as usize, &mut seed),
        );
        let (temp, res) = (
            crate::runtime::GpuBuf::<super::Runtime>::from_host(&source),
            crate::runtime::GpuBuf::<super::Runtime>::from_host(&before),
        );
        let column = crate::znx::Column {
            binding: res.view().binding().expect("buffer holds storage"),
            base: 0,
            stride: n as u32 * words,
            words,
            n,
            size,
        };
        crate::znx::move_from_temp::<super::Runtime>(&column, &temp.view().binding().expect("buffer holds storage"));
        let bytes = source.len();
        assert!(
            res.to_host()[..bytes] == source[..],
            "the copy back differs from the temporary"
        );
        assert!(
            temp.to_host().iter().all(|&byte| byte == 0),
            "the copy back left a word in the temporary"
        );
    }
}

/// The streamed sum of right operands of one limb, each term read through
/// its own automorphism, against the same terms one product at a time: the
/// upstream suite reads through automorphisms on several limbs only.
fn one_limb_sum_case<R: Ring>(module: &Module<BE<R>>, n: usize, b_n: usize, galois: &[i64]) {
    let (a_size, res_size, offset) = (5, 4, 1);
    let mut seed = 0x5eed_0f0e_e11bu64 ^ ((b_n * 31 + galois.len()) as u64);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(
        [
            module.cnv_prepare_left_tmp_bytes(a_size, a_size),
            module.cnv_prepare_right_tmp_bytes(1, 1),
            module.cnv_apply_dft_tmp_bytes(offset, res_size, a_size, 1),
            module.cnv_apply_dft_add_tmp_bytes(offset, res_size, a_size, 1),
            module.cnv_apply_dft_sum_tmp_bytes(offset, res_size, a_size, 1),
        ]
        .into_iter()
        .max()
        .unwrap_or(0),
    );
    let prepared: Vec<_> = galois
        .iter()
        .map(|_| {
            let (a, b) = (vector::<R>(n, 1, a_size, &mut seed), vector::<R>(b_n, 1, 1, &mut seed));
            let mut left = module.cnv_pvec_left_alloc(n, 1, a_size, PrepareHint::Reuse);
            let mut right = module.cnv_pvec_right_alloc(b_n, 1, 1, PrepareHint::Reuse);
            module.cnv_prepare_left(
                &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
                &vec_znx_backend_ref::<BE<R>>(&a),
                &mut scratch.arena(),
            );
            module.cnv_prepare_right(
                &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
                &vec_znx_backend_ref::<BE<R>>(&b),
                &mut scratch.arena(),
            );
            (left, right)
        })
        .collect();
    let terms: Vec<poulpy_hal::layouts::CnvDftAccTerm<'_, BE<R>>> = prepared
        .iter()
        .zip(galois)
        .map(|((left, right), &b_galois)| poulpy_hal::layouts::CnvDftAccTerm {
            a: poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(left),
            a_col: 0,
            b: poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(right),
            b_col: 0,
            b_galois,
        })
        .collect();

    // Both destinations start from the same unrelated content.
    let mut destination_seed = seed;
    let mut want = transformed(module, n, 1, res_size, &mut destination_seed);
    let mut have = transformed(module, n, 1, res_size, &mut seed);
    for (index, term) in terms.iter().enumerate() {
        let mut res = want.to_backend_mut();
        if index == 0 {
            module.cnv_apply_dft(
                offset,
                &mut res,
                0,
                &term.a,
                0,
                &term.b,
                0,
                term.b_galois,
                &mut scratch.arena(),
            );
        } else {
            module.cnv_apply_dft_add(
                offset,
                &mut res,
                0,
                &term.a,
                0,
                &term.b,
                0,
                term.b_galois,
                &mut scratch.arena(),
            );
        }
    }
    module.cnv_apply_dft_sum(offset, &mut have.to_backend_mut(), 0, &terms, &mut scratch.arena());
    assert!(
        want == have,
        "one-limb convolution sum differs: n={n} b_n={b_n} galois={galois:?}"
    );
}

#[test]
fn one_limb_convolution_sum_reads_each_term_through_its_automorphism() {
    let galois = [5, -1, 3, 25];
    for b_n in [N, N / 4] {
        one_limb_sum_case(&MODULE, N, b_n, &galois);
        // A group of fewer terms than one launch takes, and a lone term.
        one_limb_sum_case(&MODULE, N, b_n, &galois[..3]);
        one_limb_sum_case(&MODULE, N, b_n, &galois[1..2]);
        one_limb_sum_case(&MODULE_CI, N, b_n, &galois);
    }
}

/// The two products at the degree and limb counts of the bootstrapping presets.
#[test]
#[ignore = "full degree; run in release with --ignored"]
fn products_match_their_references_at_full_degree() {
    let n = 1 << 16;
    let module = Module::<Gpu>::new(n as u64);
    digits_case(&module, n, 27, 4, 7, 1, 2, 33, 30, 2, false);
    digits_case(&module, n, 27, 4, 7, 1, 2, 33, 33, 4, false);
    #[cfg(feature = "enable-core")]
    {
        mul_columns_case(&module, n, n, 30, 30, 31, 31, 30);
        mul_columns_case(&module, n, n, 23, 23, 24, 24, 0);
    }
}

/// The key switches, convolution sums and ciphertext products of one
/// full-size s2c bootstrap at radix 52, shape for shape, on the backend `$be`, each result brought back to normalized limbs on the host.
/// `$path` is `kernel` for the crate's fused products and `reference` for
/// their reference compositions.
macro_rules! bootstrap_shapes {
    // The fused kernels take no scratch, and each reference runs on its own query.
    (@digits kernel, $module:expr, $res:expr, $a:expr, $dsize:expr, $limbs:expr, $pmat:expr, $bytes:expr) => {{
        let _ = $bytes;
        crate::hal_impl::gglwe_product_digits_strided($res, $a, $dsize, $limbs, $pmat)
    }};
    (@digits reference, $module:expr, $res:expr, $a:expr, $dsize:expr, $limbs:expr, $pmat:expr, $bytes:expr) => {
        poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_reference(
            $module, $res, $a, $dsize, $limbs, $pmat, &mut ScratchOwned::alloc($bytes).arena(),
        )
    };
    (@tensor kernel, $module:expr, $offset:expr, $diag:expr, $cross:expr, $left:expr, $right:expr, $bytes:expr) => {{
        let _ = $bytes;
        crate::hal_impl::glwe_mul_columns(&$module, $offset, &mut $diag.to_backend_mut(), &mut $cross.to_backend_mut(), $left, $right, false)
    }};
    (@tensor reference, $module:expr, $offset:expr, $diag:expr, $cross:expr, $left:expr, $right:expr, $bytes:expr) => {
        $module.glwe_mul_columns_reference(
            $offset,
            &mut $diag.to_backend_mut(),
            &mut $cross.to_backend_mut(),
            $left,
            $right,
            &mut ScratchOwned::alloc($bytes).arena(),
        )
    };
    ($be:ty, $n:expr, $path:ident) => {{
        type B = $be;
        let n: usize = $n;
        let base2k = 52usize;
        let module = Module::<B>::new(n as u64);
        let arena = |bytes: usize| ScratchOwned::<B>::alloc(bytes);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let upload = |n: usize, cols: usize, size: usize, seed: &mut u64| -> VecZnx<<B as Backend>::OwnedBuf, i64> {
            VecZnx::from_data(B::from_host_bytes(&coefficients(n * cols * size, seed)), n, cols, size)
        };
        let normalized = |dft: &VecZnxDftOwned<B>| -> Vec<u8> {
            let (cols, size) = (dft.cols(), dft.size());
            let mut big = module.vec_znx_big_alloc(n, cols, size);
            let mut out = upload(n, cols, size, &mut 1);
            for col in 0..cols {
                module.vec_znx_idft_apply(
                    &mut big.to_backend_mut(),
                    col,
                    &dft.to_backend_ref(),
                    col,
                    &mut arena(module.vec_znx_idft_apply_tmp_bytes()).arena(),
                );
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<B>(&mut out),
                    base2k,
                    size * base2k,
                    0,
                    col,
                    &poulpy_hal::layouts::VecZnxBigToBackendRef::to_backend_ref(&big),
                    base2k,
                    col,
                    &mut arena(module.vec_znx_big_normalize_tmp_bytes()).arena(),
                );
            }
            B::to_host_bytes(out.data())
        };
        let mut results: Vec<(String, Vec<u8>)> = Vec::new();

        // Key switches: (dsize, rows, key_size, source sizes), the result two limbs longer than the source.
        for (dsize, rows, key_size, a_sizes) in [
            (4, 7, 33, vec![2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
            (1, 1, 3, vec![1]),
        ] {
            let matrix = MatZnx::from_data(
                B::from_host_bytes(&coefficients(n * rows * 2 * key_size, &mut seed)),
                n,
                rows,
                1,
                2,
                key_size,
            );
            let mut pmat = module.vmp_pmat_alloc(n, rows, 1, 2, key_size, PrepareHint::Reuse);
            module.vmp_prepare(
                &mut pmat.to_backend_mut(),
                &poulpy_hal::layouts::MatZnxToBackendRef::<B>::to_backend_ref(&matrix),
                &mut arena(module.vmp_prepare_tmp_bytes(rows, 1, 2, key_size)).arena(),
            );
            for a_size in a_sizes {
                let a = upload(n, 1, a_size, &mut seed);
                let mut a_dft = module.vec_znx_dft_alloc(n, 1, a_size);
                module.vec_znx_dft_apply(1, 0, &mut a_dft.to_backend_mut(), 0, &vec_znx_backend_ref::<B>(&a), 0);
                let mut switched = module.vec_znx_dft_alloc(n, 2, a_size + 2);
                let bytes = poulpy_core::reference::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_reference(
                    &module, a_size + 2, 1, a_size, dsize, rows, 1, 2, key_size,
                );
                bootstrap_shapes!(@digits $path, &module, &mut switched.to_backend_mut(), &a_dft.to_backend_ref(), dsize, 3,
                    &pmat.to_backend_ref(), bytes);
                results.push((format!("key switch dsize={dsize} a_size={a_size}"), normalized(&switched)));
            }
        }

        // Sums of baby steps by one-limb diagonals of a smaller degree: (terms, baby-step size, degree ratio).
        for (terms, size, ratio) in [
            (3, 2, 16),
            (3, 24, 4096),
            (3, 25, 256),
            (3, 26, 16),
            (3, 3, 256),
            (3, 4, 4096),
            (4, 2, 1),
            (4, 2, 16),
            (4, 24, 4096),
            (4, 25, 256),
            (4, 26, 16),
            (4, 27, 1),
            (4, 3, 256),
            (4, 4, 4096),
        ] {
            let mut lefts = Vec::new();
            let mut rights = Vec::new();
            for _ in 0..terms {
                let (left_source, right_source) = (upload(n, 2, size, &mut seed), upload(n / ratio, 1, 1, &mut seed));
                let mut left = module.cnv_pvec_left_alloc(n, 2, size, PrepareHint::Reuse);
                let mut right = module.cnv_pvec_right_alloc(n / ratio, 1, 1, PrepareHint::Reuse);
                module.cnv_prepare_left(
                    &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
                    &vec_znx_backend_ref::<B>(&left_source),
                    &mut arena(module.cnv_prepare_left_tmp_bytes(size, size)).arena(),
                );
                module.cnv_prepare_right(
                    &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
                    &vec_znx_backend_ref::<B>(&right_source),
                    &mut arena(module.cnv_prepare_right_tmp_bytes(1, 1)).arena(),
                );
                lefts.push(left);
                rights.push(right);
            }
            let mut summed = module.vec_znx_dft_alloc(n, 2, size);
            for col in 0..2 {
                let terms: Vec<poulpy_hal::layouts::CnvDftAccTerm<'_, B>> = lefts
                    .iter()
                    .zip(&rights)
                    .map(|(left, right)| poulpy_hal::layouts::CnvDftAccTerm {
                        a: poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(left),
                        a_col: col,
                        b: poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(right),
                        b_col: 0,
                        b_galois: 1,
                    })
                    .collect();
                let bytes = module.cnv_apply_dft_sum_tmp_bytes(0, size, size, 1);
                module.cnv_apply_dft_sum(0, &mut summed.to_backend_mut(), col, &terms, &mut arena(bytes).arena());
            }
            results.push((format!("convolution sum terms={terms} size={size} ratio={ratio}"), normalized(&summed)));
        }

        // Ciphertext products: (left size, right size, result size, offset).
        for (a_size, b_size, size, offset) in [
            (16, 16, 17, 14),
            (17, 17, 18, 15),
            (18, 18, 19, 16),
            (19, 19, 20, 17),
            (19, 20, 20, 18),
            (20, 20, 21, 18),
            (21, 21, 22, 19),
            (22, 21, 22, 20),
            (22, 22, 23, 20),
            (23, 23, 24, 21),
        ] {
            let (left_source, right_source) = (upload(n, 2, a_size, &mut seed), upload(n, 2, b_size, &mut seed));
            let mut left = module.cnv_pvec_left_alloc(n, 2, a_size, PrepareHint::Reuse);
            let mut right = module.cnv_pvec_right_alloc(n, 2, b_size, PrepareHint::Reuse);
            module.cnv_prepare_left(
                &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
                &vec_znx_backend_ref::<B>(&left_source),
                &mut arena(module.cnv_prepare_left_tmp_bytes(a_size, a_size)).arena(),
            );
            module.cnv_prepare_right(
                &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
                &vec_znx_backend_ref::<B>(&right_source),
                &mut arena(module.cnv_prepare_right_tmp_bytes(b_size, b_size)).arena(),
            );
            let (left, right) = (
                poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(&left),
                poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(&right),
            );
            let mut diag = module.vec_znx_dft_alloc(n, 2, size);
            let mut cross = module.vec_znx_dft_alloc(n, 1, size);
            let bytes = module
                .cnv_apply_dft_tmp_bytes(offset, size, a_size, b_size)
                .max(module.cnv_pairwise_apply_dft_tmp_bytes(offset, size, a_size, b_size));
            bootstrap_shapes!(@tensor $path, module, offset, diag, cross, &left, &right, bytes);
            let shape = format!("a_size={a_size} b_size={b_size} offset={offset}");
            results.push((format!("product diagonal {shape}"), normalized(&diag)));
            results.push((format!("product cross {shape}"), normalized(&cross)));
        }
        results
    }};
}

/// The device against the portable backend on [`bootstrap_shapes`]: every
/// result that differs is reported with its first differing coefficient.
#[test]
#[ignore = "full degree; run in release with --ignored"]
fn bootstrap_shapes_match_portable_at_full_degree() {
    let n = 1 << 16;
    let have = bootstrap_shapes!(Gpu, n, kernel);
    let want = bootstrap_shapes!(poulpy_cpu_portable::NTT4x30Portable, n, reference);
    assert_eq!(have.len(), want.len());
    let mut differences = Vec::new();
    for ((name, have), (_, want)) in have.iter().zip(&want) {
        let (have, want): (&[i64], &[i64]) = (bytemuck::cast_slice(have), bytemuck::cast_slice(want));
        assert_eq!(have.len(), want.len(), "{name}");
        let wrong = have.iter().zip(want).filter(|(have, want)| have != want).count();
        if let Some(first) = have.iter().zip(want).position(|(have, want)| have != want) {
            differences.push(format!(
                "{name}: {wrong} of {} coefficients differ, the first at block {} coefficient {}",
                have.len(),
                first / n,
                first % n
            ));
        }
    }
    assert!(differences.is_empty(), "\n{}", differences.join("\n"));
}

/// `vec_znx_big_normalize` on the backend `$be` at the radices `$a_base2k`
/// to `$res_base2k`, from the same `i128` limbs: the normalized limbs.
macro_rules! big_normalized {
    ($be:ty, $bytes:expr, $res_bytes:expr, $n:expr, $size:expr, $res_size:expr, $a_base2k:expr, $res_base2k:expr, $offset:expr) => {{
        type B = $be;
        let module = Module::<B>::new(64);
        let mut scratch = ScratchOwned::<B>::alloc(module.vec_znx_big_normalize_tmp_bytes());
        let big = poulpy_hal::layouts::VecZnxBig::<<B as Backend>::OwnedBuf, i128, B>::from_data(
            B::from_host_bytes($bytes),
            $n,
            1,
            $size,
        );
        let mut res: VecZnx<<B as Backend>::OwnedBuf, i64> =
            VecZnx::from_data(B::from_host_bytes($res_bytes), $n, 1, $res_size);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<B>(&mut res),
            $res_base2k,
            $res_size * $res_base2k,
            $offset,
            0,
            &poulpy_hal::layouts::VecZnxBigToBackendRef::to_backend_ref(&big),
            $a_base2k,
            0,
            &mut scratch.arena(),
        );
        B::to_host_bytes(res.data())
    }};
}

/// The radices of an `i128` source beyond what the kernel handles are
/// served through the host, and agree with the portable backend.
#[test]
fn big_normalize_at_wide_radices_matches_portable() {
    let (n, size) = (16usize, 3usize);
    let mut seed = 0x0123_4567_89ab_cdefu64;
    // Values of up to 120 bits, both signs.
    let bytes: Vec<u8> = coefficients(2 * n * size, &mut seed)
        .chunks(16)
        .flat_map(|pair| {
            let (lo, hi) = (
                i64::from_le_bytes(pair[..8].try_into().unwrap()),
                i64::from_le_bytes(pair[8..].try_into().unwrap()),
            );
            (((hi as i128) << 70) ^ (lo as i128)).to_le_bytes()
        })
        .collect();
    for (a_base2k, res_base2k, res_size, offset) in [
        (100, 62, 5, 0),
        (127, 64, 6, 0),
        (70, 50, 5, 13),
        (60, 64, 4, -7),
        (100, 1, 40, 0),
    ] {
        // Both destinations start from the same unrelated content.
        let before = coefficients(n * res_size, &mut seed);
        let have = big_normalized!(Gpu, &bytes, &before, n, size, res_size, a_base2k, res_base2k, offset);
        let want = big_normalized!(
            poulpy_cpu_portable::NTT4x30Portable,
            &bytes,
            &before,
            n,
            size,
            res_size,
            a_base2k,
            res_base2k,
            offset
        );
        assert!(have == want, "radix {a_base2k} to {res_base2k}, offset {offset}");
    }
}

/// `vec_znx_idft_normalize` on the backend `$be` at the radices `$a_base2k`
/// to `$res_base2k`, from the transform of the same limbs plus an addend.
macro_rules! idft_normalized {
    ($be:ty, $bytes:expr, $addend:expr, $res_bytes:expr, $n:expr, $size:expr, $res_size:expr, $a_base2k:expr, $res_base2k:expr) => {{
        type B = $be;
        let module = Module::<B>::new($n as u64);
        let mut scratch = ScratchOwned::<B>::alloc(module.vec_znx_idft_normalize_tmp_bytes($res_size, $size));
        let source = VecZnx::from_data(B::from_host_bytes($bytes), $n, 1, $size);
        let addend = VecZnx::from_data(B::from_host_bytes($addend), $n, 1, $size);
        let mut dft = module.vec_znx_dft_alloc($n, 1, $size);
        module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<B>(&source), 0);
        let mut res: VecZnx<<B as Backend>::OwnedBuf, i64> = VecZnx::from_data(B::from_host_bytes($res_bytes), $n, 1, $res_size);
        module.vec_znx_idft_normalize(
            &mut vec_znx_backend_mut::<B>(&mut res),
            $res_base2k,
            $res_size * $res_base2k,
            0,
            0,
            &mut dft.to_backend_mut(),
            0,
            $a_base2k,
            Some((&vec_znx_backend_ref::<B>(&addend), 0)),
            &mut scratch.arena(),
        );
        B::to_host_bytes(res.data())
    }};
}

/// The radices of an inverse transform beyond what the kernel handles take
/// the host stage of the normalization, and agree with the portable backend.
#[test]
fn idft_normalize_at_wide_radices_matches_portable() {
    let (n, size) = (16usize, 3usize);
    let mut seed = 0x7654_3210_fedc_ba98u64;
    let (bytes, addend) = (coefficients(n * size, &mut seed), coefficients(n * size, &mut seed));
    for (a_base2k, res_base2k, res_size) in [(100, 62, 5), (127, 64, 6), (50, 64, 3)] {
        // Both destinations start from the same unrelated content.
        let before = coefficients(n * res_size, &mut seed);
        let have = idft_normalized!(Gpu, &bytes, &addend, &before, n, size, res_size, a_base2k, res_base2k);
        let want = idft_normalized!(
            poulpy_cpu_portable::NTT4x30Portable,
            &bytes,
            &addend,
            &before,
            n,
            size,
            res_size,
            a_base2k,
            res_base2k
        );
        assert!(have == want, "radix {a_base2k} to {res_base2k}");
    }
}

/// A limb range of a DFT-domain vector is a view with a non-zero base: what
/// is written through it lands in those limbs and nowhere else.
#[test]
fn dft_limb_range_view() {
    let module = &*MODULE;
    let (n, cols, size) = (N, 2, 5);
    let mut seed = 0x5eed_5eed_5eed_5eedu64;
    let outside = vector::<poulpy_hal::layouts::Standard>(n, cols, size, &mut seed);
    let inside = vector::<poulpy_hal::layouts::Standard>(n, cols, 3, &mut seed);
    let mut dft = module.vec_znx_dft_alloc(n, cols, size);
    for col in 0..cols {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut dft.to_backend_mut(),
            col,
            &vec_znx_backend_ref::<Gpu>(&outside),
            col,
        );
    }
    {
        let mut full = dft.to_backend_mut();
        let mut middle = full.with_limb_range_mut(1, 4);
        for col in 0..cols {
            module.vec_znx_dft_apply(1, 0, &mut middle, col, &vec_znx_backend_ref::<Gpu>(&inside), col);
        }
    }
    let mut scratch = ScratchOwned::<Gpu>::alloc(module.vec_znx_idft_apply_tmp_bytes());
    let mut big = module.vec_znx_big_alloc(n, cols, size);
    for col in 0..cols {
        module.vec_znx_idft_apply(
            &mut big.to_backend_mut(),
            col,
            &dft.to_backend_ref(),
            col,
            &mut scratch.arena(),
        );
    }
    let got: Vec<i128> = bytemuck::pod_collect_to_vec(&Gpu::to_host_bytes(big.data()));
    let (outside, inside) = (download_vec_znx::<Gpu>(&outside), download_vec_znx::<Gpu>(&inside));
    for limb in 0..size {
        for col in 0..cols {
            let want = if (1..4).contains(&limb) {
                inside.at(col, limb - 1)
            } else {
                outside.at(col, limb)
            };
            let block = (limb * cols + col) * n;
            for (i, &want) in want.iter().enumerate() {
                assert_eq!(got[block + i], want as i128, "limb {limb} col {col} coefficient {i}");
            }
        }
    }
}

/// `vec_znx_idft_normalize` through windows of its destination and
/// of its addend equals the same call on dense copies of the windows, and
/// leaves the destination untouched outside the window.
#[test]
fn idft_normalize_windows() {
    let module = &*MODULE;
    let (n, base2k) = (N / 2, 50usize);
    let mut seed = 0x0bad_cafe_0bad_cafeu64;
    // (coeff_offset, limb_offset, limb_step, limbs) inside vectors of degree `N`.
    for (coeff_offset, limb_offset, limb_step, limbs) in [(N / 4, 1, 2, 2), (0, 0, 1, 3), (N / 2, 2, 3, 2)] {
        let mut scratch = ScratchOwned::<Gpu>::alloc(module.vec_znx_idft_normalize_tmp_bytes(limbs, 3));
        let shape = VecZnxShape::new(N, 1, 6)
            .window_coeffs(coeff_offset, n)
            .window_limbs(limb_offset, limb_step, limbs);
        let source = vector::<poulpy_hal::layouts::Standard>(n, 1, 3, &mut seed);
        let addend = vector::<poulpy_hal::layouts::Standard>(N, 1, 6, &mut seed);
        let mut res = vector::<poulpy_hal::layouts::Standard>(N, 1, 6, &mut seed);
        let (addend_host, res_before) = (download_vec_znx::<Gpu>(&addend), download_vec_znx::<Gpu>(&res));
        let dense = |host: &VecZnx<poulpy_hal::AlignedBuf, i64>| -> Vec<i64> {
            let view = VecZnx::<&[u8], i64>::from_shape(host.data().as_ref(), shape);
            (0..limbs).flat_map(|limb| view.at(0, limb).to_vec()).collect()
        };
        let addend_dense = VecZnx::from_data(Gpu::from_host_bytes(bytemuck::cast_slice(&dense(&addend_host))), n, 1, limbs);
        let mut res_dense = VecZnx::from_data(Gpu::from_host_bytes(bytemuck::cast_slice(&dense(&res_before))), n, 1, limbs);

        for (offset, with_addend) in [(0i64, true), (-7, true), (3, false)] {
            let mut dft = module.vec_znx_dft_alloc(n, 1, 3);
            let mut dft_dense = module.vec_znx_dft_alloc(n, 1, 3);
            for dft in [&mut dft, &mut dft_dense] {
                module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<Gpu>(&source), 0);
            }
            {
                let addend = vec_znx_backend_ref::<Gpu>(&addend).with_shape(shape);
                module.vec_znx_idft_normalize(
                    &mut vec_znx_backend_mut::<Gpu>(&mut res).with_shape(shape),
                    base2k,
                    limbs * base2k,
                    offset,
                    0,
                    &mut dft.to_backend_mut(),
                    0,
                    base2k,
                    with_addend.then_some((&addend, 0)),
                    &mut scratch.arena(),
                );
            }
            {
                let addend = vec_znx_backend_ref::<Gpu>(&addend_dense);
                module.vec_znx_idft_normalize(
                    &mut vec_znx_backend_mut::<Gpu>(&mut res_dense),
                    base2k,
                    limbs * base2k,
                    offset,
                    0,
                    &mut dft_dense.to_backend_mut(),
                    0,
                    base2k,
                    with_addend.then_some((&addend, 0)),
                    &mut scratch.arena(),
                );
            }
            let res_after = download_vec_znx::<Gpu>(&res);
            let want = download_vec_znx::<Gpu>(&res_dense);
            let want: Vec<i64> = (0..limbs).flat_map(|limb| want.at(0, limb).to_vec()).collect();
            assert_eq!(dense(&res_after), want, "windowed result differs from the dense one");
            let selected: Vec<usize> = (0..limbs).map(|limb| limb_offset + limb * limb_step).collect();
            for limb in 0..6 {
                for i in 0..N {
                    if !(selected.contains(&limb) && (coeff_offset..coeff_offset + n).contains(&i)) {
                        assert_eq!(
                            res_after.at(0, limb)[i],
                            res_before.at(0, limb)[i],
                            "limb {limb} coefficient {i} changed"
                        );
                    }
                }
            }
        }
    }
}

/// The device generator is ChaCha8: its keystream of each stream matches
/// `rand_chacha`'s, block boundaries included.
#[test]
fn device_keystream_is_chacha8() {
    use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
    use rand_core::Rng;
    let seed: [u8; 32] = std::array::from_fn(|i| (i * 37 + 11) as u8);
    for stream in [0u32, 1, 4097] {
        let have = crate::znx::sample::keystream::<super::Runtime>(&seed, stream, 40);
        let mut reference = ChaCha8Rng::from_seed(seed);
        reference.set_stream(stream as u64);
        let want: Vec<u32> = (0..40).map(|_| reference.next_u32()).collect();
        assert_eq!(have, want, "stream {stream}");
    }
}

/// The device samplers draw, coefficient for coefficient, what a host model of
/// them draws from `rand_chacha`: the keystream of stream `i` for coefficient
/// `i`, portable's cumulative tables and portable's uniform digit layout. The
/// radices cover a digit within a word, at a word and across words, the
/// shifts none, one bit and a full digit less one; both destinations, `i64`
/// and `i128`, start from other content, and the other column and the limb
/// past the precision stay unchanged.
#[cfg(feature = "enable-core")]
#[test]
#[expect(
    clippy::needless_range_loop,
    reason = "a limb and a coefficient index the model, the before and the after layouts at once"
)]
fn device_noise_matches_its_host_model() {
    use poulpy_core::{Noise, oep::SamplingImpl};
    use poulpy_cpu_portable::kernels::noise::gaussian_cumulative_table_portable;
    use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
    use rand_core::Rng;

    let module = &*MODULE;
    let n = N;
    let seed: [u8; 32] = std::array::from_fn(|i| (i * 13 + 5) as u8);
    let stream = |i: usize| {
        let mut rng = ChaCha8Rng::from_seed(seed);
        rng.set_stream(i as u64);
        rng
    };
    let next_u64 = |rng: &mut ChaCha8Rng| rng.next_u32() as u64 | (rng.next_u32() as u64) << 32;
    // The digits the samplers add to limbs `0..size` of coefficient `i`.
    let model = |noise: Noise, base2k: usize, size: usize, shift: usize, i: usize| -> Vec<i64> {
        let mut digits = vec![0i64; size];
        let mut rng = stream(i);
        let balanced = |value: i128, first: usize, digits: &mut Vec<i64>| {
            let mut value = value;
            let half = 1i128 << (base2k - 1);
            for digit in digits[first..].iter_mut().rev() {
                let carry = (value + half) >> base2k;
                *digit = (value - (carry << base2k)) as i64;
                value = carry;
            }
        };
        match noise {
            Noise::Uniform { bits } if bits <= base2k - shift => {
                let mask = (1u64 << bits) - 1;
                let sign_bit = 1u64 << (bits - 1);
                let word = next_u64(&mut rng) & mask;
                digits[size - 1] = ((word ^ sign_bit).wrapping_sub(sign_bit) as i64) << shift;
            }
            Noise::Uniform { bits } => {
                let (mask, half) = ((1u64 << base2k) - 1, 1u64 << (base2k - 1));
                let (mut remaining, mut padding, mut sign, mut carry) = (bits, shift, 0u64, 0u64);
                for digit in digits.iter_mut().rev() {
                    let width = remaining.min(base2k - padding);
                    let word = next_u64(&mut rng) & ((1u64 << width) - 1);
                    remaining -= width;
                    if width != 0 {
                        sign = word >> (width - 1);
                    }
                    let extension = 0u64.wrapping_sub(sign) << (width + padding);
                    let balanced = (((word << padding) | extension) & mask) + carry + half;
                    *digit = (balanced & mask) as i64 - half as i64;
                    carry = balanced >> base2k;
                    padding = 0;
                }
            }
            Noise::Gaussian { sigma } => {
                let table = gaussian_cumulative_table_portable(sigma);
                let negative = rng.next_u32() & 1 == 1;
                let u = (0..4).fold(0u128, |u, _| u << 32 | rng.next_u32() as u128);
                let magnitude = table.iter().filter(|&&threshold| u >= threshold).count() as i128;
                let value = if negative { -magnitude } else { magnitude } << shift;
                let first = if (table.len() as u128) << shift < 1u128 << (base2k - 1) {
                    size - 1
                } else if base2k == 1 {
                    0
                } else {
                    let bound_bits = (u64::BITS - (table.len() as u64).leading_zeros()) as usize;
                    size - size.min(2 + bound_bits / base2k)
                };
                balanced(value, first, &mut digits);
            }
        }
        digits
    };
    let mut fill = 0x0f1e_2d3c_4b5a_6978u64;
    for base2k in [1usize, 2, 17, 31, 32, 33, 50, 62, 63] {
        for shift in [0, 1, base2k - 1] {
            let size = 3;
            let k = size * base2k - shift;
            // The padding is below the radix: radix 1 has none.
            if shift >= base2k {
                continue;
            }
            for noise in [
                Noise::ENCRYPTION,
                Noise::Gaussian { sigma: 10.0 },
                Noise::Uniform { bits: 5 },
                Noise::Uniform { bits: 64 },
                Noise::Uniform { bits: 100 },
            ] {
                let case = format!("base2k={base2k} shift={shift} {noise:?}");
                let want: Vec<Vec<i64>> = (0..n).map(|i| model(noise, base2k, size, shift, i)).collect();
                // `i64` limbs: two columns, one limb past the precision.
                let before = coefficients(n * 2 * (size + 1), &mut fill);
                let mut small = VecZnx::from_data(Gpu::from_host_bytes(&before), n, 2, size + 1);
                Gpu::vec_znx_add_noise(module, base2k, k, &mut vec_znx_backend_mut::<Gpu>(&mut small), 1, noise, seed);
                let (old, new) = (
                    VecZnx::<&[u8], i64>::from_data(&before[..], n, 2, size + 1),
                    download_vec_znx::<Gpu>(&small),
                );
                for limb in 0..=size {
                    assert_eq!(old.at(0, limb), new.at(0, limb), "{case}: column 0 changed");
                    for i in 0..n {
                        let added = new.at(1, limb)[i].wrapping_sub(old.at(1, limb)[i]);
                        let expected = if limb < size { want[i][limb] } else { 0 };
                        assert_eq!(added, expected, "{case}: i64 limb {limb} coefficient {i}");
                    }
                }
                // `i128` limbs, the same draw.
                let before = coefficients(2 * n * 2 * (size + 1), &mut fill);
                let mut big = poulpy_hal::layouts::VecZnxBig::<<Gpu as Backend>::OwnedBuf, i128, Gpu>::from_data(
                    Gpu::from_host_bytes(&before),
                    n,
                    2,
                    size + 1,
                );
                Gpu::vec_znx_big_add_noise(module, base2k, k, &mut big.to_backend_mut(), 1, noise, seed);
                let after: Vec<i128> = bytemuck::pod_collect_to_vec(&Gpu::to_host_bytes(big.data()));
                let old: Vec<i128> = bytemuck::pod_collect_to_vec(&before);
                for limb in 0..=size {
                    for col in 0..2 {
                        for i in 0..n {
                            let at = (limb * 2 + col) * n + i;
                            let added = after[at].wrapping_sub(old[at]);
                            let expected = if col == 1 && limb < size { want[i][limb] as i128 } else { 0 };
                            assert_eq!(added, expected, "{case}: i128 column {col} limb {limb} coefficient {i}");
                        }
                    }
                }
            }
        }
    }
}

/// The capacity step changes the compiled convolution, never its result.
#[test]
fn convolution_capacity_step_keeps_the_product() {
    let (n, a_size, b_size, res_size, offset) = (N, 11, 9, 12, 3);
    let product = |step: Option<usize>| -> Vec<u8> {
        let module = Module::<Gpu>::new(n as u64);
        if let Some(step) = step {
            crate::handle(&module).set_convolution_capacity_step(step);
        }
        let mut seed = 0x51e9_0000_0000_0001u64;
        let (a, b) = (
            vector::<poulpy_hal::layouts::Standard>(n, 1, a_size, &mut seed),
            vector::<poulpy_hal::layouts::Standard>(n, 1, b_size, &mut seed),
        );
        let mut left = module.cnv_pvec_left_alloc(n, 1, a_size, PrepareHint::Reuse);
        let mut right = module.cnv_pvec_right_alloc(n, 1, b_size, PrepareHint::Reuse);
        module.cnv_prepare_left(
            &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut left),
            &vec_znx_backend_ref::<Gpu>(&a),
            &mut ScratchOwned::<Gpu>::alloc(module.cnv_prepare_left_tmp_bytes(a_size, a_size)).arena(),
        );
        module.cnv_prepare_right(
            &mut poulpy_hal::layouts::CnvPVecRToBackendMut::to_backend_mut(&mut right),
            &vec_znx_backend_ref::<Gpu>(&b),
            &mut ScratchOwned::<Gpu>::alloc(module.cnv_prepare_right_tmp_bytes(b_size, b_size)).arena(),
        );
        let mut res = transformed(&module, n, 1, res_size, &mut seed);
        module.cnv_apply_dft(
            offset,
            &mut res.to_backend_mut(),
            0,
            &poulpy_hal::layouts::CnvPVecLToBackendRef::to_backend_ref(&left),
            0,
            &poulpy_hal::layouts::CnvPVecRToBackendRef::to_backend_ref(&right),
            0,
            1,
            &mut ScratchOwned::<Gpu>::alloc(module.cnv_apply_dft_tmp_bytes(offset, res_size, a_size, b_size)).arena(),
        );
        Gpu::to_host_bytes(res.data())
    };
    let default = product(None);
    for step in [1, 3, 16] {
        assert!(product(Some(step)) == default, "the product at capacity step {step} differs");
    }
}

#[test]
#[should_panic(expected = "the capacity step is between one limb and MAX_CAPACITY_STEP")]
fn convolution_capacity_step_rejects_zero() {
    crate::handle(&*MODULE).set_convolution_capacity_step(0);
}

#[test]
#[should_panic(expected = "the capacity step is between one limb and MAX_CAPACITY_STEP")]
fn convolution_capacity_step_rejects_above_the_maximum() {
    crate::handle(&*MODULE).set_convolution_capacity_step(crate::MAX_CAPACITY_STEP + 1);
}

/// A fresh buffer reads as zeros, and a prepared operand written whole into
/// a fresh buffer, which skips the zero fill, equals one written into a
/// buffer that was filled first. A narrower source leaves a zero tail.
#[test]
fn fresh_buffers_are_zero_and_prepare_overwrites_them() {
    let module = &*MODULE;
    let mut seed = 0x05ee_d0ff_2e54_u64;
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 22);
    assert!(Gpu::alloc_bytes(4096).to_host().iter().all(|&byte| byte == 0));
    for a_size in [5, 3] {
        let a = vector::<poulpy_hal::layouts::Standard>(N, 2, a_size, &mut seed);
        let mut fresh = module.cnv_pvec_left_alloc(N, 2, 5, PrepareHint::Reuse);
        let mut settled = module.cnv_pvec_left_alloc(N, 2, 5, PrepareHint::Reuse);
        assert!(settled.data().to_host().iter().all(|&byte| byte == 0));
        for prepared in [&mut fresh, &mut settled] {
            module.cnv_prepare_left(
                &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(prepared),
                &vec_znx_backend_ref::<Gpu>(&a),
                &mut scratch.arena(),
            );
        }
        assert!(
            fresh.data() == settled.data(),
            "prepared operands differ for a source of {a_size} limbs"
        );
    }
}

/// A refused preparation into a fresh buffer leaves it owing its zero fill:
/// the refusal comes before the view that would drop the fill.
#[test]
fn refused_prepare_keeps_the_zero_fill() {
    let module = &*MODULE;
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 16);
    let mut res = module.cnv_pvec_left_alloc(N, 2, 5, PrepareHint::Reuse);
    let a = module.vec_znx_alloc(N / 2, 2, 5);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_left(
            &mut poulpy_hal::layouts::CnvPVecLToBackendMut::to_backend_mut(&mut res),
            &vec_znx_backend_ref::<Gpu>(&a),
            &mut scratch.arena(),
        );
    }))
    .expect_err("cnv_prepare_left accepted a source of another degree");
    assert!(res.data().owes_fill(), "a refused preparation dropped the zero fill");
    assert!(res.data().to_host().iter().all(|&byte| byte == 0));
}

/// The batched giant steps of a linear transformation (`cnv.scale_many` on
/// the diagonals `dft.gather` lays side by side), which the parity suite's
/// powers-of-two transformations reach: the suite compares every result with
/// portable's, and the trace shows the batched launches ran.
#[cfg(feature = "enable-core")]
#[test]
fn linear_transformation_parity_takes_the_batched_giant_steps() {
    use poulpy_core::test_suite::parity::{ParityShapes, test_linear_transformation_powers_of_two_parity};
    use poulpy_hal::test_suite::TestParams;

    use crate::runtime::{Label, Tracer};
    let params = TestParams {
        size: 1 << 8,
        base2k: 52,
        n: 1 << 8,
    };
    let comparison = Module::<poulpy_cpu_portable::NTT4x30Portable>::new(params.n as u64);
    let tested = Module::<Gpu>::new(params.n as u64);
    let tracer = Tracer::start(false);
    test_linear_transformation_powers_of_two_parity(&params, &ParityShapes::default(), &comparison, &tested);
    let (trace, _) = tracer.finish();
    for label in [Label::DftGather, Label::CnvScaleMany] {
        assert!(
            trace.launches.iter().any(|&(launched, _)| launched == label),
            "the parity suite did not launch {}",
            label.name()
        );
    }
}

/// The constant combination against its reference: one to four terms summed
/// inside the launch that normalizes them, every other constant shifted by a
/// sub-limb amount first, and a fifth term refused.
#[cfg(feature = "enable-core")]
#[test]
fn const_combination_matches_reference() {
    use poulpy_core::{oep::ConstTerm, reference::operations::glwe_const_combination_reference};
    let module = &*MODULE;
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 22);
    let (base2k, a_size, b_size, res_size) = (50, 6, 3, 5);
    for count in 1..=5usize {
        for (res_k, acc_size) in [(res_size * base2k, a_size + b_size), (res_size * base2k - 7, res_size)] {
            let mut seed = 0x0c0f_fee0_1234_5678u64 ^ ((count * 97 + res_k) as u64);
            let operands: Vec<_> = (0..count)
                .map(|_| vector::<poulpy_hal::layouts::Standard>(N, 2, a_size, &mut seed))
                .collect();
            let constants = vector::<poulpy_hal::layouts::Standard>(1, count, b_size, &mut seed);
            let terms: Vec<ConstTerm<'_, Gpu>> = operands
                .iter()
                .enumerate()
                .map(|(index, a)| ConstTerm {
                    a: vec_znx_backend_ref::<Gpu>(a),
                    b: vec_znx_backend_ref::<Gpu>(&constants),
                    b_col: index,
                    b_coeff: 0,
                    shift: (index % 2) * 13,
                    cnv_offset: index % 3,
                })
                .collect();
            let mut destination_seed = seed;
            let mut want = vector::<poulpy_hal::layouts::Standard>(N, 2, res_size, &mut destination_seed);
            let mut have = vector::<poulpy_hal::layouts::Standard>(N, 2, res_size, &mut seed);
            glwe_const_combination_reference(
                module,
                &mut vec_znx_backend_mut::<Gpu>(&mut want),
                base2k,
                res_k,
                acc_size,
                &terms,
                &mut scratch.arena(),
            );
            let fused = crate::core_impl::shifted_const_combination(
                module,
                &mut vec_znx_backend_mut::<Gpu>(&mut have),
                base2k,
                res_k,
                acc_size,
                &terms,
                &mut scratch.arena(),
            );
            assert_eq!(fused, count <= 4, "the kernel takes up to four terms");
            if fused {
                assert!(
                    want.data() == have.data(),
                    "constant combination differs from its reference: terms={count} res_k={res_k} acc_size={acc_size}"
                );
            }
        }
    }
}

/// The products of two and four sources against one key in one launch equal one launch per
/// source, bit for bit, at the full degree the fused inverse launch needs and at the widest
/// CoeffsToSlots shape: the product with the first inverse launch of the baby-step doubling,
/// and the product added, rotated and accumulated of the giant-step tree.
#[test]
fn many_source_products_match_one_launch_each_at_full_degree() {
    use poulpy_hal::api::ScratchArenaTakeBasic;
    let n = 1 << 16;
    let module = Module::<Gpu>::new(n as u64);
    let (a_size, dsize, rows, key_size, res_size, product_limbs) = (27, 4, 7, 33, 29, 3);
    let mut seed = 0x0dd5_0dd5_u64;
    let matrix = MatZnx::from_data(
        Gpu::from_host_bytes(&coefficients(n * rows * 2 * key_size, &mut seed)),
        n,
        rows,
        1,
        2,
        key_size,
    );
    let mut pmat = module.vmp_pmat_alloc(n, rows, 1, 2, key_size, PrepareHint::Reuse);
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 24);
    module.vmp_prepare(
        &mut pmat.to_backend_mut(),
        &poulpy_hal::layouts::MatZnxToBackendRef::<Gpu>::to_backend_ref(&matrix),
        &mut scratch.arena(),
    );
    // Both sides read through a copy, the scratch views as the owned vectors.
    let host = |module: &Module<Gpu>, view: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Gpu>, cols: usize| {
        let mut owned = module.vec_znx_dft_alloc(n, cols, view.size());
        for col in 0..cols {
            module.vec_znx_dft_copy(1, 0, &mut owned.to_backend_mut(), col, view, col);
        }
        DataView::data(&owned).to_host()
    };
    let plan = crate::ntt4x30::AutomorphismPlan::new(n, 5, false);
    for m in [2usize, 3, 4] {
        let sources: Vec<_> = (0..m).map(|_| transformed(&module, n, 1, a_size, &mut seed)).collect();
        let bodies: Vec<_> = (0..m).map(|_| transformed(&module, n, 2, res_size, &mut seed)).collect();
        let starts: Vec<_> = (0..m).map(|_| transformed(&module, n, 2, res_size, &mut seed)).collect();
        let mut arena_owned = ScratchOwned::<Gpu>::alloc(m * n * 16 * (a_size + 4 * res_size) + (1 << 20));
        let arena = arena_owned.arena();
        let (mut a, arena) = arena.take_vec_znx_dft_slice_scratch(n, m, 1, a_size);
        let (mut res, arena) = arena.take_vec_znx_dft_slice_scratch(n, m, 2, res_size);
        let (mut add, _) = arena.take_vec_znx_dft_slice_scratch(n, m, 2, res_size);
        for i in 0..m {
            module.vec_znx_dft_copy(1, 0, &mut a[i].to_backend_mut(), 0, &sources[i].to_backend_ref(), 0);
            for col in 0..2 {
                module.vec_znx_dft_copy(1, 0, &mut add[i].to_backend_mut(), col, &bodies[i].to_backend_ref(), col);
            }
        }
        let a_refs: Vec<_> = a.iter().map(|a| a.to_backend_ref()).collect();

        // The product with the first inverse launch.
        {
            let mut res_mut: Vec<_> = res.iter_mut().map(|res| res.to_backend_mut()).collect();
            assert!(crate::hal_impl::gglwe_products_digits_strided_inverse(
                &module,
                &mut res_mut,
                &a_refs,
                dsize,
                product_limbs,
                &pmat.to_backend_ref(),
            ));
        }
        for i in 0..m {
            let mut want = module.vec_znx_dft_alloc(n, 2, res_size);
            assert!(crate::hal_impl::gglwe_product_digits_strided_inverse(
                &module,
                &mut want.to_backend_mut(),
                &sources[i].to_backend_ref(),
                dsize,
                product_limbs,
                &pmat.to_backend_ref(),
                [None, None],
            ));
            assert!(
                host(&module, &res[i].to_backend_ref(), 2) == host(&module, &want.to_backend_ref(), 2),
                "product with the inverse launch, source {i} of {m}"
            );
        }

        // The product added, rotated and accumulated.
        for i in 0..m {
            for col in 0..2 {
                module.vec_znx_dft_copy(1, 0, &mut res[i].to_backend_mut(), col, &starts[i].to_backend_ref(), col);
            }
        }
        {
            let add_refs: Vec<_> = add.iter().map(|add| add.to_backend_ref()).collect();
            let add_refs: Vec<_> = add_refs.iter().collect();
            let mut res_mut: Vec<_> = res.iter_mut().map(|res| res.to_backend_mut()).collect();
            assert!(crate::hal_impl::gglwe_products_fused(
                &mut res_mut,
                res_size,
                &a_refs,
                dsize,
                product_limbs,
                &pmat.to_backend_ref(),
                Some(&add_refs),
                Some(&plan),
                true,
            ));
        }
        for i in 0..m {
            let mut want = module.vec_znx_dft_alloc(n, 2, res_size);
            for col in 0..2 {
                module.vec_znx_dft_copy(1, 0, &mut want.to_backend_mut(), col, &starts[i].to_backend_ref(), col);
            }
            crate::hal_impl::gglwe_product_fused(
                &mut want.to_backend_mut(),
                res_size,
                &sources[i].to_backend_ref(),
                dsize,
                product_limbs,
                &pmat.to_backend_ref(),
                [Some((&bodies[i].to_backend_ref(), 0)), None],
                Some(&plan),
                true,
            );
            assert!(
                host(&module, &res[i].to_backend_ref(), 2) == host(&module, &want.to_backend_ref(), 2),
                "product added, rotated and accumulated, source {i} of {m}"
            );
        }
    }
}
