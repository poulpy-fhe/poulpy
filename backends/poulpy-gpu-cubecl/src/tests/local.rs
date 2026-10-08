//! Tests of what no upstream suite reaches: the two crate-internal products
//! against their reference compositions, and views the suites do not build.

use once_cell::sync::Lazy;
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftSubAssign,
        VecZnxIdftApply, VecZnxIdftNormalize, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{
        Backend, DataView, MatZnx, Module, PrepareHint, Ring, ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxDftOwned,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxShape, VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxView,
        vec_znx_dft_backend_ref_from_mut,
    },
    test_suite::{download_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

type BE<R> = crate::NTT4x30Wgpu<R>;
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
) {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ ((a_size * 131 + dsize * 17 + rows * 7 + res_size) as u64);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(1 << 22);
    let a = transformed(module, n, cols_in, a_size, &mut seed);
    let matrix = MatZnx::from_data(
        BE::<R>::from_host_bytes(&coefficients(n * rows * cols_in * cols_out * key_size, &mut seed)),
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
        );
    }
    digits_case(&MODULE, N, 27, 4, 7, 1, 2, 33, 30, 2);
    digits_case(&MODULE_CI, 16, 27, 4, 7, 1, 2, 33, 30, 2);
}

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
) {
    let mut seed = 0x0fed_cba9_8765_4321u64 ^ ((a_size * 31 + b_size * 7 + diag_size * 3 + cross_size + offset) as u64);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(1 << 22);
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

    // `glwe_mul_columns_reference` of `poulpy-core`.
    {
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
        offset,
        &mut diag_have.to_backend_mut(),
        &mut cross_have.to_backend_mut(),
        &left,
        &right,
    );
    let shape = format!("n={n} b_n={b_n} a_size={a_size} b_size={b_size} diag={diag_size} cross={cross_size} offset={offset}");
    assert!(
        diag_want == diag_have,
        "diagonal products differ from their reference: {shape}"
    );
    assert!(cross_want == cross_have, "cross product differs from its reference: {shape}");
}

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
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 16);
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
    let mut scratch = ScratchOwned::<Gpu>::alloc(1 << 16);
    // (coeff_offset, limb_offset, limb_step, limbs) inside vectors of degree `N`.
    for (coeff_offset, limb_offset, limb_step, limbs) in [(N / 4, 1, 2, 2), (0, 0, 1, 3), (N / 2, 2, 3, 2)] {
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
