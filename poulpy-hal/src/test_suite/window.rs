//! Window views: every coefficient-wise operation applied through a window
//! equals the same operation applied to a dense copy of the window, and
//! leaves everything outside the window untouched.

use crate::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, VecZnxAdd, VecZnxAutomorphism, VecZnxBigAdd, VecZnxBigFromSmall,
        VecZnxBigNegate, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxCopy, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxLsh, VecZnxLshAdd, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxNegate, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRsh, VecZnxRshAdd, VecZnxRshAssign, VecZnxRshSub,
        VecZnxRshTmpBytes, VecZnxSub, VecZnxZero,
    },
    layouts::{
        Backend, CnvPVecLToBackendMut, DataView, FillUniform, HostBytesBackend, HostDataRef, Module, PrepareHint, ScratchOwned,
        VecZnx, VecZnxBig, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxOwned,
        VecZnxShape, ZnxView, ZnxViewMut, ZnxWord,
    },
    source::Source,
    test_suite::{TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

/// The windows exercised, as (coeff_offset, coeff_len, limb_offset, limb_step, limb_count).
fn windows(n: usize, size: usize) -> Vec<(usize, usize, usize, usize, usize)> {
    vec![
        (0, n, 0, 1, size),         // dense
        (3, 5, 0, 1, size),         // coefficient window only
        (0, n, 1, 2, size / 2),     // strided limb window only
        (n / 2, n / 4, 1, 2, 1),    // both
        (n - 1, 1, size - 1, 1, 1), // single coefficient, single limb
    ]
}

fn shape_of(base: VecZnxShape, w: (usize, usize, usize, usize, usize)) -> VecZnxShape {
    base.window_coeffs(w.0, w.1).window_limbs(w.2, w.3, w.4)
}

/// Dense copy of the elements a host container shows through `shape`.
fn materialize<W: ZnxWord>(host: &VecZnx<impl HostDataRef, W>, shape: VecZnxShape) -> VecZnxOwned<W> {
    let view = VecZnx::<&[u8], W>::from_shape(host.data().as_ref(), shape);
    let mut out = VecZnxOwned::<W>::alloc(shape.n(), shape.cols(), shape.size());
    for j in 0..shape.size() {
        for i in 0..shape.cols() {
            out.at_mut(i, j).copy_from_slice(view.at(i, j));
        }
    }
    out
}

/// Asserts `after == before` at every element outside `shape`.
fn assert_untouched_outside<T: ZnxView>(before: &T, after: &T, shape: VecZnxShape)
where
    T::Scalar: PartialEq,
{
    let sel_limbs: Vec<usize> = (0..shape.size())
        .map(|j| shape.limb_offset() + j * shape.limb_step())
        .collect();
    for j in 0..before.size() {
        for i in 0..before.cols() {
            for k in 0..before.n() {
                let inside = sel_limbs.contains(&j) && (shape.coeff_offset()..shape.coeff_offset() + shape.n()).contains(&k);
                if !inside {
                    assert_eq!(
                        before.at(i, j)[k],
                        after.at(i, j)[k],
                        "outside-window element changed at col {i} limb {j} coeff {k}"
                    );
                }
            }
        }
    }
}

pub fn test_vec_znx_window_ops<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxZero<BE> + VecZnxCopy<BE> + VecZnxAdd<BE> + VecZnxSub<BE> + VecZnxNegate<BE>,
{
    let n = params.size;
    let base2k = params.base2k;
    let (cols, size) = (2usize, 4usize);
    let mut source = Source::new([7u8; 32]);
    let base = VecZnxShape::new(n, cols, size);

    for w in windows(n, size) {
        let shape = shape_of(base, w);
        for op in 0..5 {
            let mut a = VecZnxOwned::<i64>::alloc(n, cols, size);
            let mut b = VecZnxOwned::<i64>::alloc(n, cols, size);
            let mut res = VecZnxOwned::<i64>::alloc(n, cols, size);
            a.fill_uniform(base2k, &mut source);
            b.fill_uniform(base2k, &mut source);
            res.fill_uniform(base2k, &mut source);
            let res_before = res.clone();

            // Dense oracle on materialized windows.
            let am = materialize(&a, shape);
            let bm = materialize(&b, shape);
            let rm = materialize(&res, shape);
            let am_be = upload_vec_znx::<BE>(&am);
            let bm_be = upload_vec_znx::<BE>(&bm);
            let mut rm_be = upload_vec_znx::<BE>(&rm);

            // Windowed run.
            let a_be = upload_vec_znx::<BE>(&a);
            let b_be = upload_vec_znx::<BE>(&b);
            let mut res_be = upload_vec_znx::<BE>(&res);

            for col in 0..cols {
                match op {
                    0 => {
                        module.vec_znx_zero(&mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape), col);
                        module.vec_znx_zero(&mut vec_znx_backend_mut::<BE>(&mut rm_be), col);
                    }
                    1 => {
                        module.vec_znx_copy(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_copy(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                        );
                    }
                    2 => {
                        module.vec_znx_add(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_add(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&bm_be),
                            col,
                        );
                    }
                    3 => {
                        module.vec_znx_sub(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_sub(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&bm_be),
                            col,
                        );
                    }
                    _ => {
                        module.vec_znx_negate(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_negate(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                        );
                    }
                }
            }

            let res_after = download_vec_znx::<BE>(&res_be);
            let rm_after = download_vec_znx::<BE>(&rm_be);
            assert_eq!(
                materialize(&res_after, shape),
                rm_after,
                "op {op} window {w:?}: windowed result differs from dense oracle"
            );
            assert_untouched_outside(&res_before, &res_after, shape);
        }
    }
}

/// Ring operations require `n == n_full == N` and must reject a coefficient
/// window rather than silently compute in `Z[X]/(X^n+1)` for the window's
/// `n`. Cases: `vec_znx_rotate`, `vec_znx_automorphism`,
/// `vec_znx_dft_apply` and `cnv_prepare_left` panic on a windowed input, while
/// a coefficient-wise op (`vec_znx_add`) keeps accepting it.
pub fn test_vec_znx_window_rejected_by_ring_ops<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRotate<BE>
        + VecZnxAutomorphism<BE>
        + VecZnxAdd<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>,
{
    let n = params.size;
    let base2k = params.base2k;
    let (cols, size) = (2usize, 4usize);
    let mut source = Source::new([11u8; 32]);
    let shape = VecZnxShape::new(n, cols, size).window_coeffs(0, 4);

    let mut a = VecZnxOwned::<i64>::alloc(n, cols, size);
    let mut b = VecZnxOwned::<i64>::alloc(n, cols, size);
    let mut res = VecZnxOwned::<i64>::alloc(n, cols, size);
    a.fill_uniform(base2k, &mut source);
    b.fill_uniform(base2k, &mut source);
    res.fill_uniform(base2k, &mut source);

    let a_be = upload_vec_znx::<BE>(&a);
    let b_be = upload_vec_znx::<BE>(&b);
    let mut res_be = upload_vec_znx::<BE>(&res);

    let rotate_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.vec_znx_rotate(
            1,
            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
            0,
            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
            0,
        );
    }))
    .is_err();
    assert!(
        rotate_panicked,
        "vec_znx_rotate accepted a windowed view instead of panicking"
    );

    let automorphism_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.vec_znx_automorphism(
            5,
            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
            0,
            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
            0,
        );
    }))
    .is_err();
    assert!(
        automorphism_panicked,
        "vec_znx_automorphism accepted a windowed view instead of panicking"
    );

    let mut dft = module.vec_znx_dft_alloc(cols, size);
    let dft_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut dft.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
            0,
        );
    }))
    .is_err();
    assert!(
        dft_panicked,
        "vec_znx_dft_apply accepted a windowed view instead of panicking"
    );

    let mut prep = module.cnv_pvec_left_alloc(cols, size, PrepareHint::Reuse);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_left_tmp_bytes(size, size));
    let cnv_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_left(
            &mut prep.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
            !0i64,
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(cnv_panicked, "cnv_prepare_left accepted a windowed view instead of panicking");

    let add_into_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.vec_znx_add(
            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
            0,
            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
            0,
            &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
            0,
        );
    }))
    .is_ok();
    assert!(add_into_ok, "vec_znx_add rejected a windowed view unexpectedly");
}

fn download_big<BE: Backend>(v: &VecZnxBigOwned<BE>) -> VecZnxBig<Vec<u8>, BE::BigWord, BE> {
    let host_bytes = BE::to_host_bytes(v.data());
    VecZnxBig::from_shape(HostBytesBackend::from_host_bytes(&host_bytes), v.shape())
}

pub fn test_vec_znx_big_window_ops<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxBigFromSmall<BE> + VecZnxBigAdd<BE> + VecZnxBigSub<BE> + VecZnxBigNegate<BE>,
    BE::BigWord: PartialEq + std::fmt::Debug,
{
    let n = params.size;
    let base2k = params.base2k;
    let (cols, size) = (2usize, 4usize);
    let mut source = Source::new([9u8; 32]);
    let base = VecZnxShape::new(n, cols, size);

    for w in windows(n, size) {
        let shape = shape_of(base, w);
        let mut a = VecZnxOwned::<i64>::alloc(n, cols, size);
        let mut b = VecZnxOwned::<i64>::alloc(n, cols, size);
        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);
        let (am, bm) = (materialize(&a, shape), materialize(&b, shape));

        // Windowed path: promote through windows, then add / sub / negate through windows.
        let a_be = upload_vec_znx::<BE>(&a);
        let b_be = upload_vec_znx::<BE>(&b);
        let mut big_a = VecZnxBigOwned::<BE>::alloc(n, cols, size);
        let mut big_b = VecZnxBigOwned::<BE>::alloc(n, cols, size);
        let mut big_r = VecZnxBigOwned::<BE>::alloc(n, cols, size);
        // Seed the whole dense buffers with unrelated content, so that a kernel
        // running past the window shows up as a changed outside-window element.
        for big in [&mut big_a, &mut big_b, &mut big_r] {
            let mut seed = VecZnxOwned::<i64>::alloc(n, cols, size);
            seed.fill_uniform(base2k, &mut source);
            let seed_be = upload_vec_znx::<BE>(&seed);
            for col in 0..cols {
                module.vec_znx_big_from_small(&mut big.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&seed_be), col);
            }
        }
        let (seed_a, seed_b, seed_r) = (
            download_big::<BE>(&big_a),
            download_big::<BE>(&big_b),
            download_big::<BE>(&big_r),
        );
        // Dense oracle on the materialized windows.
        let am_be = upload_vec_znx::<BE>(&am);
        let bm_be = upload_vec_znx::<BE>(&bm);
        let mut big_am = VecZnxBigOwned::<BE>::alloc(shape.n(), cols, shape.size());
        let mut big_bm = VecZnxBigOwned::<BE>::alloc(shape.n(), cols, shape.size());
        let mut big_rm = VecZnxBigOwned::<BE>::alloc(shape.n(), cols, shape.size());

        for col in 0..cols {
            module.vec_znx_big_from_small(
                &mut big_a.to_backend_mut().with_shape(shape),
                col,
                &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                col,
            );
            module.vec_znx_big_from_small(
                &mut big_b.to_backend_mut().with_shape(shape),
                col,
                &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                col,
            );
            module.vec_znx_big_add(
                &mut big_r.to_backend_mut().with_shape(shape),
                col,
                &big_a.to_backend_ref().with_shape(shape),
                col,
                &big_b.to_backend_ref().with_shape(shape),
                col,
            );
            module.vec_znx_big_sub(
                &mut big_a.to_backend_mut().with_shape(shape),
                col,
                &big_r.to_backend_ref().with_shape(shape),
                col,
                &big_b.to_backend_ref().with_shape(shape),
                col,
            );
            module.vec_znx_big_negate(
                &mut big_b.to_backend_mut().with_shape(shape),
                col,
                &big_a.to_backend_ref().with_shape(shape),
                col,
            );

            module.vec_znx_big_from_small(&mut big_am.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&am_be), col);
            module.vec_znx_big_from_small(&mut big_bm.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&bm_be), col);
            module.vec_znx_big_add(
                &mut big_rm.to_backend_mut(),
                col,
                &big_am.to_backend_ref(),
                col,
                &big_bm.to_backend_ref(),
                col,
            );
            module.vec_znx_big_sub(
                &mut big_am.to_backend_mut(),
                col,
                &big_rm.to_backend_ref(),
                col,
                &big_bm.to_backend_ref(),
                col,
            );
            module.vec_znx_big_negate(&mut big_bm.to_backend_mut(), col, &big_am.to_backend_ref(), col);
        }

        let (after_a, after_b, after_r) = (
            download_big::<BE>(&big_a),
            download_big::<BE>(&big_b),
            download_big::<BE>(&big_r),
        );
        assert_untouched_outside(&seed_a, &after_a, shape);
        assert_untouched_outside(&seed_b, &after_b, shape);
        assert_untouched_outside(&seed_r, &after_r, shape);

        let hr = after_r.with_shape(shape);
        let hb = after_b.with_shape(shape);
        let hrm = download_big::<BE>(&big_rm);
        let hbm = download_big::<BE>(&big_bm);
        for j in 0..shape.size() {
            for col in 0..cols {
                assert_eq!(hr.at(col, j), hrm.at(col, j), "big add window {w:?} col {col} limb {j}");
                assert_eq!(
                    hb.at(col, j),
                    hbm.at(col, j),
                    "big sub/negate window {w:?} col {col} limb {j}"
                );
            }
        }
    }
}

/// `normalize`, `normalize_assign` and the shift family applied through a
/// window equal the same operation on the materialized window and leave every
/// element outside the window untouched.
pub fn test_vec_znx_window_normalize_ops<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxNormalize<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRsh<BE>
        + VecZnxLshAdd<BE>
        + VecZnxRshAdd<BE>
        + VecZnxLshSub<BE>
        + VecZnxRshSub<BE>
        + VecZnxLshAssign<BE>
        + VecZnxRshAssign<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let n = params.size;
    let base2k = params.base2k;
    let (cols, size) = (2usize, 4usize);
    let mut source = Source::new([11u8; 32]);
    let base = VecZnxShape::new(n, cols, size);
    let tmp_bytes = module
        .vec_znx_normalize_tmp_bytes()
        .max(module.vec_znx_lsh_tmp_bytes())
        .max(module.vec_znx_rsh_tmp_bytes());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);

    for w in windows(n, size) {
        let shape = shape_of(base, w);
        let wsize = shape.size();
        for op in 0..11 {
            for k in [0usize, 3, base2k, base2k + 2] {
                let mut a = VecZnxOwned::<i64>::alloc(n, cols, size);
                let mut res = VecZnxOwned::<i64>::alloc(n, cols, size);
                a.fill_uniform(base2k, &mut source);
                res.fill_uniform(base2k, &mut source);
                let res_before = res.clone();

                let am = materialize(&a, shape);
                let rm = materialize(&res, shape);
                let am_be = upload_vec_znx::<BE>(&am);
                let mut rm_be = upload_vec_znx::<BE>(&rm);
                let a_be = upload_vec_znx::<BE>(&a);
                let mut res_be = upload_vec_znx::<BE>(&res);

                // normalize parameters derived from k: partial precision and a signed offset.
                let res_offset = k as i64 - base2k as i64;
                let cross_base2k = base2k - 1;
                let res_k_inter = wsize * base2k - (k % base2k);
                let res_k_cross = wsize * cross_base2k - (k % cross_base2k);

                for col in 0..cols {
                    // Windowed run on (res_be, a_be), dense run on (rm_be, am_be).
                    macro_rules! both {
                        (|$r:ident, $a:ident| $body:expr) => {{
                            {
                                let mut $r = vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape);
                                let $a = vec_znx_backend_ref::<BE>(&a_be).with_shape(shape);
                                $body;
                            }
                            {
                                let mut $r = vec_znx_backend_mut::<BE>(&mut rm_be);
                                let $a = vec_znx_backend_ref::<BE>(&am_be);
                                $body;
                            }
                        }};
                    }
                    match op {
                        0 => both!(|r, a| module.vec_znx_normalize(
                            &mut r,
                            base2k,
                            res_k_inter,
                            res_offset,
                            col,
                            &a,
                            base2k,
                            col,
                            &mut scratch.arena()
                        )),
                        1 => both!(|r, a| module.vec_znx_normalize(
                            &mut r,
                            cross_base2k,
                            res_k_cross,
                            res_offset,
                            col,
                            &a,
                            base2k,
                            col,
                            &mut scratch.arena()
                        )),
                        2 => {
                            both!(|r, _a| module.vec_znx_normalize_assign(base2k, res_k_inter, &mut r, col, &mut scratch.arena()))
                        }
                        3 => both!(|r, a| module.vec_znx_lsh(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        4 => both!(|r, a| module.vec_znx_rsh(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        5 => both!(|r, a| module.vec_znx_lsh_add(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        6 => both!(|r, a| module.vec_znx_rsh_add(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        7 => both!(|r, a| module.vec_znx_lsh_sub(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        8 => both!(|r, a| module.vec_znx_rsh_sub(base2k, k, &mut r, col, &a, col, &mut scratch.arena())),
                        9 => both!(|r, _a| module.vec_znx_lsh_assign(base2k, k, &mut r, col, &mut scratch.arena())),
                        _ => both!(|r, _a| module.vec_znx_rsh_assign(base2k, k, &mut r, col, &mut scratch.arena())),
                    }
                }

                let res_after = download_vec_znx::<BE>(&res_be);
                assert_eq!(
                    materialize(&res_after, shape),
                    download_vec_znx::<BE>(&rm_be),
                    "window {w:?} op {op} k {k}: windowed result differs from the dense oracle"
                );
                assert_untouched_outside(&res_before, &res_after, shape);
            }
        }
    }
}

/// `vec_znx_big_normalize` through windows on both operands.
pub fn test_vec_znx_big_window_normalize<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxBigFromSmall<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let n = params.size;
    let base2k = params.base2k;
    let (cols, size) = (2usize, 4usize);
    let mut source = Source::new([17u8; 32]);
    let base = VecZnxShape::new(n, cols, size);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    for w in windows(n, size) {
        let shape = shape_of(base, w);
        let wsize = shape.size();
        for (res_offset, res_k) in [(-3i64, wsize * base2k), (0, wsize * base2k - 2), (5, wsize * base2k)] {
            let mut a = VecZnxOwned::<i64>::alloc(n, cols, size);
            let mut res = VecZnxOwned::<i64>::alloc(n, cols, size);
            a.fill_uniform(base2k, &mut source);
            res.fill_uniform(base2k, &mut source);
            let res_before = res.clone();
            let am = materialize(&a, shape);
            let rm = materialize(&res, shape);

            // Dense big operands: big_a holds the whole box, big_am the materialized window.
            let a_be = upload_vec_znx::<BE>(&a);
            let am_be = upload_vec_znx::<BE>(&am);
            let mut big_a = VecZnxBigOwned::<BE>::alloc(n, cols, size);
            let mut big_am = VecZnxBigOwned::<BE>::alloc(shape.n(), cols, wsize);
            for col in 0..cols {
                module.vec_znx_big_from_small(&mut big_a.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&a_be), col);
                module.vec_znx_big_from_small(&mut big_am.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&am_be), col);
            }
            let mut res_be = upload_vec_znx::<BE>(&res);
            let mut rm_be = upload_vec_znx::<BE>(&rm);
            for col in 0..cols {
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                    base2k,
                    res_k,
                    res_offset,
                    col,
                    &big_a.to_backend_ref().with_shape(shape),
                    base2k,
                    col,
                    &mut scratch.arena(),
                );
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                    base2k,
                    res_k,
                    res_offset,
                    col,
                    &big_am.to_backend_ref(),
                    base2k,
                    col,
                    &mut scratch.arena(),
                );
            }
            let res_after = download_vec_znx::<BE>(&res_be);
            assert_eq!(
                materialize(&res_after, shape),
                download_vec_znx::<BE>(&rm_be),
                "window {w:?} offset {res_offset} k {res_k}: windowed big normalize differs from the dense oracle"
            );
            assert_untouched_outside(&res_before, &res_after, shape);
        }
    }
}
