//! Window views: every coefficient-wise operation applied through a window
//! equals the same operation applied to a dense copy of the window, and
//! leaves everything outside the window untouched.

use crate::{
    api::{
        VecZnxAddIntoBackend, VecZnxBigAddInto, VecZnxBigFromSmallBackend, VecZnxBigNegate, VecZnxBigSub, VecZnxCopyBackend,
        VecZnxNegateBackend, VecZnxSubBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, DataView, FillUniform, HostBytesBackend, HostDataRef, Module, VecZnx, VecZnxBig, VecZnxBigOwned,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxOwned, VecZnxShape, ZnxView, ZnxViewMut, ZnxWord,
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
    Module<BE>:
        VecZnxZeroBackend<BE> + VecZnxCopyBackend<BE> + VecZnxAddIntoBackend<BE> + VecZnxSubBackend<BE> + VecZnxNegateBackend<BE>,
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
                        module.vec_znx_zero_backend(&mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape), col);
                        module.vec_znx_zero_backend(&mut vec_znx_backend_mut::<BE>(&mut rm_be), col);
                    }
                    1 => {
                        module.vec_znx_copy_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_copy_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                        );
                    }
                    2 => {
                        module.vec_znx_add_into_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_add_into_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&bm_be),
                            col,
                        );
                    }
                    3 => {
                        module.vec_znx_sub_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_sub_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut rm_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&am_be),
                            col,
                            &vec_znx_backend_ref::<BE>(&bm_be),
                            col,
                        );
                    }
                    _ => {
                        module.vec_znx_negate_backend(
                            &mut vec_znx_backend_mut::<BE>(&mut res_be).with_shape(shape),
                            col,
                            &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                            col,
                        );
                        module.vec_znx_negate_backend(
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

fn download_big<BE: Backend>(v: &VecZnxBigOwned<BE>) -> VecZnxBig<Vec<u8>, BE::BigWord, BE> {
    let host_bytes = BE::to_host_bytes(v.data());
    VecZnxBig::from_shape(HostBytesBackend::from_host_bytes(&host_bytes), v.shape())
}

pub fn test_vec_znx_big_window_ops<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxBigFromSmallBackend<BE> + VecZnxBigAddInto<BE> + VecZnxBigSub<BE> + VecZnxBigNegate<BE>,
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
                module.vec_znx_big_from_small_backend(&mut big.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&seed_be), col);
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
            module.vec_znx_big_from_small_backend(
                &mut big_a.to_backend_mut().with_shape(shape),
                col,
                &vec_znx_backend_ref::<BE>(&a_be).with_shape(shape),
                col,
            );
            module.vec_znx_big_from_small_backend(
                &mut big_b.to_backend_mut().with_shape(shape),
                col,
                &vec_znx_backend_ref::<BE>(&b_be).with_shape(shape),
                col,
            );
            module.vec_znx_big_add_into(
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

            module.vec_znx_big_from_small_backend(&mut big_am.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&am_be), col);
            module.vec_znx_big_from_small_backend(&mut big_bm.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&bm_be), col);
            module.vec_znx_big_add_into(
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
