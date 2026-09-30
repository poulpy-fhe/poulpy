//! The required HAL operations, written directly. Every optional operation
//! keeps its HAL-derived body, except the one noted on
//! [`HalVecZnxMonomialImpl`]. Temporaries live on the heap, so every scratch
//! size is zero.

use poulpy_hal::{
    layouts::{
        CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, MatZnxBackendRef, Module,
        ScalarZnxBackendRef, ScratchArena, Standard, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
        ZnxView, ZnxViewMut, assert_dense,
    },
    oep::{
        HalConvolutionImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxCIImpl, HalVecZnxDftImpl, HalVecZnxImpl,
        HalVecZnxMonomialImpl, HalVmpImpl,
    },
    source::Source,
};

use crate::{
    backend::Oracle,
    embed::{with_vec_znx, with_vec_znx_big},
    family::{Family, Int},
    limbs::{apply, apply_poly, map, map_poly, update, zip},
    normalize::{normalize, normalize_assign},
    ring::{self, OracleRing},
};

/// `res = X^p a` in `Z[X]/(X^n + 1)`.
fn rotate(p: i64, res: &mut [i64], a: &[i64]) {
    let n = a.len() as i64;
    for (i, &x) in a.iter().enumerate() {
        let k = (i as i64 + p).rem_euclid(2 * n);
        if k < n {
            res[k as usize] = x;
        } else {
            res[(k - n) as usize] = x.wrapping_neg();
        }
    }
}

/// Coefficient insertion or selection between degrees dividing one another.
fn switch_ring(res: &mut [i64], a: &[i64]) {
    if a.len() >= res.len() {
        let gap = a.len() / res.len();
        (0..res.len()).for_each(|i| res[i] = a[i * gap]);
    } else {
        let gap = res.len() / a.len();
        res.fill(0);
        (0..a.len()).for_each(|i| res[i * gap] = a[i]);
    }
}

/// The transform at degree `N = res.len()` of the degree embedding of the
/// degree-`n` polynomial whose transform is `a`.
fn lift<F: Family, R: OracleRing>(module: &Module<Oracle<F, R>>, res: &mut [F::Dft], a: &[F::Dft]) {
    let (n, big) = (a.len(), res.len());
    let mut coeffs = vec![F::Big::default(); n];
    ring::inverse(module, &mut coeffs, a);
    let mut spread = vec![0i64; big];
    for (i, c) in coeffs.into_iter().enumerate() {
        spread[i * (big / n)] = i64::try_from(c.into()).expect("prepared coefficient exceeds 64 bits");
    }
    ring::forward(module, res, &spread);
}

unsafe impl<F: Family, R: OracleRing> HalVecZnxImpl for Oracle<F, R> {
    fn vec_znx_zero(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize) {
        apply(res, res_col, |_| 0);
    }

    fn vec_znx_normalize_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col);
    }

    fn vec_znx_normalize_assign(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        normalize_assign(a, base2k, k, a_offset, a_col);
    }

    fn vec_znx_add(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let n = res.n();
        with_vec_znx::<Self, _>(a, n, |a| {
            with_vec_znx::<Self, _>(b, n, |b| zip(res, res_col, a, a_col, b, b_col, i64::add, |x| x, |y| y))
        });
    }

    fn vec_znx_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, i64::add, |r| r));
    }

    fn vec_znx_sub(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let n = res.n();
        with_vec_znx::<Self, _>(a, n, |a| {
            with_vec_znx::<Self, _>(b, n, |b| zip(res, res_col, a, a_col, b, b_col, i64::sub, |x| x, i64::neg))
        });
    }

    fn vec_znx_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, i64::sub, |r| r));
    }

    fn vec_znx_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, |r, x| x.sub(r), i64::neg));
    }

    fn vec_znx_negate(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        map(res, res_col, a, a_col, i64::neg);
    }

    fn vec_znx_negate_assign(_module: &Module<Self>, a: &mut VecZnxBackendMut<'_, Self>, a_col: usize) {
        apply(a, a_col, i64::neg);
    }

    fn vec_znx_automorphism(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_automorphism");
        assert_dense(a, "vec_znx_automorphism");
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| ring::automorphism::<R, _>(k, r, x));
    }

    fn vec_znx_automorphism_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_automorphism_assign(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(res, "vec_znx_automorphism_assign");
        apply_poly(res, res_col, |r, x| ring::automorphism::<R, _>(k, r, x));
    }

    fn vec_znx_switch_ring(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_switch_ring");
        assert_dense(a, "vec_znx_switch_ring");
        map_poly(res, res_col, a, a_col, switch_ring);
    }

    fn vec_znx_copy(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        map(res, res_col, a, a_col, |x| x);
    }

    // Digits centered in [-2^(base2k-1), 2^(base2k-1)), low bits of the last
    // limb cleared to precision k, limbs past k zero.
    fn vec_znx_fill_uniform(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        seed: [u8; 32],
    ) {
        assert!(k != 0, "uniform sampling precision must be non-zero");
        let size = k.div_ceil(base2k);
        assert!(size <= res.size(), "k ({k}) exceeds the allocation ({} limbs)", res.size());
        let mut source = Source::new(seed);
        let (range, half) = (1u64 << base2k, 1i64 << (base2k - 1));
        for j in 0..res.size() {
            let limb = res.at_mut(res_col, j);
            if j < size {
                limb.iter_mut()
                    .for_each(|x| *x = source.next_u64n(range, range - 1) as i64 - half);
            } else {
                limb.fill(0);
            }
        }
        if !k.is_multiple_of(base2k) {
            let mask = !0i64 << (base2k - k % base2k);
            res.at_mut(res_col, size - 1).iter_mut().for_each(|x| *x &= mask);
        }
    }
}

// The HAL-derived in-place multiplication by X^p - 1 stages a full-size
// temporary in scratch, which Core's in-place callers do not provide. This
// override works limb by limb on the heap.
unsafe impl<F: Family> HalVecZnxMonomialImpl for Oracle<F, Standard> {
    fn vec_znx_rotate(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_rotate");
        assert_dense(a, "vec_znx_rotate");
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| rotate(k, r, x));
    }

    fn vec_znx_rotate_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_rotate_assign(
        _module: &Module<Self>,
        k: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(a, "vec_znx_rotate_assign");
        apply_poly(a, a_col, |r, x| rotate(k, r, x));
    }

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(_module: &Module<Self>, _size: usize) -> usize {
        0
    }

    fn vec_znx_mul_xp_minus_one_assign(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(res, "vec_znx_mul_xp_minus_one_assign");
        apply_poly(res, res_col, |r, x| {
            rotate(k, r, x);
            (0..r.len()).for_each(|i| r[i] = r[i].wrapping_sub(x[i]));
        });
    }
}

unsafe impl<F: Family, R: OracleRing> HalVecZnxBigImpl for Oracle<F, R> {
    fn vec_znx_big_from_small(
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| map(res, res_col, a, a_col, F::Big::from));
    }

    fn vec_znx_big_add(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let n = res.n();
        with_vec_znx_big::<Self, _>(a, n, |a| {
            with_vec_znx_big::<Self, _>(b, n, |b| zip(res, res_col, a, a_col, b, b_col, F::Big::add, |x| x, |y| y))
        });
    }

    fn vec_znx_big_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx_big::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, F::Big::add, |r| r));
    }

    fn vec_znx_big_add_small_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, |r, x| r.add(x.into()), |r| r));
    }

    fn vec_znx_big_sub(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let n = res.n();
        with_vec_znx_big::<Self, _>(a, n, |a| {
            with_vec_znx_big::<Self, _>(b, n, |b| {
                zip(res, res_col, a, a_col, b, b_col, F::Big::sub, |x| x, F::Big::neg)
            })
        });
    }

    fn vec_znx_big_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx_big::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, F::Big::sub, |r| r));
    }

    fn vec_znx_big_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx_big::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, |r, x| x.sub(r), F::Big::neg));
    }

    fn vec_znx_big_sub_small_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| update(res, res_col, a, a_col, |r, x| r.sub(x.into()), |r| r));
    }

    fn vec_znx_big_sub_small_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        with_vec_znx::<Self, _>(a, res.n(), |a| {
            update(res, res_col, a, a_col, |r, x| F::Big::from(x).sub(r), F::Big::neg)
        });
    }

    // res[res_col, j, res_coeff] = sum_i a[a_col, j, i]
    fn vec_znx_big_inner_sum(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(res_coeff < res.n());
        assert!(res.size() <= a.size());
        for j in 0..res.size() {
            let sum = a.at(a_col, j).iter().fold(F::Big::default(), |acc, &x| acc.add(x));
            res.at_mut(res_col, j)[res_coeff] = sum;
        }
    }

    // res[res_col, j, i] = sum_{c < cols} weights[c] a[c, j, i] for i < coeffs
    fn vec_znx_big_col_weighted_sum(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        weights: &ScalarZnxBackendRef<'_, Self>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) {
        assert!(cols <= a.cols() && cols <= weights.n() && weights_col < weights.cols());
        assert!(coeffs <= a.n() && coeffs <= res.n() && res.size() <= a.size());
        let weights = weights.at(weights_col, 0);
        for j in 0..res.size() {
            let out = res.at_mut(res_col, j);
            out.fill(F::Big::default());
            for (c, &w) in weights.iter().enumerate().take(cols) {
                let x = a.at(c, j);
                (0..coeffs).for_each(|i| out[i] = out[i].add(F::Big::from(x[i]).mul(w.into())));
            }
        }
    }

    // res[res_col, j, i] = a[a_col, j, i] b[b_col, 0, i]
    fn vec_znx_scalar_product(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let n = a.n();
        assert!(n == b.n() && res.n() >= n && res.size() <= a.size());
        let b = b.at(b_col, 0);
        for j in 0..res.size() {
            let (x, out) = (a.at(a_col, j), res.at_mut(res_col, j));
            (0..n).for_each(|i| out[i] = F::Big::from(x[i]).mul(b[i].into()));
        }
    }

    fn vec_znx_big_negate(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        map(res, res_col, a, a_col, F::Big::neg);
    }

    fn vec_znx_big_negate_assign(_module: &Module<Self>, a: &mut VecZnxBigBackendMut<'_, Self>, a_col: usize) {
        apply(a, a_col, F::Big::neg);
    }

    fn vec_znx_big_normalize_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_big_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col);
    }

    fn vec_znx_big_automorphism(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_big_automorphism");
        assert_dense(a, "vec_znx_big_automorphism");
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| ring::automorphism::<R, _>(k, r, x));
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_big_automorphism_assign(
        _module: &Module<Self>,
        k: i64,
        a: &mut VecZnxBigBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(a, "vec_znx_big_automorphism_assign");
        apply_poly(a, a_col, |r, x| ring::automorphism::<R, _>(k, r, x));
    }
}

/// Maps between the conjugate-invariant ring of degree `N` and the standard
/// ring of degree `2N`, from their HAL definitions, on the standard oracle.
unsafe impl<F: Family> HalVecZnxCIImpl for Oracle<F, Standard> {
    // res = a_0 + sum_{0<i<N} a_i (X^i + X^-i), with X^-i = -X^(2N-i)
    fn vec_znx_ci_embed(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_ci_embed");
        assert_dense(a, "vec_znx_ci_embed");
        assert_eq!(res.n(), 2 * a.n());
        map_poly(res, res_col, a, a_col, |r, x| {
            let n = x.len();
            r.fill(0);
            r[0] = x[0];
            for i in 1..n {
                r[i] = x[i];
                r[2 * n - i] = x[i].wrapping_neg();
            }
        });
    }

    // res_0 = 2 a_0 and res_i = a_i - a_(2N-i): the coordinates of a(X) + a(X^-1)
    fn vec_znx_ci_trace(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_ci_trace");
        assert_dense(a, "vec_znx_ci_trace");
        assert_eq!(a.n(), 2 * res.n());
        map_poly(res, res_col, a, a_col, |r, x| {
            let n = r.len();
            r[0] = x[0].wrapping_mul(2);
            for i in 1..n {
                r[i] = x[i].wrapping_sub(x[2 * n - i]);
            }
        });
    }
}

/// An automorphism of the transformed domain: the exponent is all it needs.
pub struct DftAutomorphismPlan {
    p: i64,
}

unsafe impl<F: Family, R: OracleRing> HalVecZnxDftImpl for Oracle<F, R> {
    // res[j] = DFT(a[offset + j step]) while that limb exists, zero after.
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
        assert_dense(a, "vec_znx_dft_apply");
        assert_eq!(res.n(), a.n());
        for j in 0..res.size() {
            let limb = offset + j * step;
            if j < a.size().div_ceil(step) && limb < a.size() {
                ring::forward(module, res.at_mut(res_col, j), a.at(a_col, limb));
            } else {
                res.at_mut(res_col, j).fill(F::Dft::default());
            }
        }
    }

    fn vec_znx_idft_apply_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(res, "vec_znx_idft_apply");
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| ring::inverse(module, r, x));
    }

    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(res, "vec_znx_idft_apply_tmpa");
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| ring::inverse(module, r, x));
    }

    fn vec_znx_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        zip(res, res_col, a, a_col, b, b_col, F::dft_add, |x| x, |y| y);
    }

    fn vec_znx_dft_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        update(res, res_col, a, a_col, F::dft_add, |r| r);
    }

    fn vec_znx_dft_sub(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        zip(
            res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            |x, y| F::dft_add(x, F::dft_neg(y)),
            |x| x,
            F::dft_neg,
        );
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        update(res, res_col, a, a_col, |r, x| F::dft_add(r, F::dft_neg(x)), |r| r);
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        update(res, res_col, a, a_col, |r, x| F::dft_add(x, F::dft_neg(r)), F::dft_neg);
    }

    // res[j] = a[offset + j step] while that limb exists, zero after.
    fn vec_znx_dft_copy(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(step >= 1, "vec_znx_dft_copy: step must be >= 1");
        assert_eq!(res.n(), a.n());
        for j in 0..res.size() {
            let limb = offset + j * step;
            if j < a.size().div_ceil(step) && limb < a.size() {
                res.at_mut(res_col, j).copy_from_slice(a.at(a_col, limb));
            } else {
                res.at_mut(res_col, j).fill(F::Dft::default());
            }
        }
    }

    fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        apply(res, res_col, |_| F::Dft::default());
    }

    type AutomorphismPlan = DftAutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(_module: &Module<Self>, _n: usize, p: i64) -> DftAutomorphismPlan {
        DftAutomorphismPlan { p }
    }

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &DftAutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_eq!(res.n(), a.n());
        map_poly(res, res_col, a, a_col, |r, x| ring::dft_automorphism::<F, R>(plan.p, r, x));
    }
}

unsafe impl<F: Family, R: OracleRing> HalSvpImpl for Oracle<F, R> {
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_eq!(res.n(), a.n());
        ring::forward(module, res.at_mut(res_col, 0), a.at(a_col, 0));
    }

    fn svp_ppol_copy(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_eq!(res.hint(), a.hint());
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_apply_dft_to_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let a = a.at(a_col, 0);
        map_poly(res, res_col, b, b_col, |r, x| {
            r.fill(F::Dft::default());
            ring::mul_acc::<F, R>(r, a, x);
        });
    }

    fn svp_apply_dft_to_dft_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        for j in 0..res.size() {
            ring::mul_assign::<F, R>(res.at_mut(res_col, j), a.at(a_col, 0));
        }
    }
}

// A prepared matrix keeps the MatZnx order: row, input column, limb, output
// column, each block the transform of one polynomial.
unsafe impl<F: Family, R: OracleRing> HalVmpImpl for Oracle<F, R> {
    fn vmp_prepare_tmp_bytes(_module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        0
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(
            (res.n(), res.rows(), res.cols_in(), res.cols_out(), res.size()),
            (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size())
        );
        let n = res.n();
        for (out, x) in res.raw_mut().chunks_exact_mut(n).zip(a.raw().chunks_exact(n)) {
            ring::forward(module, out, x);
        }
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        _a_size: usize,
        _b_rows: usize,
        _b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    // res[col, j] = sum_{row, input} a[input, row] b[row, input, col, j + limb_offset]
    fn vmp_apply_dft_to_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert!(res.n() == a.n() && res.n() == b.n());
        assert!(a.cols() == b.cols_in() && res.cols() == b.cols_out());
        let n = res.n();
        for j in 0..res.size() {
            for col in 0..res.cols() {
                let out = res.at_mut(col, j);
                out.fill(F::Dft::default());
                let Some(k) = j.checked_add(limb_offset).filter(|&k| k < b.size()) else {
                    continue;
                };
                for row in 0..a.size().min(b.rows()) {
                    for input in 0..a.cols() {
                        let at = ((row * b.cols_in() + input) * b.size() * b.cols_out() + k * b.cols_out() + col) * n;
                        ring::mul_acc::<F, R>(out, a.at(input, row), &b.raw()[at..at + n]);
                    }
                }
            }
        }
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        assert!(row_step > 0, "row_step must be positive");
        assert_eq!(
            res.hint(),
            a.hint(),
            "vmp_extract_selected_rows: res and a must carry the same PrepareHint"
        );
        assert!(res.n() == a.n() && res.cols_in() == a.cols_in() && res.cols_out() == a.cols_out());
        assert!(res.size() <= a.size(), "res.size(): {} > a.size(): {}", res.size(), a.size());
        if let Some(last) = res.rows().checked_sub(1) {
            let last = last.checked_mul(row_step).and_then(|x| x.checked_add(first_row));
            assert!(
                last.is_some_and(|last| last < a.rows()),
                "selected rows exceed a.rows(): {}",
                a.rows()
            );
        }
        let width = res.n() * res.cols_out() * res.size();
        let a_width = a.n() * a.cols_out() * a.size();
        for row in 0..res.rows() {
            for input in 0..res.cols_in() {
                let dst = (row * res.cols_in() + input) * width;
                let src = ((first_row + row * row_step) * a.cols_in() + input) * a_width;
                res.raw_mut()[dst..dst + width].copy_from_slice(&a.raw()[src..src + width]);
            }
        }
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        res.raw_mut().fill(F::Dft::default());
    }
}

/// Prepares every limb of every column of `a` into `raw`, limb-major.
fn cnv_prepare<F: Family, R: OracleRing>(
    module: &Module<Oracle<F, R>>,
    raw: &mut [F::Dft],
    cols: usize,
    size: usize,
    a: &VecZnxBackendRef<'_, Oracle<F, R>>,
) {
    assert_dense(a, "cnv_prepare");
    assert_eq!(cols, a.cols());
    let n = a.n();
    for j in 0..size {
        for col in 0..cols {
            let out = &mut raw[(j * cols + col) * n..(j * cols + col + 1) * n];
            if j < a.size() {
                ring::forward(module, out, a.at(col, j));
            } else {
                out.fill(F::Dft::default());
            }
        }
    }
}

unsafe impl<F: Family, R: OracleRing> HalConvolutionImpl for Oracle<F, R> {
    fn cnv_prepare_left_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(res.n(), a.n());
        let (cols, size) = (res.cols(), res.size());
        cnv_prepare(module, res.raw_mut(), cols, size, a);
    }

    fn cnv_prepare_right_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_eq!(res.n(), a.n());
        let (cols, size) = (res.cols(), res.size());
        cnv_prepare(module, res.raw_mut(), cols, size, a);
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_by_const_apply_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    // res[j] = sum_{i + l = j + cnv_offset} a[a_col, i] b[b_col, l, b_coeff]
    fn cnv_by_const_apply(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(res, "cnv_by_const_apply");
        assert_dense(a, "cnv_by_const_apply");
        assert_dense(b, "cnv_by_const_apply");
        assert_eq!(res.n(), a.n());
        let bound = (a.size() + b.size()).saturating_sub(1);
        for j in 0..res.size() {
            let out = res.at_mut(res_col, j);
            out.fill(F::Big::default());
            let Some(k) = j.checked_add(cnv_offset).filter(|&k| k < bound) else {
                continue;
            };
            for i in 0..a.size().min(k + 1) {
                if k - i < b.size() {
                    let scalar = F::Big::from(b.at(b_col, k - i)[b_coeff]);
                    for (r, &x) in out.iter_mut().zip(a.at(a_col, i)) {
                        *r = r.add(F::Big::from(x).mul(scalar));
                    }
                }
            }
        }
    }

    // res[j] = sum_{i + l = j + cnv_offset} a[a_col, i] b[b_col, l]. A sparse
    // right operand is read through its degree embedding.
    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n = res.n();
        assert_eq!(n, a.n());
        assert!(b.n().is_power_of_two() && n.is_multiple_of(b.n()));
        let lifted: Option<Vec<Vec<F::Dft>>> = (b.n() != n).then(|| {
            (0..b.size())
                .map(|l| {
                    let mut limb = vec![F::Dft::default(); n];
                    lift(module, &mut limb, b.at(b_col, l));
                    limb
                })
                .collect()
        });
        let b_limb = |l: usize| lifted.as_ref().map_or(b.at(b_col, l), |limbs| &limbs[l][..]);
        let bound = (a.size() + b.size()).saturating_sub(1);
        for j in 0..res.size() {
            let out = res.at_mut(res_col, j);
            out.fill(F::Dft::default());
            let Some(k) = j.checked_add(cnv_offset).filter(|&k| k < bound) else {
                continue;
            };
            for i in 0..a.size().min(k + 1) {
                if k - i < b.size() {
                    ring::mul_acc::<F, R>(out, a.at(a_col, i), b_limb(k - i));
                }
            }
        }
    }
}
