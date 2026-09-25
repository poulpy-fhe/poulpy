//! Exact arithmetic parity on canonical coefficients, independent of encryption.
use super::helpers::*;
use crate::{CKKSInfos, CKKSLayout, CKKSMeta, SlotsKind, layouts::*, oep::*, test_suite::CKKSTestParams};
use poulpy_core::{GLWEMaskFill, layouts::*, oep::GLWENormalizeImpl};
use poulpy_hal::layouts::{Backend, Module};

pub trait ArithmeticParityBackend:
    Backend<ZnxWord = i64> + CKKSAddImpl + CKKSSubImpl + CKKSCopyImpl + CKKSNegImpl + CKKSPow2Impl + CKKSImagImpl + GLWENormalizeImpl
{
}
impl<B> ArithmeticParityBackend for B where
    B: Backend<ZnxWord = i64>
        + CKKSAddImpl
        + CKKSSubImpl
        + CKKSCopyImpl
        + CKKSNegImpl
        + CKKSPow2Impl
        + CKKSImagImpl
        + GLWENormalizeImpl
{
}

pub(crate) fn layout(params: CKKSTestParams, rank: usize, k: usize, delta: usize, sparse: usize, slots: SlotsKind) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: params.n.into(),
            base2k: params.base2k.into(),
            k: k.into(),
            rank: rank.into(),
        },
        meta: CKKSMeta {
            log_delta: delta,
            log_sparsity: sparse,
            slots,
        },
    }
}

fn arithmetic<B: ArithmeticParityBackend>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    Module<B>: GLWEMaskFill<B>,
{
    let b = params.base2k;
    let mut results = Vec::new();
    {
        let rank = params.rank;
        for (sparse, slots) in [(0, SlotsKind::Complex), (2, SlotsKind::Real)] {
            for delta_difference in [0, 3] {
                let la = layout(params, rank, 3 * b + 5, b - 1, sparse, slots);
                let lb = layout(
                    params,
                    rank,
                    3 * b + 1,
                    b - 1 + delta_difference,
                    if delta_difference == 0 { sparse } else { 0 },
                    if delta_difference == 0 { slots } else { SlotsKind::Complex },
                );
                let a = fixture_ciphertext(module, &la, 31);
                let rhs = fixture_ciphertext(module, &lb, 32);
                let before_a = snapshot::<B, _>(&a);
                let before_b = snapshot::<B, _>(&rhs);
                for (width, compact) in [(2 * b + 3, false), (4 * b + 1, false), (2 * b + 3, true)] {
                    let out_layout = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
                    let mut pt_layout = layout(params, 0, b + 3, b - 1, sparse, slots);
                    if compact {
                        pt_layout.glwe_layout.n = (params.n / 4).into();
                    }
                    let pt = fixture_plaintext(module, &pt_layout, 33);
                    let before_pt = snapshot::<B, _>(&pt);
                    macro_rules! run {
                ($method:ident, $query:ident, $initial:expr, [$($arg:expr),*]) => {{
                    let mut out = fixture_ciphertext(module, &$initial, 31);
                    let bytes = B::$query(module, out.max_size());
                    with_scratch::<B,_>(bytes, |scratch| B::$method(module, &mut out, $($arg,)* scratch)).unwrap();
                    results.push((stringify!($method), snapshot::<B,_>(&out)));
                }};
            }
                    macro_rules! raw {
                ($method:ident, $query:ident, $initial:expr, [$($arg:expr),*]) => {{
                    let mut out = UnnormalizedCKKSCiphertext::new(fixture_ciphertext(module, &$initial, 31));
                    let bytes = B::$query(module, out.max_size());
                    with_scratch::<B,_>(bytes, |scratch| B::$method(module, &mut out, $($arg,)* scratch)).unwrap();
                    let bytes = B::glwe_normalize_tmp_bytes(module);
                    let out = with_scratch::<B,_>(bytes, |scratch| out.normalize(module, scratch));
                    results.push((stringify!($method), snapshot::<B,_>(&out)));
                }};
            }
                    run!(ckks_add_into_impl, ckks_add_tmp_bytes_impl, out_layout, [&a, &rhs]);
                    raw!(
                        ckks_add_into_unnormalized_impl,
                        ckks_add_tmp_bytes_impl,
                        out_layout,
                        [&a, &rhs]
                    );
                    run!(ckks_add_assign_impl, ckks_add_tmp_bytes_impl, la, [&rhs]);
                    raw!(ckks_add_assign_unnormalized_impl, ckks_add_tmp_bytes_impl, la, [&rhs]);
                    run!(ckks_add_one_assign_impl, ckks_add_one_tmp_bytes_impl, la, []);
                    run!(
                        ckks_add_pt_vec_into_impl,
                        ckks_add_pt_vec_tmp_bytes_impl,
                        out_layout,
                        [&a, &pt]
                    );
                    raw!(
                        ckks_add_pt_vec_into_unnormalized_impl,
                        ckks_add_pt_vec_tmp_bytes_impl,
                        out_layout,
                        [&a, &pt]
                    );
                    run!(ckks_add_pt_vec_assign_impl, ckks_add_pt_vec_tmp_bytes_impl, la, [&pt]);
                    raw!(
                        ckks_add_pt_vec_assign_unnormalized_impl,
                        ckks_add_pt_vec_tmp_bytes_impl,
                        la,
                        [&pt]
                    );
                    run!(
                        ckks_add_pt_const_into_impl,
                        ckks_add_pt_const_tmp_bytes_impl,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    raw!(
                        ckks_add_pt_const_into_unnormalized_impl,
                        ckks_add_pt_const_tmp_bytes_impl,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    run!(
                        ckks_add_pt_const_assign_impl,
                        ckks_add_pt_const_tmp_bytes_impl,
                        la,
                        [7, &pt, 3]
                    );
                    raw!(
                        ckks_add_pt_const_assign_unnormalized_impl,
                        ckks_add_pt_const_tmp_bytes_impl,
                        la,
                        [7, &pt, 3]
                    );
                    {
                        let mut out = fixture_ciphertext(module, &la, 31);
                        let bytes = B::ckks_add_tmp_bytes_impl(module, out.max_size());
                        {
                            let mut view = crate::layouts::ciphertext::UnnormalizedCKKSCiphertextRefMut::new(&mut out);
                            with_scratch::<B, _>(bytes, |scratch| {
                                B::ckks_add_assign_unnormalized_ref_impl(module, &mut view, &rhs, scratch)
                            })
                            .unwrap();
                            with_scratch::<B, _>(B::glwe_normalize_tmp_bytes(module), |scratch| view.normalize(module, scratch));
                        }
                        results.push(("ckks_add_assign_unnormalized_ref_impl", snapshot::<B, _>(&out)));
                    }
                    run!(ckks_sub_into_impl, ckks_sub_tmp_bytes_impl, out_layout, [&a, &rhs]);
                    raw!(
                        ckks_sub_into_unnormalized_impl,
                        ckks_sub_tmp_bytes_impl,
                        out_layout,
                        [&a, &rhs]
                    );
                    run!(ckks_sub_assign_impl, ckks_sub_tmp_bytes_impl, la, [&rhs]);
                    raw!(ckks_sub_assign_unnormalized_impl, ckks_sub_tmp_bytes_impl, la, [&rhs]);
                    run!(ckks_sub_one_assign_impl, ckks_sub_one_tmp_bytes_impl, la, []);
                    run!(
                        ckks_sub_pt_vec_into_impl,
                        ckks_sub_pt_vec_tmp_bytes_impl,
                        out_layout,
                        [&a, &pt]
                    );
                    raw!(
                        ckks_sub_pt_vec_into_unnormalized_impl,
                        ckks_sub_pt_vec_tmp_bytes_impl,
                        out_layout,
                        [&a, &pt]
                    );
                    run!(ckks_sub_pt_vec_assign_impl, ckks_sub_pt_vec_tmp_bytes_impl, la, [&pt]);
                    raw!(
                        ckks_sub_pt_vec_assign_unnormalized_impl,
                        ckks_sub_pt_vec_tmp_bytes_impl,
                        la,
                        [&pt]
                    );
                    run!(
                        ckks_sub_pt_const_into_impl,
                        ckks_sub_pt_const_tmp_bytes_impl,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    raw!(
                        ckks_sub_pt_const_into_unnormalized_impl,
                        ckks_sub_pt_const_tmp_bytes_impl,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    run!(
                        ckks_sub_pt_const_assign_impl,
                        ckks_sub_pt_const_tmp_bytes_impl,
                        la,
                        [7, &pt, 3]
                    );
                    raw!(
                        ckks_sub_pt_const_assign_unnormalized_impl,
                        ckks_sub_pt_const_tmp_bytes_impl,
                        la,
                        [7, &pt, 3]
                    );
                    {
                        let mut out = fixture_ciphertext(module, &la, 31);
                        let bytes = B::ckks_sub_tmp_bytes_impl(module, out.max_size());
                        {
                            let mut view = crate::layouts::ciphertext::UnnormalizedCKKSCiphertextRefMut::new(&mut out);
                            with_scratch::<B, _>(bytes, |scratch| {
                                B::ckks_sub_assign_unnormalized_ref_impl(module, &mut view, &rhs, scratch)
                            })
                            .unwrap();
                            with_scratch::<B, _>(B::glwe_normalize_tmp_bytes(module), |scratch| view.normalize(module, scratch));
                        }
                        results.push(("ckks_sub_assign_unnormalized_ref_impl", snapshot::<B, _>(&out)));
                    }
                    {
                        let mut out = fixture_ciphertext(module, &out_layout, 31);
                        let bytes = B::ckks_copy_tmp_bytes_impl(module, &out, &a);
                        with_scratch::<B, _>(bytes, |scratch| B::ckks_copy_impl(module, &mut out, &a, scratch)).unwrap();
                        results.push(("ckks_copy_impl", snapshot::<B, _>(&out)));
                    }
                    run!(ckks_neg_into_impl, ckks_neg_tmp_bytes_impl, out_layout, [&a]);
                    let mut neg = fixture_ciphertext(module, &la, 31);
                    B::ckks_neg_assign_impl(module, &mut neg).unwrap();
                    results.push(("neg_assign", snapshot::<B, _>(&neg)));
                    for bits in [0, 1, b + 1] {
                        run!(ckks_mul_pow2_into_impl, ckks_mul_pow2_tmp_bytes_impl, out_layout, [&a, bits]);
                        run!(ckks_mul_pow2_assign_impl, ckks_mul_pow2_tmp_bytes_impl, la, [bits]);
                        run!(ckks_div_pow2_into_impl, ckks_div_pow2_tmp_bytes_impl, out_layout, [&a, bits]);
                        let mut div = fixture_ciphertext(module, &la, 31);
                        B::ckks_div_pow2_assign_impl(module, &mut div, bits).unwrap();
                        results.push(("div_pow2_assign", snapshot::<B, _>(&div)));
                    }
                    run!(ckks_mul_i_into_impl, ckks_mul_i_tmp_bytes_impl, out_layout, [&a]);
                    run!(ckks_mul_i_assign_impl, ckks_mul_i_tmp_bytes_impl, la, []);
                    run!(ckks_div_i_into_impl, ckks_div_i_tmp_bytes_impl, out_layout, [&a]);
                    run!(ckks_div_i_assign_impl, ckks_div_i_tmp_bytes_impl, la, []);
                    assert_eq!(before_a, snapshot::<B, _>(&a), "arithmetic changed source a");
                    assert_eq!(before_b, snapshot::<B, _>(&rhs), "arithmetic changed source b");
                    assert_eq!(before_pt, snapshot::<B, _>(&pt), "arithmetic changed plaintext");
                }
                // A metadata-only budget failure must not change the destination.
                let mut out = fixture_ciphertext(module, &la, 31);
                let before = snapshot::<B, _>(&out);
                let bits = out.log_budget() + 1;
                assert!(B::ckks_div_pow2_assign_impl(module, &mut out, bits).is_err());
                assert_eq!(before, snapshot::<B, _>(&out));
            }
        }
    }
    // All these reference contracts reject a result too narrow for its scale
    // before mutating coefficient storage or semantic metadata.
    let source_layout = layout(params, 1, 3 * b + 5, 2 * b + 1, 0, SlotsKind::Complex);
    let source = fixture_ciphertext(module, &source_layout, 71);
    let narrow = layout(params, 1, 1, 0, 0, SlotsKind::Real);
    macro_rules! rejects {
        ($method:ident, $query:ident, [$($arg:expr),*]) => {{
            let mut out = fixture_ciphertext(module, &narrow, 72);
            let unchanged = snapshot::<B,_>(&out);
            let bytes = B::$query(module,out.max_size());
            assert!(with_scratch::<B,_>(bytes,|scratch|B::$method(module,&mut out,$($arg,)*scratch)).is_err(), stringify!($method));
            assert_eq!(unchanged,snapshot::<B,_>(&out), "failed {} mutated its destination", stringify!($method));
        }};
    }
    rejects!(ckks_add_into_impl, ckks_add_tmp_bytes_impl, [&source, &source]);
    rejects!(ckks_sub_into_impl, ckks_sub_tmp_bytes_impl, [&source, &source]);
    {
        let mut out = fixture_ciphertext(module, &narrow, 72);
        let unchanged = snapshot::<B, _>(&out);
        let bytes = B::ckks_copy_tmp_bytes_impl(module, &out, &source);
        assert!(with_scratch::<B, _>(bytes, |scratch| B::ckks_copy_impl(module, &mut out, &source, scratch)).is_err());
        assert_eq!(unchanged, snapshot::<B, _>(&out), "failed copy mutated its destination");
    }
    rejects!(ckks_neg_into_impl, ckks_neg_tmp_bytes_impl, [&source]);
    rejects!(ckks_mul_i_into_impl, ckks_mul_i_tmp_bytes_impl, [&source]);
    rejects!(ckks_div_i_into_impl, ckks_div_i_tmp_bytes_impl, [&source]);
    rejects!(ckks_mul_pow2_into_impl, ckks_mul_pow2_tmp_bytes_impl, [&source, 0]);
    rejects!(ckks_div_pow2_into_impl, ckks_div_pow2_tmp_bytes_impl, [&source, 0]);
    results
}

/// All add/sub carry variants, copy, negation, powers of two and imaginary-unit
/// multiplication, with unequal widths, rank, sparsity and slot metadata.
pub fn test_arithmetic_parity<BR: ArithmeticParityBackend, BT: ArithmeticParityBackend, F>(
    params: CKKSTestParams,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    assert_eq!(arithmetic(params, r), arithmetic(params, t));
}

fn products<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSMulImpl,
    Module<B>: GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    use super::keys::{key_layout, prepared_tensor_key};
    let b = params.base2k;
    let mut results = Vec::new();
    {
        let rank = params.rank;
        for dsize in [1, 2] {
            let la = layout(params, rank, 4 * b + 5, b - 1, 2, SlotsKind::Real);
            let lb = layout(params, rank, 4 * b + 1, b + 1, 0, SlotsKind::Complex);
            let a = fixture_ciphertext(module, &la, 51);
            let rhs = fixture_ciphertext(module, &lb, 52);
            let before_a = snapshot::<B, _>(&a);
            let before_b = snapshot::<B, _>(&rhs);
            let key_infos = key_layout(params.n, b, 4 * b + 5, dsize, rank * (rank + 1) / 2, rank);
            let key = prepared_tensor_key(module, &key_infos, 53);
            for width in [2 * b + 3, 5 * b + 1] {
                let lr = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_mul_tmp_bytes_impl(module, &out, &a, &rhs, &key);
                with_scratch::<B, _>(bytes, |scratch| {
                    B::ckks_mul_into_impl(module, &mut out, &a, &rhs, &key, scratch)
                })
                .unwrap();
                results.push(("mul_into", snapshot::<B, _>(&out)));
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_square_tmp_bytes_impl(module, &out, &a, &key);
                with_scratch::<B, _>(bytes, |scratch| B::ckks_square_into_impl(module, &mut out, &a, &key, scratch)).unwrap();
                results.push(("square_into", snapshot::<B, _>(&out)));
            }
            let mut assigned = fixture_ciphertext(module, &la, 51);
            let bytes = B::ckks_mul_tmp_bytes_impl(module, &assigned, &assigned, &rhs, &key);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_mul_assign_impl(module, &mut assigned, &rhs, &key, scratch)
            })
            .unwrap();
            let ordinary = snapshot::<B, _>(&assigned);
            let prepared = with_scratch::<B, _>(bytes, |scratch| B::ckks_prepare_right_impl(module, &rhs, scratch)).unwrap();
            let mut assigned = fixture_ciphertext(module, &la, 51);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_mul_prepared_assign_impl(module, &mut assigned, &prepared, &key, scratch)
            })
            .unwrap();
            assert_eq!(
                ordinary,
                snapshot::<B, _>(&assigned),
                "prepared multiplication differs from ordinary multiplication"
            );
            results.push(("mul_assign", ordinary));
            results.push(("mul_prepared_assign", snapshot::<B, _>(&assigned)));
            let mut assigned = fixture_ciphertext(module, &la, 51);
            let bytes = B::ckks_square_tmp_bytes_impl(module, &assigned, &assigned, &key);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_square_assign_impl(module, &mut assigned, &key, scratch)
            })
            .unwrap();
            results.push(("square_assign", snapshot::<B, _>(&assigned)));
            for compact in [false, true] {
                let mut lp = layout(params, 0, b + 3, b - 1, 2, SlotsKind::Real);
                if compact {
                    lp.glwe_layout.n = (params.n / 4).into();
                }
                let pt = fixture_plaintext(module, &lp, 54);
                let before_pt = snapshot::<B, _>(&pt);
                let lr = layout(params, rank, 2 * b + 3, b - 1, 0, SlotsKind::Complex);
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_mul_pt_vec_tmp_bytes_impl(module, &out, &a, pt.k());
                with_scratch::<B, _>(bytes, |scratch| {
                    B::ckks_mul_pt_vec_into_impl(module, &mut out, &a, &pt, scratch)
                })
                .unwrap();
                results.push(("mul_pt_vec_into", snapshot::<B, _>(&out)));
                let mut out = fixture_ciphertext(module, &la, 51);
                let bytes = B::ckks_mul_pt_vec_tmp_bytes_impl(module, &out, &out, pt.k());
                with_scratch::<B, _>(bytes, |scratch| {
                    B::ckks_mul_pt_vec_assign_impl(module, &mut out, &pt, scratch)
                })
                .unwrap();
                results.push(("mul_pt_vec_assign", snapshot::<B, _>(&out)));
                for coeff in [0, 3] {
                    let mut out = fixture_ciphertext(module, &lr, 99);
                    let bytes = B::ckks_mul_pt_const_tmp_bytes_impl(module, &out, &a, pt.k());
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_mul_pt_const_into_impl(module, &mut out, &a, &pt, coeff, scratch)
                    })
                    .unwrap();
                    results.push(("mul_pt_const_into", snapshot::<B, _>(&out)));
                    let mut out = fixture_ciphertext(module, &la, 51);
                    let bytes = B::ckks_mul_pt_const_tmp_bytes_impl(module, &out, &out, pt.k());
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_mul_pt_const_assign_impl(module, &mut out, &pt, coeff, scratch)
                    })
                    .unwrap();
                    results.push(("mul_pt_const_assign", snapshot::<B, _>(&out)));
                }
                assert_eq!(before_pt, snapshot::<B, _>(&pt));
            }
            assert_eq!(before_a, snapshot::<B, _>(&a));
            assert_eq!(before_b, snapshot::<B, _>(&rhs));
            let mut bad_layout = la;
            bad_layout.glwe_layout.base2k = (b - 1).into();
            let mut out = fixture_ciphertext(module, &bad_layout, 99);
            let before = snapshot::<B, _>(&out);
            // Prepared-object shape validation happens before any scratch use.
            assert!(
                with_scratch::<B, _>(0, |scratch| B::ckks_mul_prepared_assign_impl(
                    module, &mut out, &prepared, &key, scratch
                ))
                .is_err()
            );
            assert_eq!(before, snapshot::<B, _>(&out));
        }
    }
    results
}

/// Ciphertext, square, prepared-right, vector and scalar multiplication.
pub fn test_multiplication_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSMulImpl,
    BT: Backend<ZnxWord = i64> + CKKSMulImpl,
    Module<BR>: GLWETensorKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWETensorKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    assert_eq!(products(params, r), products(params, t));
}

fn automorphisms<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSRotateImpl + CKKSConjugateImpl,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    use super::keys::{key_layout, prepared_automorphism_key};
    let b = params.base2k;
    let mut results = Vec::new();
    {
        let rank = params.rank;
        for dsize in [1, 2] {
            for p in [5, -1] {
                let la = layout(params, rank, 3 * b + 5, b - 1, 2, SlotsKind::Complex);
                let input = fixture_ciphertext(module, &la, 61);
                let before = snapshot::<B, _>(&input);
                let key_infos = key_layout(params.n, b, 3 * b + 5, dsize, rank, rank);
                let key = prepared_automorphism_key(module, &key_infos, p, 62);
                for width in [2 * b + 3, 4 * b + 1] {
                    let lr = layout(params, rank, width, b - 1, 0, SlotsKind::Real);
                    let mut out = fixture_ciphertext(module, &lr, 99);
                    let bytes = B::ckks_rotate_tmp_bytes_impl(module, &input, &key);
                    if p == 5 {
                        with_scratch::<B, _>(bytes, |scratch| {
                            B::ckks_rotate_into_impl(
                                module,
                                &mut out,
                                &input,
                                &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                                scratch,
                            )
                        })
                        .unwrap();
                        results.push(("rotate_into", snapshot::<B, _>(&out)));
                    } else {
                        let bytes = B::ckks_conjugate_tmp_bytes_impl(module, &input, &key);
                        with_scratch::<B, _>(bytes, |scratch| {
                            B::ckks_conjugate_into_impl(
                                module,
                                &mut out,
                                &input,
                                &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                                scratch,
                            )
                        })
                        .unwrap();
                        results.push(("conjugate_into", snapshot::<B, _>(&out)));
                    }
                }
                let mut out = fixture_ciphertext(module, &la, 61);
                if p == 5 {
                    let bytes = B::ckks_rotate_tmp_bytes_impl(module, &out, &key);
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_rotate_assign_impl(
                            module,
                            &mut out,
                            &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                            scratch,
                        )
                    })
                    .unwrap();
                    results.push(("rotate_assign", snapshot::<B, _>(&out)));
                } else {
                    let bytes = B::ckks_conjugate_tmp_bytes_impl(module, &out, &key);
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_conjugate_assign_impl(
                            module,
                            &mut out,
                            &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                            scratch,
                        )
                    })
                    .unwrap();
                    results.push(("conjugate_assign", snapshot::<B, _>(&out)));
                }
                assert_eq!(before, snapshot::<B, _>(&input));
            }
        }
    }
    results
}

pub fn test_automorphism_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSRotateImpl + CKKSConjugateImpl,
    BT: Backend<ZnxWord = i64> + CKKSRotateImpl + CKKSConjugateImpl,
    Module<BR>: GLWEAutomorphismKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWEAutomorphismKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    assert_eq!(automorphisms(params, r), automorphisms(params, t));
}
