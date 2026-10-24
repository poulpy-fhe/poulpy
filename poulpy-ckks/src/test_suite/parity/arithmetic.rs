//! Exact arithmetic parity on canonical coefficients, independent of encryption.
use super::helpers::*;
use crate::Scale;
use crate::{
    CKKSInfos, CKKSLayout, GLWEPlaintextMeta, SlotsKind,
    api::{CKKSDotProductOps, CKKSModuleInfos},
    oep::*,
    test_suite::CKKSTestParams,
};
use poulpy_core::{GLWEAdd, GLWEMaskFill, layouts::*, oep::GLWENormalizeImpl};
use poulpy_hal::layouts::{Backend, Module, PrepareHint};

pub trait ArithmeticParityBackend:
    Backend<ZnxWord = i64> + CKKSAddImpl + CKKSSubImpl + CKKSCopyImpl + CKKSNegImpl + CKKSPow2Impl + GLWENormalizeImpl
{
}
impl<B> ArithmeticParityBackend for B where
    B: Backend<ZnxWord = i64> + CKKSAddImpl + CKKSSubImpl + CKKSCopyImpl + CKKSNegImpl + CKKSPow2Impl + GLWENormalizeImpl
{
}

/// A CKKS layout at the parameter set's degree and radix.
pub fn layout(params: CKKSTestParams, rank: usize, k: usize, delta: usize, sparse: usize, slots: SlotsKind) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: params.n.into(),
            base2k: params.base2k.into(),
            k: k.into(),
            rank: rank.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(delta),
            slots,
            log_sparsity: sparse,
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
                // `ckks_double_into` is the lazy `a + a` at the source's width and
                // `ckks_mul_pow2_into` by one bit below it, equal once normalized.
                for (width, lazy) in [(3 * b + 5, true), (2 * b + 3, false)] {
                    let dst = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
                    let mut doubled = fixture_ciphertext(module, &dst, 31);
                    let mut shifted = fixture_ciphertext(module, &dst, 31);
                    let bytes = B::ckks_mul_pow2_tmp_bytes(module, doubled.max_size()).max(B::glwe_normalize_tmp_bytes(module));
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_double_into(module, &mut doubled, &a, scratch)?;
                        let raw = snapshot::<B, _>(&doubled);
                        assert_eq!(raw.canonical, !lazy, "double_into k={width}");
                        results.push(("ckks_double_into", raw));
                        B::glwe_normalize_assign(module, &mut doubled, scratch);
                        B::ckks_mul_pow2_into(module, &mut shifted, &a, 1, scratch)
                    })
                    .unwrap();
                    assert_eq!(
                        snapshot::<B, _>(&doubled).digits,
                        snapshot::<B, _>(&shifted).digits,
                        "double_into k={width} differs from mul_pow2 by one bit"
                    );
                }
                for (width, below_ring) in [(2 * b + 3, false), (4 * b + 1, false), (2 * b + 3, true)] {
                    let out_layout = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
                    let mut pt_layout = layout(params, 0, b + 3, b - 1, sparse, slots);
                    if below_ring {
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
                    run!(ckks_add_into, ckks_add_tmp_bytes, out_layout, [&a, &rhs]);
                    run!(ckks_add_assign, ckks_add_tmp_bytes, la, [&rhs]);
                    run!(ckks_add_one_assign, ckks_add_one_tmp_bytes, la, []);
                    run!(ckks_add_pt_vec_into, ckks_add_pt_vec_tmp_bytes, out_layout, [&a, &pt]);
                    run!(ckks_add_pt_vec_assign, ckks_add_pt_vec_tmp_bytes, la, [&pt]);
                    run!(
                        ckks_add_pt_const_into,
                        ckks_add_pt_const_tmp_bytes,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    run!(ckks_add_pt_const_assign, ckks_add_pt_const_tmp_bytes, la, [7, &pt, 3]);
                    run!(ckks_sub_into, ckks_sub_tmp_bytes, out_layout, [&a, &rhs]);
                    run!(ckks_sub_assign, ckks_sub_tmp_bytes, la, [&rhs]);
                    run!(ckks_sub_one_assign, ckks_sub_one_tmp_bytes, la, []);
                    run!(ckks_sub_pt_vec_into, ckks_sub_pt_vec_tmp_bytes, out_layout, [&a, &pt]);
                    run!(ckks_sub_pt_vec_assign, ckks_sub_pt_vec_tmp_bytes, la, [&pt]);
                    run!(
                        ckks_sub_pt_const_into,
                        ckks_sub_pt_const_tmp_bytes,
                        out_layout,
                        [&a, 7, &pt, 3]
                    );
                    run!(ckks_sub_pt_const_assign, ckks_sub_pt_const_tmp_bytes, la, [7, &pt, 3]);
                    {
                        let mut out = fixture_ciphertext(module, &out_layout, 31);
                        let bytes = B::ckks_copy_tmp_bytes(module, &out, &a);
                        with_scratch::<B, _>(bytes, |scratch| B::ckks_copy(module, &mut out, &a, scratch)).unwrap();
                        results.push(("ckks_copy", snapshot::<B, _>(&out)));
                    }
                    run!(ckks_neg_into, ckks_neg_tmp_bytes, out_layout, [&a]);
                    run!(ckks_double_into, ckks_mul_pow2_tmp_bytes, out_layout, [&a]);
                    let mut neg = fixture_ciphertext(module, &la, 31);
                    B::ckks_neg_assign(module, &mut neg).unwrap();
                    results.push(("neg_assign", snapshot::<B, _>(&neg)));
                    for bits in [0, 1, b + 1] {
                        run!(ckks_mul_pow2_into, ckks_mul_pow2_tmp_bytes, out_layout, [&a, bits]);
                        run!(ckks_mul_pow2_assign, ckks_mul_pow2_tmp_bytes, la, [bits]);
                        run!(ckks_div_pow2_into, ckks_div_pow2_tmp_bytes, out_layout, [&a, bits]);
                        let mut div = fixture_ciphertext(module, &la, 31);
                        B::ckks_div_pow2_assign(module, &mut div, bits).unwrap();
                        results.push(("div_pow2_assign", snapshot::<B, _>(&div)));
                    }
                    assert_eq!(before_a, snapshot::<B, _>(&a), "arithmetic changed source a");
                    assert_eq!(before_b, snapshot::<B, _>(&rhs), "arithmetic changed source b");
                    assert_eq!(before_pt, snapshot::<B, _>(&pt), "arithmetic changed plaintext");
                }
                // A metadata-only budget failure must not change the destination.
                let mut out = fixture_ciphertext(module, &la, 31);
                let before = snapshot::<B, _>(&out);
                let bits = out.log_budget() + 1;
                assert!(B::ckks_div_pow2_assign(module, &mut out, bits).is_err());
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
    rejects!(ckks_add_into, ckks_add_tmp_bytes, [&source, &source]);
    rejects!(ckks_sub_into, ckks_sub_tmp_bytes, [&source, &source]);
    {
        let mut out = fixture_ciphertext(module, &narrow, 72);
        let unchanged = snapshot::<B, _>(&out);
        let bytes = B::ckks_copy_tmp_bytes(module, &out, &source);
        assert!(with_scratch::<B, _>(bytes, |scratch| B::ckks_copy(module, &mut out, &source, scratch)).is_err());
        assert_eq!(unchanged, snapshot::<B, _>(&out), "failed copy mutated its destination");
    }
    rejects!(ckks_neg_into, ckks_neg_tmp_bytes, [&source]);
    rejects!(ckks_mul_pow2_into, ckks_mul_pow2_tmp_bytes, [&source, 0]);
    rejects!(ckks_div_pow2_into, ckks_div_pow2_tmp_bytes, [&source, 0]);
    results
}

/// All add/sub carry variants, copy, negation and powers of two, with unequal
/// widths, rank, sparsity and slot metadata.
pub fn test_arithmetic_parity<BR: ArithmeticParityBackend, BT: ArithmeticParityBackend, F>(
    params: CKKSTestParams,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = arithmetic(params, r);
    assert_untagged(
        "arithmetic",
        want.iter()
            .filter(|(label, _)| !label.contains("copy"))
            .map(|(_, output)| output),
    );
    assert_eq!(want, arithmetic(params, t));
}

fn imaginary<B: Backend<ZnxWord = i64> + CKKSImagImpl>(
    params: CKKSTestParams,
    module: &Module<B>,
) -> Vec<(&'static str, Snapshot)>
where
    Module<B>: GLWEMaskFill<B>,
{
    let b = params.base2k;
    let rank = params.rank;
    let mut results = Vec::new();
    for (sparse, slots) in [(0, SlotsKind::Complex), (2, SlotsKind::Real)] {
        let la = layout(params, rank, 3 * b + 5, b - 1, sparse, slots);
        let a = fixture_ciphertext(module, &la, 31);
        let before = snapshot::<B, _>(&a);
        for width in [2 * b + 3, 4 * b + 1] {
            let out_layout = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
            macro_rules! run {
                ($method:ident, $query:ident, $initial:expr, [$($arg:expr),*]) => {{
                    let mut out = fixture_ciphertext(module, &$initial, 31);
                    let bytes = B::$query(module, out.max_size());
                    with_scratch::<B, _>(bytes, |scratch| B::$method(module, &mut out, $($arg,)* scratch)).unwrap();
                    results.push((stringify!($method), snapshot::<B, _>(&out)));
                }};
            }
            run!(ckks_mul_i_into, ckks_mul_i_tmp_bytes, out_layout, [&a]);
            run!(ckks_mul_i_assign, ckks_mul_i_tmp_bytes, la, []);
            run!(ckks_div_i_into, ckks_div_i_tmp_bytes, out_layout, [&a]);
            run!(ckks_div_i_assign, ckks_div_i_tmp_bytes, la, []);
        }
        assert_eq!(
            before,
            snapshot::<B, _>(&a),
            "imaginary-unit multiplication changed its source"
        );
    }
    // Both reject a result too narrow for its scale before mutating it.
    let source = fixture_ciphertext(module, &layout(params, 1, 3 * b + 5, 2 * b + 1, 0, SlotsKind::Complex), 71);
    let narrow = layout(params, 1, 1, 0, 0, SlotsKind::Real);
    macro_rules! rejects {
        ($method:ident, $query:ident) => {{
            let mut out = fixture_ciphertext(module, &narrow, 72);
            let unchanged = snapshot::<B, _>(&out);
            let bytes = B::$query(module, out.max_size());
            assert!(
                with_scratch::<B, _>(bytes, |scratch| B::$method(module, &mut out, &source, scratch)).is_err(),
                stringify!($method)
            );
            assert_eq!(
                unchanged,
                snapshot::<B, _>(&out),
                "failed {} mutated its destination",
                stringify!($method)
            );
        }};
    }
    rejects!(ckks_mul_i_into, ckks_mul_i_tmp_bytes);
    rejects!(ckks_div_i_into, ckks_div_i_tmp_bytes);
    results
}

/// Multiplication and division by the imaginary unit, standard ring only.
pub fn test_imag_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSImagImpl,
    BT: Backend<ZnxWord = i64> + CKKSImagImpl,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = imaginary(params, r);
    assert_untagged(
        "imaginary",
        want.iter()
            .filter(|(label, _)| !label.contains("copy"))
            .map(|(_, output)| output),
    );
    assert_eq!(want, imaginary(params, t));
}

fn products<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSMulImpl,
    Module<B>: GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B> + GLWEAdd<B> + CKKSDotProductOps<B>,
{
    use super::keys::{key_layout, prepared_tensor_key};
    let b = params.base2k;
    let mut results = Vec::new();
    {
        let rank = params.rank;
        for (dsize, lazy) in [(1, false), (2, true)] {
            let la = layout(params, rank, 4 * b + 5, b - 1, 2, SlotsKind::Real);
            let lb = layout(params, rank, 4 * b + 1, b + 1, 0, SlotsKind::Complex);
            let a = fixture_operand(module, &la, 51, lazy);
            let rhs = fixture_operand(module, &lb, 52, lazy);
            let before_a = snapshot::<B, _>(&a);
            let before_b = snapshot::<B, _>(&rhs);
            let key_infos = key_layout(params.n, b, 4 * b + 5, dsize, rank * (rank + 1) / 2, rank);
            let key = prepared_tensor_key(module, &key_infos, 53);
            let prepare_bytes = B::ckks_prepare_tmp_bytes(module, &a).max(B::ckks_prepare_tmp_bytes(module, &rhs));
            let left = with_scratch::<B, _>(prepare_bytes, |scratch| {
                B::ckks_prepare_left(module, &a, PrepareHint::Reuse, scratch)
            })
            .unwrap();
            let right = with_scratch::<B, _>(prepare_bytes, |scratch| {
                B::ckks_prepare_right(module, &rhs, PrepareHint::OneShot, scratch)
            })
            .unwrap();
            for width in [2 * b + 3, 5 * b + 1] {
                let lr = layout(params, rank, width, b - 1, 0, SlotsKind::Complex);
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_mul_tmp_bytes(module, &out, &a, &rhs, &key);
                with_scratch::<B, _>(bytes, |scratch| B::ckks_mul_into(module, &mut out, &a, &rhs, &key, scratch)).unwrap();
                results.push(("mul_into", snapshot::<B, _>(&out)));
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_square_tmp_bytes(module, &out, &a, &key);
                with_scratch::<B, _>(bytes, |scratch| B::ckks_square_into(module, &mut out, &a, &key, scratch)).unwrap();
                results.push(("square_into", snapshot::<B, _>(&out)));
                // Sides of other widths and scales than each other, prepared from the operands, lazy or not.
                for pairs in [1, 2] {
                    let mut want = fixture_ciphertext(module, &lr, 99);
                    let bytes = module.ckks_dot_product_ct_tmp_bytes(pairs, &want, &a, &rhs, &key);
                    with_scratch::<B, _>(bytes, |scratch| {
                        module.ckks_dot_product_ct(&mut want, &vec![&a; pairs], &vec![&rhs; pairs], &key, scratch)
                    })
                    .unwrap();
                    let mut have = fixture_ciphertext(module, &lr, 99);
                    with_scratch::<B, _>(bytes, |scratch| {
                        module.ckks_dot_product_prepared(&mut have, &vec![&left; pairs], &vec![&right; pairs], &key, scratch)
                    })
                    .unwrap();
                    let want = snapshot::<B, _>(&want);
                    assert_eq!(snapshot::<B, _>(&have), want, "prepared dot product of {pairs} pairs");
                    results.push(("dot_product_prepared", want));
                }
            }
            let mut assigned = fixture_operand(module, &la, 51, lazy);
            let bytes = B::ckks_mul_tmp_bytes(module, &assigned, &assigned, &rhs, &key);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_mul_assign(module, &mut assigned, &rhs, &key, scratch)
            })
            .unwrap();
            let ordinary = snapshot::<B, _>(&assigned);
            let prepared = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_prepare_right(module, &rhs, PrepareHint::OneShot, scratch)
            })
            .unwrap();
            let mut assigned = fixture_operand(module, &la, 51, lazy);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_mul_prepared_assign(module, &mut assigned, &prepared, &key, scratch)
            })
            .unwrap();
            assert_eq!(
                ordinary,
                snapshot::<B, _>(&assigned),
                "prepared multiplication differs from ordinary multiplication"
            );
            results.push(("mul_assign", ordinary));
            results.push(("mul_prepared_assign", snapshot::<B, _>(&assigned)));
            let mut assigned = fixture_operand(module, &la, 51, lazy);
            let bytes = B::ckks_square_tmp_bytes(module, &assigned, &assigned, &key);
            with_scratch::<B, _>(bytes, |scratch| B::ckks_square_assign(module, &mut assigned, &key, scratch)).unwrap();
            results.push(("square_assign", snapshot::<B, _>(&assigned)));
            for below_ring in [false, true] {
                let mut lp = layout(params, 0, b + 3, b - 1, 2, SlotsKind::Real);
                if below_ring {
                    lp.glwe_layout.n = (params.n / 4).into();
                }
                let pt = fixture_plaintext(module, &lp, 54);
                let before_pt = snapshot::<B, _>(&pt);
                let lr = layout(params, rank, 2 * b + 3, b - 1, 0, SlotsKind::Complex);
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_mul_pt_vec_tmp_bytes(module, &out, &a, &pt);
                with_scratch::<B, _>(bytes, |scratch| B::ckks_mul_pt_vec_into(module, &mut out, &a, &pt, scratch)).unwrap();
                results.push(("mul_pt_vec_into", snapshot::<B, _>(&out)));
                let mut out = fixture_operand(module, &la, 51, lazy);
                let bytes = B::ckks_mul_pt_vec_tmp_bytes(module, &out, &out, &pt);
                with_scratch::<B, _>(bytes, |scratch| B::ckks_mul_pt_vec_assign(module, &mut out, &pt, scratch)).unwrap();
                results.push(("mul_pt_vec_assign", snapshot::<B, _>(&out)));
                for coeff in [0, 3] {
                    let mut out = fixture_ciphertext(module, &lr, 99);
                    let bytes = B::ckks_mul_pt_const_tmp_bytes(module, &out, &a, &pt);
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_mul_pt_const_into(module, &mut out, &a, &pt, coeff, scratch)
                    })
                    .unwrap();
                    results.push(("mul_pt_const_into", snapshot::<B, _>(&out)));
                    let mut out = fixture_operand(module, &la, 51, lazy);
                    let bytes = B::ckks_mul_pt_const_tmp_bytes(module, &out, &out, &pt);
                    with_scratch::<B, _>(bytes, |scratch| {
                        B::ckks_mul_pt_const_assign(module, &mut out, &pt, coeff, scratch)
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
                with_scratch::<B, _>(0, |scratch| B::ckks_mul_prepared_assign(
                    module, &mut out, &prepared, &key, scratch
                ))
                .is_err()
            );
            assert_eq!(before, snapshot::<B, _>(&out));
        }
    }
    results
}

/// Ciphertext, square, prepared, vector and scalar multiplication, and the prepared dot product, which equals
/// `ckks_dot_product_ct` of its operands' sources.
pub fn test_multiplication_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSMulImpl,
    BT: Backend<ZnxWord = i64> + CKKSMulImpl,
    Module<BR>: GLWETensorKeyPreparedFactory<BR> + GLWEMaskFill<BR> + GLWEAdd<BR> + CKKSDotProductOps<BR>,
    Module<BT>: GLWETensorKeyPreparedFactory<BT> + GLWEMaskFill<BT> + GLWEAdd<BT> + CKKSDotProductOps<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = products(params, r);
    assert_untagged(
        "products",
        want.iter()
            .filter(|(label, _)| !label.contains("copy"))
            .map(|(_, output)| output),
    );
    assert_eq!(want, products(params, t));
}

/// Prepared, ordinary and square products at caller-selected large-ring layouts.
pub fn test_prepared_multiplication_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSMulImpl,
    BT: Backend<ZnxWord = i64> + CKKSMulImpl,
    Module<BR>: GLWETensorKeyPreparedFactory<BR> + GLWEMaskFill<BR> + GLWEAdd<BR>,
    Module<BT>: GLWETensorKeyPreparedFactory<BT> + GLWEMaskFill<BT> + GLWEAdd<BT>,
{
    fn products<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<Snapshot>
    where
        B: Backend<ZnxWord = i64> + CKKSMulImpl,
        Module<B>: GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B> + GLWEAdd<B>,
    {
        use super::keys::{key_layout, prepared_tensor_key};
        let base = params.base2k;
        let infos = layout(params, 1, 17 * base - 3, base - 1, 0, SlotsKind::Complex);
        let key_infos = key_layout(params.n, base, infos.k().as_usize(), 4, 1, 1);
        let key = prepared_tensor_key(module, &key_infos, 83);
        let mut results = Vec::new();
        for lazy in [false, true] {
            let a = fixture_operand(module, &infos, 81, lazy);
            let before = snapshot::<B, _>(&a);
            let bytes = B::ckks_mul_tmp_bytes(module, &a, &a, &a, &key).max(B::ckks_square_tmp_bytes(module, &a, &a, &key));
            let prepared = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_prepare_right(module, &a, PrepareHint::Reuse, scratch)
            })
            .unwrap();
            let mut ordinary = fixture_operand(module, &infos, 81, lazy);
            with_scratch::<B, _>(bytes, |scratch| B::ckks_mul_assign(module, &mut ordinary, &a, &key, scratch)).unwrap();
            let expected = snapshot::<B, _>(&ordinary);
            for _ in 0..2 {
                let mut dst = fixture_operand(module, &infos, 81, lazy);
                with_scratch::<B, _>(bytes, |scratch| {
                    B::ckks_mul_prepared_assign(module, &mut dst, &prepared, &key, scratch)
                })
                .unwrap();
                assert_eq!(expected, snapshot::<B, _>(&dst), "prepared versus ordinary multiplication");
            }
            let mut square = fixture_operand(module, &infos, 81, lazy);
            with_scratch::<B, _>(bytes, |scratch| B::ckks_square_assign(module, &mut square, &key, scratch)).unwrap();
            assert_eq!(expected, snapshot::<B, _>(&square), "square versus ordinary multiplication");
            assert_eq!(before, snapshot::<B, _>(&a), "operand modified");
            results.push(expected);
        }
        results
    }
    let _scalar = std::marker::PhantomData::<F>;
    let want = products(params, r);
    assert_untagged("prepared products", &want);
    assert_eq!(want, products(params, t));
}

fn rotations<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSRotateImpl,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWEMaskFill<B> + GLWEAdd<B>,
{
    use super::keys::{key_layout, prepared_automorphism_key};
    let b = params.base2k;
    let rank = params.rank;
    // One slot needs the key of `ckks_galois_element(1) = 5`; a full turn is the identity copy.
    let shifts = [1, module.ckks_max_slots() as i64];
    let mut results = Vec::new();
    for (dsize, lazy) in [(1, false), (2, true)] {
        let la = layout(params, rank, 3 * b + 5, b - 1, 2, SlotsKind::Complex);
        let input = fixture_operand(module, &la, 61, lazy);
        let before = snapshot::<B, _>(&input);
        let key_infos = key_layout(params.n, b, 3 * b + 5, dsize, rank, rank);
        let key = prepared_automorphism_key(module, &key_infos, 5, 62);
        for width in [2 * b + 3, 4 * b + 1] {
            let lr = layout(params, rank, width, b - 1, 0, SlotsKind::Real);
            let mut out = fixture_ciphertext(module, &lr, 99);
            let bytes = B::ckks_rotate_with_key_tmp_bytes(module, &input, &key);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_rotate_with_key_into(
                    module,
                    &mut out,
                    &input,
                    &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                    scratch,
                )
            })
            .unwrap();
            results.push(("rotate_with_key_into", snapshot::<B, _>(&out)));
            for k in shifts {
                let mut out = fixture_ciphertext(module, &lr, 99);
                let bytes = B::ckks_rotate_tmp_bytes(module, &input, &key);
                with_scratch::<B, _>(bytes, |scratch| {
                    B::ckks_rotate_into(module, &mut out, &input, k, &key, scratch)
                })
                .unwrap();
                results.push(("rotate_into", snapshot::<B, _>(&out)));
            }
        }
        let mut out = fixture_operand(module, &la, 61, lazy);
        let bytes = B::ckks_rotate_with_key_tmp_bytes(module, &out, &key);
        with_scratch::<B, _>(bytes, |scratch| {
            B::ckks_rotate_with_key_assign(
                module,
                &mut out,
                &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                scratch,
            )
        })
        .unwrap();
        results.push(("rotate_with_key_assign", snapshot::<B, _>(&out)));
        for k in shifts {
            let mut out = fixture_operand(module, &la, 61, lazy);
            let bytes = B::ckks_rotate_tmp_bytes(module, &out, &key);
            with_scratch::<B, _>(bytes, |scratch| B::ckks_rotate_assign(module, &mut out, k, &key, scratch)).unwrap();
            results.push(("rotate_assign", snapshot::<B, _>(&out)));
        }
        assert_eq!(before, snapshot::<B, _>(&input));
    }
    results
}

/// Keyed rotation and rotation by a slot shift, including the identity copy.
pub fn test_rotate_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSRotateImpl,
    BT: Backend<ZnxWord = i64> + CKKSRotateImpl,
    Module<BR>: GLWEAutomorphismKeyPreparedFactory<BR> + GLWEMaskFill<BR> + GLWEAdd<BR>,
    Module<BT>: GLWEAutomorphismKeyPreparedFactory<BT> + GLWEMaskFill<BT> + GLWEAdd<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = rotations(params, r);
    assert_untagged(
        "rotations",
        want.iter()
            .filter(|(label, _)| !label.contains("copy"))
            .map(|(_, output)| output),
    );
    assert_eq!(want, rotations(params, t));
}

fn conjugations<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSConjugateImpl,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWEMaskFill<B> + GLWEAdd<B>,
{
    use super::keys::{key_layout, prepared_automorphism_key};
    let b = params.base2k;
    let rank = params.rank;
    let mut results = Vec::new();
    for (dsize, lazy) in [(1, false), (2, true)] {
        let la = layout(params, rank, 3 * b + 5, b - 1, 2, SlotsKind::Complex);
        let input = fixture_operand(module, &la, 61, lazy);
        let before = snapshot::<B, _>(&input);
        let key_infos = key_layout(params.n, b, 3 * b + 5, dsize, rank, rank);
        let key = prepared_automorphism_key(module, &key_infos, -1, 62);
        for width in [2 * b + 3, 4 * b + 1] {
            let lr = layout(params, rank, width, b - 1, 0, SlotsKind::Real);
            let mut out = fixture_ciphertext(module, &lr, 99);
            let bytes = B::ckks_conjugate_tmp_bytes(module, &input, &key);
            with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_conjugate_with_key_into(
                    module,
                    &mut out,
                    &input,
                    &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                    scratch,
                )
            })
            .unwrap();
            results.push(("conjugate_with_key_into", snapshot::<B, _>(&out)));
        }
        let mut out = fixture_operand(module, &la, 61, lazy);
        let bytes = B::ckks_conjugate_tmp_bytes(module, &out, &key);
        with_scratch::<B, _>(bytes, |scratch| {
            B::ckks_conjugate_with_key_assign(
                module,
                &mut out,
                &GLWEAutomorphismKeyPreparedToBackendRef::<B>::to_backend_ref(&key),
                scratch,
            )
        })
        .unwrap();
        results.push(("conjugate_with_key_assign", snapshot::<B, _>(&out)));
        assert_eq!(before, snapshot::<B, _>(&input));
    }
    results
}

/// Conjugation, standard ring only.
pub fn test_conjugate_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSConjugateImpl,
    BT: Backend<ZnxWord = i64> + CKKSConjugateImpl,
    Module<BR>: GLWEAutomorphismKeyPreparedFactory<BR> + GLWEMaskFill<BR> + GLWEAdd<BR>,
    Module<BT>: GLWEAutomorphismKeyPreparedFactory<BT> + GLWEMaskFill<BT> + GLWEAdd<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = conjugations(params, r);
    assert_untagged(
        "conjugations",
        want.iter()
            .filter(|(label, _)| !label.contains("copy"))
            .map(|(_, output)| output),
    );
    assert_eq!(want, conjugations(params, t));
}
