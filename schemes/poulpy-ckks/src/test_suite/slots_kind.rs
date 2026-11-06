//! Composition of the [`SlotsKind`] claim across the arithmetic ops.
//!
//! The reals are a subring of the complexes, so `Real` survives only when every
//! operand is `Real`, and any op that leaves the reals (multiplication by `i`,
//! a linear transformation with complex diagonals) yields `Complex`.

use poulpy_core::layouts::LWEInfos;
use poulpy_core::layouts::{GLWEPlaintextInfos, SetGLWEPlaintextInfos};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, add_sub_const_pt, alloc_ct, alloc_scratch, assert_ckks_error,
    ckks_encrypt, gen_sk_with_raw, gen_tsk,
};
use super::parity::helpers::{snapshot, with_scratch};
use crate::{
    CKKSCompositionError, SlotsKind,
    api::{CKKSAddOps, CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops, CKKSSubOps},
    test_suite::{CKKSTestParams, reference_encoder::ReferenceEncoder},
};

pub fn test_slots_kind_composition<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut encrypt = |re: &[F], im: &[F]| {
        ckks_encrypt(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            re,
            im,
            &mut scratch.borrow(),
        )
    };
    let mut a = encrypt(&re1, &im1);
    let mut b = encrypt(&re2, &im2);

    // An unstated kind is complex.
    assert_eq!(a.slots(), SlotsKind::Complex);

    let mut res = alloc_ct(&params, module, params.k);
    for (a_kind, b_kind, want) in [
        (SlotsKind::Real, SlotsKind::Real, SlotsKind::Real),
        (SlotsKind::Real, SlotsKind::Complex, SlotsKind::Complex),
        (SlotsKind::Complex, SlotsKind::Real, SlotsKind::Complex),
        (SlotsKind::Complex, SlotsKind::Complex, SlotsKind::Complex),
    ] {
        a.set_slots(a_kind);
        b.set_slots(b_kind);

        with_scratch::<BE, _>(module.ckks_add_tmp_bytes(res.max_size()), |s| {
            module.ckks_add_into(&mut res, &a, &b, s)
        })
        .unwrap();
        assert_eq!(res.slots(), want, "add_into({a_kind:?}, {b_kind:?})");

        with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(res.max_size()), |s| {
            module.ckks_sub_into(&mut res, &a, &b, s)
        })
        .unwrap();
        assert_eq!(res.slots(), want, "sub_into({a_kind:?}, {b_kind:?})");

        with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&res, &a, &b, &tsk), |s| {
            module.ckks_mul_into(&mut res, &a, &b, &tsk, s)
        })
        .unwrap();
        assert_eq!(res.slots(), want, "mul_into({a_kind:?}, {b_kind:?})");

        let mut acc = a.clone();
        with_scratch::<BE, _>(module.ckks_add_tmp_bytes(acc.max_size()), |s| {
            module.ckks_add_assign(&mut acc, &b, s)
        })
        .unwrap();
        assert_eq!(acc.slots(), want, "add_assign({a_kind:?}, {b_kind:?})");

        let mut acc = a.clone();
        with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&acc, &acc, &b, &tsk), |s| {
            module.ckks_mul_assign(&mut acc, &b, &tsk, s)
        })
        .unwrap();
        assert_eq!(acc.slots(), want, "mul_assign({a_kind:?}, {b_kind:?})");
    }

    // Squaring and negation stay inside whichever subfield the operand is in.
    for kind in [SlotsKind::Real, SlotsKind::Complex] {
        a.set_slots(kind);
        with_scratch::<BE, _>(module.ckks_square_tmp_bytes(&res, &a, &tsk), |s| {
            module.ckks_square_into(&mut res, &a, &tsk, s)
        })
        .unwrap();
        assert_eq!(res.slots(), kind, "square_into({kind:?})");

        with_scratch::<BE, _>(module.ckks_neg_tmp_bytes(res.max_size()), |s| {
            module.ckks_neg_into(&mut res, &a, s)
        })
        .unwrap();
        assert_eq!(res.slots(), kind, "neg_into({kind:?})");
    }

    // Multiplying by `i` maps the reals onto the imaginary axis.
    a.set_slots(SlotsKind::Real);
    with_scratch::<BE, _>(module.ckks_mul_i_tmp_bytes(res.max_size()), |s| {
        module.ckks_mul_i_into(&mut res, &a, s)
    })
    .unwrap();
    assert_eq!(res.slots(), SlotsKind::Complex);

    let mut acc = a.clone();
    with_scratch::<BE, _>(module.ckks_mul_i_tmp_bytes(acc.max_size()), |s| {
        module.ckks_mul_i_assign(&mut acc, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex);

    let mut acc = a.clone();
    with_scratch::<BE, _>(module.ckks_div_i_tmp_bytes(acc.max_size()), |s| {
        module.ckks_div_i_assign(&mut acc, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex);

    // A constant is one quantized coefficient, always a real scalar, so the
    // plaintext's own (default `Complex`) kind must not leak into the result.
    // What decides an added constant is where it lands: coefficient 0 is the
    // real constant term, `n/2` the imaginary one.
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    assert_eq!(cst.slots(), SlotsKind::Complex, "the constant plaintext is unlabelled");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    with_scratch::<BE, _>(module.ckks_add_pt_const_tmp_bytes(acc.max_size()), |s| {
        module.ckks_add_pt_const_assign(&mut acc, 0, &cst, 0, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Real, "a real constant term keeps real slots");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    with_scratch::<BE, _>(module.ckks_add_pt_const_tmp_bytes(acc.max_size()), |s| {
        module.ckks_add_pt_const_assign(&mut acc, m, &cst, 0, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex, "an imaginary constant term leaves the reals");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    with_scratch::<BE, _>(module.ckks_sub_pt_const_tmp_bytes(acc.max_size()), |s| {
        module.ckks_sub_pt_const_assign(&mut acc, m, &cst, 0, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex, "subtracting one is no different");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    with_scratch::<BE, _>(module.ckks_mul_pt_const_tmp_bytes(&acc, &acc, &cst), |s| {
        module.ckks_mul_pt_const_assign(&mut acc, &cst, 0, s)
    })
    .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Real, "a real scalar multiplier keeps real slots");

    // Halving an integer need not give an integer, so division refuses
    // `Integer` slots rather than carry a false claim.
    a.set_slots(SlotsKind::Integer);
    let before = snapshot::<BE, _>(&res);
    let err = with_scratch::<BE, _>(0, |s| module.ckks_div_pow2_into(1, &mut res, &a, s)).unwrap_err();
    assert_ckks_error(
        "div_pow2_into of Integer slots",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_div_pow2_into",
            reason: "dividing Integer slots by 2^1 need not give integers; relabel them Real first".into(),
        },
    );
    assert_eq!(snapshot::<BE, _>(&res), before, "a refused division wrote res");
    let mut acc = a.clone();
    let before = snapshot::<BE, _>(&acc);
    let err = module.ckks_div_pow2_assign(1, &mut acc).unwrap_err();
    assert_ckks_error(
        "div_pow2_assign of Integer slots",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_div_pow2_assign",
            reason: "dividing Integer slots by 2^1 need not give integers; relabel them Real first".into(),
        },
    );
    assert_eq!(snapshot::<BE, _>(&acc), before, "a refused division is a no-op");
    module.ckks_div_pow2_assign(0, &mut acc).unwrap();
    assert_eq!(acc.slots(), SlotsKind::Integer, "dividing by one keeps integers");
}
