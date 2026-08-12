//! Composition of the [`SlotsKind`] claim across the arithmetic ops.
//!
//! The reals are a subring of the complexes, so `Real` survives only when every
//! operand is `Real`, and any op that leaves the reals (multiplication by `i`,
//! a linear transformation with complex diagonals) yields `Complex`.

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use super::helpers::{
    TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, add_sub_const_pt, alloc_ct, alloc_scratch,
    ckks_encrypt, gen_sk_with_raw, gen_tsk,
};
use crate::{
    CKKSInfos, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSSubOps},
    test_suite::{CKKSTestParams, reference_encoder::ReferenceEncoder},
};

pub fn test_slots_kind_composition<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
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

        module.ckks_add_into(&mut res, &a, &b, &mut scratch.borrow()).unwrap();
        assert_eq!(res.slots(), want, "add_into({a_kind:?}, {b_kind:?})");

        module.ckks_sub_into(&mut res, &a, &b, &mut scratch.borrow()).unwrap();
        assert_eq!(res.slots(), want, "sub_into({a_kind:?}, {b_kind:?})");

        module.ckks_mul_into(&mut res, &a, &b, &tsk, &mut scratch.borrow()).unwrap();
        assert_eq!(res.slots(), want, "mul_into({a_kind:?}, {b_kind:?})");

        let mut acc = a.clone();
        module.ckks_add_assign(&mut acc, &b, &mut scratch.borrow()).unwrap();
        assert_eq!(acc.slots(), want, "add_assign({a_kind:?}, {b_kind:?})");

        let mut acc = a.clone();
        module.ckks_mul_assign(&mut acc, &b, &tsk, &mut scratch.borrow()).unwrap();
        assert_eq!(acc.slots(), want, "mul_assign({a_kind:?}, {b_kind:?})");
    }

    // Squaring and negation stay inside whichever subfield the operand is in.
    for kind in [SlotsKind::Real, SlotsKind::Complex] {
        a.set_slots(kind);
        module.ckks_square_into(&mut res, &a, &tsk, &mut scratch.borrow()).unwrap();
        assert_eq!(res.slots(), kind, "square_into({kind:?})");

        module.ckks_neg_into(&mut res, &a, &mut scratch.borrow()).unwrap();
        assert_eq!(res.slots(), kind, "neg_into({kind:?})");
    }

    // Multiplying by `i` maps the reals onto the imaginary axis.
    a.set_slots(SlotsKind::Real);
    module.ckks_mul_i_into(&mut res, &a, &mut scratch.borrow()).unwrap();
    assert_eq!(res.slots(), SlotsKind::Complex);

    let mut acc = a.clone();
    module.ckks_mul_i_assign(&mut acc, &mut scratch.borrow()).unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex);

    let mut acc = a.clone();
    module.ckks_div_i_assign(&mut acc, &mut scratch.borrow()).unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex);

    // A constant is one quantized coefficient, always a real scalar, so the
    // plaintext's own (default `Complex`) kind must not leak into the result.
    // What decides an added constant is where it lands: coefficient 0 is the
    // real constant term, `n/2` the imaginary one.
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    assert_eq!(cst.slots(), SlotsKind::Complex, "the constant plaintext is unlabelled");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    module
        .ckks_add_pt_const_assign(&mut acc, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Real, "a real constant term keeps real slots");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    module
        .ckks_add_pt_const_assign(&mut acc, m, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex, "an imaginary constant term leaves the reals");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    module
        .ckks_sub_pt_const_assign(&mut acc, m, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Complex, "subtracting one is no different");

    let mut acc = a.clone();
    acc.set_slots(SlotsKind::Real);
    module
        .ckks_mul_pt_const_assign(&mut acc, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_eq!(acc.slots(), SlotsKind::Real, "a real scalar multiplier keeps real slots");
}
