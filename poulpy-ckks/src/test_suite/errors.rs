use crate::{
    CKKSCompositionError, CKKSError, CKKSInfos, CoeffsMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSEncodingOps, LinearTransformation},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals, EvalModPlan, compile_eval_mod},
    polynomial::{Basis, EncodeBSGS, Polynomial, SplitStrategy},
    reference::{ckks_encode_linear_transformation_from_diagonals, ckks_encode_linear_transformation_into},
};
use poulpy_core::layouts::{Diagonals, LWEInfos, LinearTransformationStrategy, TorusPrecision};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchArena, Standard},
};

use super::{
    helpers::{
        TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, ckks_encrypt, download_pt, gen_sk,
        test_vector_1,
    },
    parity::helpers::with_scratch,
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

pub fn test_add_pt_vec_alignment_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    ct.set_log_budget(0);
    let mut pt = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
    pt.set_meta(params.prec().meta());
    let err = module
        .ckks_add_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_vec_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_vec",
            ct_log_budget: 0,
            pt_log_delta: params.prec().log_delta(),
            // Alignment is checked against the plaintext's effective (meaningful)
            // precision `log_delta + log_budget`, not its physical `max_k`.
            pt_k: params.prec().log_delta() + params.prec().log_budget(),
        },
    );
}

/// `ckks_encode_linear_transformation_from_diagonals` rejects a zero torus width, `coeffs_meta.k = 0`, with
/// `InvalidOperand` before it allocates or writes anything: here on an empty scratch, on which encoding a diagonal
/// fails with another error.
pub fn test_encode_linear_transformation_zero_k_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let slots = params.n / 2;
    let mut re = Diagonals::new(slots);
    re.set(0, vec![F::from_f64(1.0).unwrap(); slots]);
    let identity = ComplexDiagonals::new(re, Diagonals::new(slots));
    let err = with_scratch::<BE, _>(0, |scratch| {
        ckks_encode_linear_transformation_from_diagonals(
            module,
            params.base2k.into(),
            CoeffsMeta::from_delta_budget(0, 0),
            &identity,
            LinearTransformationStrategy::Direct,
            false,
            scratch,
        )
        .map(drop)
    })
    .unwrap_err();
    assert_ckks_error(
        "encode_linear_transformation_zero_k",
        &err.into(),
        CKKSCompositionError::InvalidOperand {
            op: "ckks_encode_linear_transformation_from_diagonals",
            reason: "the operands' torus width, k, is 0".into(),
        },
    );
}

/// `ckks_encode_linear_transformation_into` rejects a malformed strategy with an error and a transform of another
/// schedule with `InvalidOperand`, both before any write, and refills a transform with the bits
/// `ckks_encode_linear_transformation_from_diagonals` encodes. A `CoeffsMeta` whose scale exceeds its width surfaces
/// from the allocating encoder as `LimbReallocationShrinksBelowMetadata`, through its context.
pub fn test_encode_linear_transformation_into<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let slots = params.n / 2;
    let map = |indexes: &[i64], value: f64| {
        let mut re = Diagonals::new(slots);
        for &index in indexes {
            re.set(index, vec![F::from_f64(value).unwrap(); slots]);
        }
        ComplexDiagonals::new(re, Diagonals::new(slots))
    };
    let bytes = |lt: &LinearTransformation<CKKSPlaintextOwned<BE>>| -> Vec<Vec<u8>> {
        lt.giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .map(|d| download_pt::<BE>(&d.plaintext).inner.data().data().to_vec())
            .collect()
    };
    let coeffs_meta: CoeffsMeta = params.prec().into();
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    let (ones, twos) = (map(&[0, 1, 2, 3], 1.0), map(&[0, 1, 2, 3], 2.0));
    let mut scratch = alloc_scratch(&params, module);
    let encode = |diagonals: &ComplexDiagonals<F>, coeffs_meta: CoeffsMeta, scratch: &mut ScratchArena<'_, BE>| {
        ckks_encode_linear_transformation_from_diagonals(
            module,
            params.base2k.into(),
            coeffs_meta,
            diagonals,
            strategy,
            false,
            scratch,
        )
    };

    let mut lt = encode(&ones, coeffs_meta, &mut scratch.borrow()).unwrap();
    let before = bytes(&lt);
    let other = map(&[0, 1, 2], 2.0);
    let err = ckks_encode_linear_transformation_into(module, &mut lt, coeffs_meta.meta, &other, strategy, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "encode_linear_transformation_into",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_encode_linear_transformation_into",
            reason: "the transform's schedule is not the diagonals'".into(),
        },
    );
    let zero_giant_step = LinearTransformationStrategy::Bsgs { giant_step: 0 };
    let res = ckks_encode_linear_transformation_into(
        module,
        &mut lt,
        coeffs_meta.meta,
        &twos,
        zero_giant_step,
        &mut scratch.borrow(),
    );
    assert!(res.is_err(), "a zero giant step was accepted");
    assert_eq!(bytes(&lt), before, "a rejected encoding wrote the transform");

    ckks_encode_linear_transformation_into(module, &mut lt, coeffs_meta.meta, &twos, strategy, &mut scratch.borrow()).unwrap();
    let want = encode(&twos, coeffs_meta, &mut scratch.borrow()).unwrap();
    assert_eq!(bytes(&lt), bytes(&want), "encoding again does not refill the transform");

    let narrow = CoeffsMeta {
        k: TorusPrecision(1),
        ..coeffs_meta
    };
    let err: CKKSError = encode(&ones, narrow, &mut scratch.borrow()).map(drop).unwrap_err().into();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::LimbReallocationShrinksBelowMetadata { .. })
        ),
        "unexpected error: {err}"
    );
}

/// `EncodeBSGS::encode_bsgs_with`, `encode_bsgs_folded_with` and `compile_eval_mod` reject a zero torus width,
/// `coeff_meta.k = 0`, with `InvalidOperand` before they allocate anything: here on an odd polynomial the encoders
/// would otherwise encode, panicking on the first baby step, and on an empty scratch for `compile_eval_mod`.
pub fn test_encode_bsgs_zero_k_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // x + x³: odd, so the folded entry folds it.
    let poly = Polynomial::new(Basis::Monomial, vec![F::zero(), F::one(), F::zero(), F::one()]);
    let base2k = params.base2k.into();
    let zero_k = CoeffsMeta::from_delta_budget(0, 0);
    let split = SplitStrategy::MinDepth;
    let with = poly.encode_bsgs_with(host_module, base2k, zero_k, split).map(drop);
    let folded = poly.encode_bsgs_folded_with(host_module, base2k, zero_k, split).map(drop);
    let plan = EvalModPlan::complex_exponential(4, 2, 0, split, zero_k, params.prec().log_delta());
    let eval_mod = with_scratch::<BE, _>(0, |scratch| {
        compile_eval_mod::<BE, F>(base2k, plan, module, scratch).map(drop)
    });
    for (op, res) in [
        ("encode_bsgs_with", with),
        ("encode_bsgs_folded_with", folded),
        ("compile_eval_mod", eval_mod),
    ] {
        assert_ckks_error(
            op,
            &res.unwrap_err().into(),
            CKKSCompositionError::InvalidOperand {
                op,
                reason: "the operands' torus width, k, is 0".into(),
            },
        );
    }
}
