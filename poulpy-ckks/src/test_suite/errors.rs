use crate::{
    CKKSCompositionError, CKKSInfos, CoeffsMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSEncodingOps},
    layouts::{CKKSModuleAlloc, ComplexDiagonals},
    polynomial::{Basis, EncodeBSGS, Polynomial, SplitStrategy},
    reference::ckks_encode_linear_transformation_from_diagonals,
};
use poulpy_core::layouts::{Diagonals, LWEInfos, LinearTransformationStrategy};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use super::{
    helpers::{
        TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, ckks_encrypt, gen_sk, test_vector_1,
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
            reason: "the diagonals' torus width, coeffs_meta.k, is 0".into(),
        },
    );
}

/// `EncodeBSGS::encode_bsgs_with` and `encode_bsgs_folded_with` reject a zero torus width, `coeff_meta.k = 0`, with
/// `InvalidOperand` before they allocate anything: here on an odd polynomial both would otherwise encode, panicking on
/// the first baby step.
pub fn test_encode_bsgs_zero_k_error<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
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
    for (op, res) in [("encode_bsgs_with", with), ("encode_bsgs_folded_with", folded)] {
        assert_ckks_error(
            op,
            &res.unwrap_err().into(),
            CKKSCompositionError::InvalidOperand {
                op,
                reason: "the baby steps' torus width, coeff_meta.k, is 0".into(),
            },
        );
    }
}
