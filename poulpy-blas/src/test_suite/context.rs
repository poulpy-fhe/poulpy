//! The encoded maps of a context and of a public operand.

use poulpy_ckks::{
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch},
    },
};
use poulpy_core::layouts::DiagonalArithmetic;
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, ScratchArena, Standard},
};

use crate::layouts::{BlasContextUnprepared, BlasPtOperandUnprepared};

/// The context lists the rotations of every map and rejects matrices wider than the slots; a public operand is a grid
/// of maps whose count divides the matrix slots. Assumes `params.n = 256`.
pub fn test_blas_context<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
{
    let mut scratch = alloc_scratch(&params, module);
    let ctx = BlasContextUnprepared::encode::<F>(module, 2, params.base2k.into(), params.prec().into(), &mut scratch.borrow())
        .unwrap()
        .prepare(module, &mut scratch.borrow());
    assert_eq!(ctx.log_d(), 2);
    let elements = ctx.galois_elements(module.cyclotomic_order());
    // phi^1 rotates by 1 and by 1 - d.
    assert!(elements.contains(&module.ckks_galois_element(1)));
    assert!(elements.contains(&module.ckks_galois_element(1 - 4)));

    // 128 slots hold no 16 x 16 matrix; 2^40 x 2^40 has more entries than a usize counts.
    for log_d in [4, 40] {
        let res = BlasContextUnprepared::encode::<F>(
            module,
            log_d,
            params.base2k.into(),
            params.prec().into(),
            &mut scratch.borrow(),
        );
        assert!(res.is_err());
    }

    let entry = |m: usize, r: usize, c: usize| F::from_f64(((m + r * 3 + c) % 5) as f64 - 2.0).unwrap();
    let encode = |count: usize, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode(
            module,
            2,
            3,
            2,
            count,
            entry,
            params.base2k.into(),
            params.prec().into(),
            scratch,
        )
    };
    // 128 slots hold 8 matrix slots: 3 and 0 public matrices do not divide them.
    for count in [0, 3] {
        assert!(encode(count, &mut scratch.borrow()).is_err());
    }
    let b = encode(2, &mut scratch.borrow())
        .unwrap()
        .prepare(module, &mut scratch.borrow());
    assert_eq!((b.log_d(), b.rows(), b.cols(), b.count()), (2, 3, 2, 2));
    let order = module.cyclotomic_order();
    assert_eq!(b.galois_elements(order), b.block(2, 1).galois_elements(order));
}
