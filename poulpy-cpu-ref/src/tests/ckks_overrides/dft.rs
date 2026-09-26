use crate::FFT64Ref;
use poulpy_ckks::api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSLinearTransformationOps};
use poulpy_ckks::layouts::{DFTMatrix, DFTOutputFormat, DFTPlan, Decode, Encode, Repack, Standard};
use poulpy_ckks::{CoeffsMeta};
use poulpy_core::layouts::LinearTransformationPrepared;
use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{DataView, Module, ScratchOwned};

fn plan(kind: poulpy_ckks::layouts::DFTType, format: DFTOutputFormat) -> DFTPlan {
    DFTPlan::new(kind, vec![(1, 1), (2, 2)], format, CoeffsMeta::from_delta_budget(12, 2)).unwrap()
}

#[test]
fn downstream_dft_preparation_preserves_plan_and_factors() {
    let module = Module::<FFT64Ref>::new(64);
    let mut scratch = ScratchOwned::<FFT64Ref>::alloc(1 << 20);
    let plan = plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::Standard);
    let matrix = <Module<FFT64Ref> as CKKSDFTMatrixOps<FFT64Ref, f64>>::ckks_new_dft_matrix::<Encode, Standard>(&module, 16usize.into(), &plan, &mut scratch.borrow()).unwrap();
    let mut factors = Vec::new();
    for factor in matrix.factor_operands() {
        let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(
            &module, &factor.index(), factor.first_diagonal_plaintext().unwrap(),
        );
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, factor, &mut scratch.borrow());
        factors.push(prepared);
    }
    let custom = matrix.try_with_factor_operands(&module, factors).unwrap();
    let reference = module.ckks_prepare_dft_matrix(&matrix, &mut scratch.borrow());
    assert_eq!(custom.plan().schedule(), reference.plan().schedule());
    assert_eq!(custom.plan().kind(), reference.plan().kind());
    assert_eq!(custom.plan().format(), reference.plan().format());
    assert_eq!(custom.consumed_bits(), reference.consumed_bits());
    for (a, b) in custom.factor_operands().iter().zip(reference.factor_operands()) {
        assert_eq!(a.index(), b.index());
        for (a, b) in a.giant_steps.iter().zip(&b.giant_steps) {
            for (a, b) in a.diagonals.iter().zip(&b.diagonals) {
                assert_eq!(a.plaintext.cnv().data(), b.plaintext.cnv().data());
            }
        }
    }
}

#[test]
fn checked_dft_construction_rejects_invalid_markers_and_layouts() {
    use poulpy_ckks::layouts::DFTType;
    let module = Module::<FFT64Ref>::new(64);
    let mut scratch = ScratchOwned::<FFT64Ref>::alloc(1 << 20);
    let build = |scratch: &mut ScratchOwned<FFT64Ref>| {
        <Module<FFT64Ref> as CKKSDFTMatrixOps<FFT64Ref, f64>>::ckks_new_dft_matrix::<Encode, Standard>(&module, 
            16usize.into(), &plan(DFTType::Encode, DFTOutputFormat::Standard), &mut scratch.borrow(),
        ).unwrap()
    };
    let matrix = build(&mut scratch);
    let factors = || {
        matrix.factor_operands().iter().map(|factor| {
            let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(
                &module, &factor.index(), factor.first_diagonal_plaintext().unwrap(),
            );
            module.ckks_prepare_linear_transformation_rhs(&mut prepared, factor, &mut ScratchOwned::<FFT64Ref>::alloc(1 << 20).borrow());
            prepared
        }).collect::<Vec<_>>()
    };
    assert!(DFTMatrix::<FFT64Ref, Encode, Standard, _>::try_from_factor_operands(
        &module, matrix.plan().clone(), factors(),
    ).is_ok());
    assert!(DFTMatrix::<FFT64Ref, Decode, Standard, _>::try_from_factor_operands(
        &module, matrix.plan().clone(), factors(),
    ).is_err());
    assert!(DFTMatrix::<FFT64Ref, Encode, Repack, _>::try_from_factor_operands(
        &module, matrix.plan().clone(), factors(),
    ).is_err());
    let mut empty = factors();
    empty.clear();
    assert!(matrix.try_with_factor_operands(&module, empty).is_err());
    let mut bad = factors();
    bad[0].giant_steps[0].diagonals[0].plaintext.set_log_scale(0);
    assert!(matrix.try_with_factor_operands(&module, bad).is_err());
    let mut bad = factors();
    bad[0].giant_steps.clear();
    assert!(matrix.try_with_factor_operands(&module, bad).is_err());
    let mut bad = factors();
    bad[0].baby_steps.clear();
    assert!(matrix.try_with_factor_operands(&module, bad).is_err());
    let dense = DFTPlan::new(DFTType::Encode, vec![(5, 1)], DFTOutputFormat::RepackImagAsReal, CoeffsMeta::from_delta_budget(12, 2)).unwrap();
    assert!(DFTMatrix::<FFT64Ref, Encode, Repack, _>::try_from_factor_operands(&module, dense, factors()).is_err());
}
