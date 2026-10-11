use crate::FFT64Portable;
use poulpy_ckks::api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSLinearTransformationOps};
use poulpy_ckks::layouts::{DFTMatrix, DFTOutputFormat, DFTPlan, Decode, Encode, Repack, Standard};
use poulpy_ckks::{CKKSInfos, CoeffsMeta};
use poulpy_core::layouts::LinearTransformationPrepared;
use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{DataView, Module, ScratchOwned};

fn plan(kind: poulpy_ckks::layouts::DFTType, format: DFTOutputFormat) -> DFTPlan {
    DFTPlan::new(kind, vec![(1, 1), (2, 2)], format, CoeffsMeta::from_delta_budget(12, 2)).unwrap()
}

#[test]
fn downstream_dft_preparation_preserves_plan_and_factors() {
    let module = Module::<FFT64Portable>::new(64);
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(1 << 20);
    let plan = plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::Standard);
    let matrix = <Module<FFT64Portable> as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_new_dft_matrix::<Encode, Standard>(
        &module,
        16usize.into(),
        &plan,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut factors = Vec::new();
    for factor in matrix.factor_operands() {
        let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(
            &module,
            &factor.index(),
            factor.first_diagonal_plaintext().unwrap(),
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
    let module = Module::<FFT64Portable>::new(64);
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(1 << 20);
    let build = |scratch: &mut ScratchOwned<FFT64Portable>| {
        <Module<FFT64Portable> as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_new_dft_matrix::<Encode, Standard>(
            &module,
            16usize.into(),
            &plan(DFTType::Encode, DFTOutputFormat::Standard),
            &mut scratch.borrow(),
        )
        .unwrap()
    };
    let matrix = build(&mut scratch);
    let factors = || {
        matrix
            .factor_operands()
            .iter()
            .map(|factor| {
                let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(
                    &module,
                    &factor.index(),
                    factor.first_diagonal_plaintext().unwrap(),
                );
                module.ckks_prepare_linear_transformation_rhs(
                    &mut prepared,
                    factor,
                    &mut ScratchOwned::<FFT64Portable>::alloc(1 << 20).borrow(),
                );
                prepared
            })
            .collect::<Vec<_>>()
    };
    assert!(
        DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors(),)
            .is_ok()
    );
    assert!(
        DFTMatrix::<FFT64Portable, Decode, Standard, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors(),)
            .is_err()
    );
    assert!(
        DFTMatrix::<FFT64Portable, Encode, Repack, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors(),)
            .is_err()
    );
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
    let dense = DFTPlan::new(
        DFTType::Encode,
        vec![(5, 1)],
        DFTOutputFormat::RepackImagAsReal,
        CoeffsMeta::from_delta_budget(12, 2),
    )
    .unwrap();
    assert!(DFTMatrix::<FFT64Portable, Encode, Repack, _>::try_from_factor_operands(&module, dense, factors()).is_err());
}

#[test]
fn checked_dft_construction_rejects_repeated_and_inconsistent_diagonals() {
    use poulpy_ckks::layouts::{CKKSModuleAlloc, DFTType};
    use poulpy_core::layouts::{LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<FFT64Portable>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let dft_plan = || DFTPlan::new(DFTType::Encode, vec![(1, 1)], DFTOutputFormat::Standard, meta).unwrap();
    // A diagonal of the plan's precision, stored across `storage_k` bits.
    let diagonal = |baby: i64, storage_k: usize, real: bool| {
        let mut pt = module.ckks_pt_vec_alloc(16usize.into(), storage_k.into());
        pt.set_meta(meta.meta);
        poulpy_core::layouts::SetK::set_k(&mut pt, meta.k);
        if real {
            pt.set_slots(poulpy_ckks::SlotsKind::Real);
        }
        LinearTransformationDiagonal { baby, plaintext: pt }
    };
    let build = |diagonals| {
        DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(
            &module,
            dft_plan(),
            vec![LinearTransformation {
                baby_steps: vec![0, 1],
                giant_steps: vec![LinearTransformationGiantStep { rot: 0, diagonals }],
            }],
        )
    };
    let k = meta.k.as_usize();
    assert!(build(vec![diagonal(0, k, false), diagonal(1, k, false)]).is_ok());
    let rejection = |diagonals| build(diagonals).err().unwrap().to_string();
    // A repeated (giant, baby) pair would overwrite a prepared diagonal.
    assert!(rejection(vec![diagonal(0, k, false), diagonal(0, k, false)]).contains("repeats the diagonal"));
    // Storage wider than the precision selects would be truncated by preparation.
    assert!(rejection(vec![diagonal(0, 32, false), diagonal(1, 32, false)]).contains("is encoded across 32 bits"));
    assert!(rejection(vec![diagonal(0, k, false), diagonal(1, k, true)]).contains("slot kind"));
}

#[test]
fn checked_dft_construction_preserves_exact_encoded_width() {
    use poulpy_ckks::api::{Diagonal, GiantStep};
    use poulpy_ckks::layouts::DFTType;
    use poulpy_core::layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision};

    // A downstream operand may declare a width narrower than its one-limb allocation.
    struct DiagonalWidth(usize);
    impl LWEInfos for DiagonalWidth {
        fn n(&self) -> Degree {
            64usize.into()
        }
        fn base2k(&self) -> Base2K {
            16usize.into()
        }
        fn k(&self) -> TorusPrecision {
            14usize.into()
        }
        fn max_size(&self) -> usize {
            1
        }
    }
    impl GLWEInfos for DiagonalWidth {
        fn rank(&self) -> Rank {
            0usize.into()
        }
    }
    impl IntPolyInfos for DiagonalWidth {
        fn encoded_k(&self) -> TorusPrecision {
            self.0.into()
        }
    }
    impl LtDiagonalMeta for DiagonalWidth {
        fn lt_log_scale(&self) -> usize {
            12
        }
        fn lt_slots(&self) -> poulpy_ckks::SlotsKind {
            poulpy_ckks::SlotsKind::Complex
        }
    }
    let module = Module::<FFT64Portable>::new(64);
    let plan = DFTPlan::new(
        DFTType::Encode,
        vec![(1, 1)],
        DFTOutputFormat::Standard,
        CoeffsMeta::from_delta_budget(12, 2),
    )
    .unwrap();
    let factors = |width| {
        vec![LinearTransformation {
            baby_steps: vec![0],
            giant_steps: vec![GiantStep {
                rot: 0,
                diagonals: vec![Diagonal {
                    baby: 0,
                    plaintext: DiagonalWidth(width),
                }],
            }],
        }]
    };
    let matrix =
        DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(&module, plan.clone(), factors(16)).unwrap();
    let error = DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(&module, plan, factors(14))
        .err()
        .expect("preparation would change the encoded width from 14 to 16 bits");
    assert!(error.to_string().contains("is encoded across 14 bits"));
    assert!(matrix.try_with_factor_operands(&module, factors(14)).is_err());
    assert!(matrix.try_with_factor_operands(&module, factors(16)).is_ok());
}

use super::OverrideBackend;
use poulpy_ckks::api::LtDiagonalMeta;
use poulpy_ckks::layouts::{CKKSModuleAlloc, DFTMatrixPrepared};
use poulpy_ckks::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos};
use poulpy_core::layouts::{
    GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LinearTransformation, SetGLWEPlaintextInfos,
};
use poulpy_core::reference::linear_transformation::DiagonalProd;
use poulpy_hal::layouts::ScratchArena;
use std::cell::{Cell, RefCell};
const DFT_EXTRA_SCRATCH: usize = 1 << 25;
const DFT_PREPARE_SCRATCH: usize = 512;
thread_local! {
    static DFT_QUERIES: Cell<usize> = const { Cell::new(0) };
    /// Transform kind, destination width and source width of each query, in bits.
    static DFT_QUERY_LAYOUTS: RefCell<Vec<(poulpy_ckks::layouts::DFTType, usize, usize)>> = const { RefCell::new(Vec::new()) };
    static DFT_CALLS: Cell<usize> = const { Cell::new(0) };
    static REPACK_CALLS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl poulpy_ckks::oep::DFTImpl for OverrideBackend {
    fn ckks_prepare_dft_matrix_tmp_bytes_impl<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        DFT_PREPARE_SCRATCH + poulpy_ckks::reference::dft::ckks_prepare_dft_matrix_tmp_bytes(module, dft)
    }

    fn ckks_dft_tmp_bytes_impl<Dir, Fmt, P, Dst, Src, K>(
        module: &Module<Self>,
        dst: &Dst,
        src: &Src,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        key: &K,
    ) -> usize
    where
        P: poulpy_core::layouts::GLWEInfos,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        DFT_QUERIES.set(DFT_QUERIES.get() + 1);
        DFT_QUERY_LAYOUTS
            .with_borrow_mut(|layouts| layouts.push((dft.plan().kind(), dst.max_k().as_usize(), src.max_k().as_usize())));
        poulpy_ckks::reference::dft::ckks_dft_tmp_bytes(module, dst, src, dft, key) + DFT_EXTRA_SCRATCH
    }

    fn ckks_prepare_dft_matrix_impl<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> DFTMatrixPrepared<Self, Dir, Fmt>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>,
    {
        let (marker, mut remaining) = scratch.borrow().take_region(DFT_PREPARE_SCRATCH);
        marker.fill(0xD7);
        let prepared = poulpy_ckks::reference::dft::ckks_prepare_dft_matrix(module, dft, &mut remaining);
        assert!(marker.iter().all(|&byte| byte == 0xD7));
        prepared
    }

    fn ckks_dft_evaluate_assign_impl<Dir, Fmt, P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        let _ = (module, ct, dft, keys);
        let (mut workspace, _) = scratch.borrow().take_region(DFT_EXTRA_SCRATCH);
        <Self as poulpy_hal::layouts::Backend>::copy_host_to_view(&mut workspace, &vec![0x3C; DFT_EXTRA_SCRATCH]);
        DFT_CALLS.set(DFT_CALLS.get() + 1);
        Err(anyhow::anyhow!("DFT override probe").into())
    }

    fn ckks_slots_to_coeffs_repack_impl<P, Dst, Src, H>(
        module: &Module<Self>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        REPACK_CALLS.set(REPACK_CALLS.get() + 1);
        poulpy_ckks::oep::defaults::ckks_slots_to_coeffs_repack(module, op_out, ct_in, dft, keys, scratch)
    }
}

struct NoAutomorphismKey;
impl<BE: poulpy_hal::layouts::Backend> GetAutomorphismKey<BE> for NoAutomorphismKey {
    fn lookup_automorphism_key(
        &self,
        _: i64,
        _: poulpy_core::layouts::TorusPrecision,
    ) -> poulpy_core::Result<poulpy_core::layouts::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        panic!("DFT dispatch probe must not request keys")
    }
}

#[test]
fn conditional_dft_fallback_reenters_selected_evaluation() {
    use poulpy_core::layouts::{LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let mut pt = module.ckks_pt_vec_alloc(16usize.into(), meta.k);
    pt.set_meta(meta.meta);
    let matrix = DFTMatrix::<OverrideBackend, Decode, Repack, _>::try_from_factor_operands(
        &module,
        DFTPlan::new(
            poulpy_ckks::layouts::DFTType::Decode,
            vec![(2, 1)],
            DFTOutputFormat::RepackImagAsReal,
            meta,
        )
        .unwrap(),
        vec![LinearTransformation {
            baby_steps: vec![0],
            giant_steps: vec![LinearTransformationGiantStep {
                rot: 0,
                diagonals: vec![LinearTransformationDiagonal { baby: 0, plaintext: pt }],
            }],
        }],
    )
    .unwrap();
    let src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    let mut dst = module.ckks_ciphertext_alloc_from_infos(&src);
    let key = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        dsize: 1usize.into(),
    };
    let bytes = module.ckks_dft_tmp_bytes(&dst, &src, &matrix, &key);
    assert!(bytes >= DFT_EXTRA_SCRATCH);
    let mut scratch = ScratchOwned::<OverrideBackend>::alloc(bytes);
    DFT_CALLS.set(0);
    REPACK_CALLS.set(0);
    let error = module
        .ckks_slots_to_coeffs_repack(&mut dst, &src, &matrix, &NoAutomorphismKey, &mut scratch.borrow())
        .unwrap_err();
    assert!(error.to_string().contains("DFT override probe"));
    assert_eq!(REPACK_CALLS.get(), 1);
    assert_eq!(DFT_CALLS.get(), 1);
}

impl<F: poulpy_ckks::api::CKKSEncodingScalar> crate::ckks_encoding::CKKSEncodingTransform<F> for OverrideBackend {
    type Fft = crate::ckks_encoding::EncodingFFTTable<F>;
}
crate::impl_ckks_encoding!(OverrideBackend);
poulpy_ckks::impl_ckks_encapsulated_mod_up_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_bootstrapping_reference!(OverrideBackend);
unsafe impl<F: poulpy_ckks::api::CKKSEncodingScalar + poulpy_ckks::reference::dft::DftScalar> poulpy_ckks::oep::DFTMatrixImpl<F>
    for OverrideBackend
{
    fn ckks_new_dft_matrix_impl<Dir: poulpy_ckks::layouts::DftDirection, Fmt: poulpy_ckks::layouts::DftFormat>(
        module: &Module<Self>,
        base2k: poulpy_core::layouts::Base2K,
        plan: &DFTPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<DFTMatrix<Self, Dir, Fmt>> {
        poulpy_ckks::reference::dft::ckks_new_dft_matrix::<Dir, Fmt, Self, F>(module, base2k, plan, scratch)
    }
}

#[test]
fn bootstrap_sizing_includes_selected_dft_workspace() {
    use poulpy_ckks::api::CKKSBootstrappingOps;
    use poulpy_ckks::layouts::DFTType;
    use poulpy_ckks::layouts::eval_mod::EvalModPlan;
    use poulpy_ckks::layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
    };
    use poulpy_ckks::polynomial::SplitStrategy;
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let eval_mod = EvalModPlan::complex_exponential(1, 1, 0, SplitStrategy::MinDepth, meta, 16);
    let plan = BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques::default(),
        plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::SplitRealAndImag),
        eval_mod,
        plan(poulpy_ckks::layouts::DFTType::Decode, DFTOutputFormat::SplitRealAndImag),
    )
    .unwrap();
    let unprepared = BootstrappingContext::<OverrideBackend, f64>::compile_unprepared(
        &module,
        16usize.into(),
        &plan,
        &mut ScratchOwned::<OverrideBackend>::alloc(1 << 20).borrow(),
    )
    .unwrap();
    let prepare_bytes = unprepared.prepare_tmp_bytes(&module);
    assert!(prepare_bytes >= DFT_PREPARE_SCRATCH);
    let context = unprepared.prepare(&module, &mut ScratchOwned::<OverrideBackend>::alloc(prepare_bytes).borrow());
    let key = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 16usize.into(),
        dsize: 1usize.into(),
    };
    let keys = BootstrappingKeysLayout {
        automorphism_key: poulpy_core::layouts::GLWEAutomorphismKeyLayout {
            n: key.n,
            base2k: key.base2k,
            k_aux: key.k_aux,
            rank: key.rank,
            dnum: key.dnum,
            dsize: key.dsize,
        },
        tensor_key: key,
        encapsulation: None,
    };
    let src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    let dst = module.ckks_ciphertext_alloc(16usize.into(), 256usize.into());
    DFT_QUERIES.set(0);
    DFT_QUERY_LAYOUTS.with_borrow_mut(Vec::clear);
    let bytes = module.ckks_bootstrap_tmp_bytes(&dst, &src, &context, &keys);
    assert!(bytes >= DFT_EXTRA_SCRATCH);
    assert!(DFT_QUERIES.get() >= 2);
    // S2C-first: SlotsToCoeffs runs in place on the input, CoeffsToSlots reads the raised ciphertext.
    let layouts = DFT_QUERY_LAYOUTS.with_borrow(Vec::clone);
    assert_eq!(layouts, [(DFTType::Encode, 256, 256), (DFTType::Decode, 64, 64)]);

    let plan = BootstrappingPlan::new(
        BootstrappingPipeline::C2SFirst,
        BootstrappingTechniques::default(),
        self::plan(DFTType::Encode, DFTOutputFormat::SplitRealAndImag),
        EvalModPlan::complex_exponential(1, 1, 0, SplitStrategy::MinDepth, meta, 16),
        self::plan(DFTType::Decode, DFTOutputFormat::SplitRealAndImag),
    )
    .unwrap();
    let context = BootstrappingContext::<OverrideBackend, f64>::compile(
        &module,
        16usize.into(),
        &plan,
        &mut ScratchOwned::<OverrideBackend>::alloc(1 << 20).borrow(),
    )
    .unwrap();
    DFT_QUERY_LAYOUTS.with_borrow_mut(Vec::clear);
    module.ckks_bootstrap_tmp_bytes(&dst, &src, &context, &keys);
    // C2S-first: both transforms run on working-width ciphertexts.
    let layouts = DFT_QUERY_LAYOUTS.with_borrow(Vec::clone);
    assert_eq!(layouts, [(DFTType::Encode, 256, 256), (DFTType::Decode, 256, 256)]);
}

#[test]
fn dft_scratch_covers_factor_copy_with_working_buffer() {
    use poulpy_core::layouts::{LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let mut pt = module.ckks_pt_vec_alloc(16usize.into(), meta.k);
    pt.set_meta(meta.meta);
    let matrix = DFTMatrix::<OverrideBackend, Encode, Standard, _>::try_from_factor_operands(
        &module,
        DFTPlan::new(
            poulpy_ckks::layouts::DFTType::Encode,
            vec![(1, 1)],
            DFTOutputFormat::Standard,
            meta,
        )
        .unwrap(),
        vec![LinearTransformation {
            baby_steps: vec![0],
            giant_steps: vec![LinearTransformationGiantStep {
                rot: 0,
                diagonals: vec![LinearTransformationDiagonal { baby: 0, plaintext: pt }],
            }],
        }],
    )
    .unwrap();
    let mut ct = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    ct.set_meta(CoeffsMeta::from_delta_budget(32, 32).meta);
    let key = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        dsize: 1usize.into(),
    };
    let bytes = poulpy_ckks::reference::dft::ckks_dft_tmp_bytes(&module, &ct, &ct, &matrix, &key);
    poulpy_ckks::reference::dft::ckks_dft_evaluate_assign(
        &module,
        &mut ct,
        &matrix,
        &NoAutomorphismKey,
        &mut ScratchOwned::<OverrideBackend>::alloc(bytes).borrow(),
    )
    .unwrap();
    assert_eq!(ct.log_budget(), 20);
}

#[test]
fn dft_preparation_uses_selected_scratch() {
    let module = Module::<OverrideBackend>::new(64);
    let plan = plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::Standard);
    let matrix = <Module<OverrideBackend> as CKKSDFTMatrixOps<OverrideBackend, f64>>::ckks_new_dft_matrix::<Encode, Standard>(
        &module,
        16usize.into(),
        &plan,
        &mut ScratchOwned::<OverrideBackend>::alloc(1 << 20).borrow(),
    )
    .unwrap();
    let bytes = module.ckks_prepare_dft_matrix_tmp_bytes(&matrix);
    let prepared = module.ckks_prepare_dft_matrix(&matrix, &mut ScratchOwned::<OverrideBackend>::alloc(bytes).borrow());
    for (factor, prepared) in matrix.factor_operands().iter().zip(prepared.factor_operands()) {
        assert_eq!(factor.index(), prepared.index());
    }
}
