use crate::FFT64Portable;
use poulpy_ckks::CKKSModuleInfos;
use poulpy_ckks::api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSLinearTransformationOps};
use poulpy_ckks::layouts::{DFTMatrix, DFTOutputFormat, DFTPlan, Decode, Encode, Repack, Standard};
use poulpy_ckks::test_suite::{
    helpers::{assert_ckks_error, assert_ckks_internal},
    parity::helpers::with_scratch,
};
use poulpy_ckks::{CKKSCompositionError, CKKSInfos, CoeffsMeta};
use poulpy_core::layouts::LinearTransformationPrepared;
use poulpy_hal::layouts::Module;

fn plan(kind: poulpy_ckks::layouts::DFTType, format: DFTOutputFormat) -> DFTPlan {
    DFTPlan::new(kind, vec![(1, 1), (2, 2)], format, CoeffsMeta::from_delta_budget(12, 2)).unwrap()
}

/// A matrix of the encoding plan on `module`, built on exactly its query.
fn new_matrix(module: &Module<FFT64Portable>, plan: &DFTPlan) -> DFTMatrix<FFT64Portable, Encode, Standard> {
    type Ops = Module<FFT64Portable>;
    let bytes =
        <Ops as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_new_dft_matrix_tmp_bytes::<Encode, Standard>(module, plan).unwrap();
    with_scratch::<FFT64Portable, _>(bytes, |s| {
        <Ops as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_new_dft_matrix::<Encode, Standard>(module, 16usize.into(), plan, s)
    })
    .unwrap()
}

/// The factors of `matrix`, each prepared on exactly its query.
fn prepared_factors(
    module: &Module<FFT64Portable>,
    matrix: &DFTMatrix<FFT64Portable, Encode, Standard>,
) -> Vec<LinearTransformationPrepared<FFT64Portable>> {
    matrix
        .factor_operands()
        .iter()
        .map(|factor| {
            let first = factor.first_diagonal_plaintext().unwrap();
            let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(module, &factor.index(), first);
            with_scratch::<FFT64Portable, _>(module.ckks_prepare_linear_transformation_rhs_tmp_bytes(first), |s| {
                module.ckks_prepare_linear_transformation_rhs(&mut prepared, factor, s)
            })
            .unwrap();
            prepared
        })
        .collect()
}

#[test]
fn downstream_dft_preparation_preserves_plan_and_factors() {
    let module = Module::<FFT64Portable>::new(64);
    let plan = plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::Standard);
    let matrix = new_matrix(&module, &plan);
    let custom = matrix
        .try_with_factor_operands(&module, prepared_factors(&module, &matrix))
        .unwrap();
    let reference = with_scratch::<FFT64Portable, _>(module.ckks_prepare_dft_matrix_tmp_bytes(&matrix), |s| {
        module.ckks_prepare_dft_matrix_alloc(&matrix, s)
    })
    .unwrap();
    assert_eq!(custom.plan().schedule(), reference.plan().schedule());
    assert_eq!(custom.plan().kind(), reference.plan().kind());
    assert_eq!(custom.plan().format(), reference.plan().format());
    assert_eq!(custom.consumed_bits(), reference.consumed_bits());
    for (a, b) in custom.factor_operands().iter().zip(reference.factor_operands()) {
        assert_eq!(a.index(), b.index());
        for (a, b) in a.giant_steps.iter().zip(&b.giant_steps) {
            for (a, b) in a.diagonals.iter().zip(&b.diagonals) {
                assert!(
                    a.plaintext.cnv() == b.plaintext.cnv(),
                    "a factor prepared downstream differs from the prepared matrix's"
                );
            }
        }
    }
}

#[test]
fn checked_dft_construction_rejects_invalid_markers_and_layouts() {
    use poulpy_ckks::layouts::DFTType;
    let module = Module::<FFT64Portable>::new(64);
    let matrix = new_matrix(&module, &plan(DFTType::Encode, DFTOutputFormat::Standard));
    let factors = || prepared_factors(&module, &matrix);
    assert!(
        DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors())
            .is_ok()
    );
    let refused = |label: &str, result: poulpy_ckks::CKKSResult<()>, op: &'static str, reason: &str| {
        assert_ckks_error(
            label,
            &result.expect_err("an invalid matrix was accepted"),
            CKKSCompositionError::InvalidOperand {
                op,
                reason: reason.into(),
            },
        );
    };
    const FROM: &str = "try_from_factor_operands";
    const WITH: &str = "try_with_factor_operands";
    refused(
        "direction",
        DFTMatrix::<FFT64Portable, Decode, Standard, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors())
            .map(drop),
        FROM,
        "DFT direction does not match its marker",
    );
    refused(
        "format",
        DFTMatrix::<FFT64Portable, Encode, Repack, _>::try_from_factor_operands(&module, matrix.plan().clone(), factors())
            .map(drop),
        FROM,
        "DFT format does not match its marker",
    );
    let mut empty = factors();
    empty.clear();
    refused(
        "no factor",
        matrix.try_with_factor_operands(&module, empty).map(drop),
        WITH,
        "DFT factor count does not match its plan",
    );
    let mut bad = factors();
    bad[0].giant_steps[0].diagonals[0]
        .plaintext
        .set_plaintext_meta(Default::default());
    refused(
        "scale",
        matrix.try_with_factor_operands(&module, bad).map(drop),
        WITH,
        "DFT diagonal layout does not match its plan",
    );
    let mut bad = factors();
    bad[0].giant_steps.clear();
    refused(
        "no diagonal",
        matrix.try_with_factor_operands(&module, bad).map(drop),
        WITH,
        "DFT factor has no diagonals",
    );
    let mut bad = factors();
    bad[0].baby_steps.clear();
    refused(
        "no baby step",
        matrix.try_with_factor_operands(&module, bad).map(drop),
        WITH,
        "DFT diagonal has no baby step",
    );
    let dense = DFTPlan::new(
        DFTType::Encode,
        vec![(5, 1)],
        DFTOutputFormat::RepackImagAsReal,
        CoeffsMeta::from_delta_budget(12, 2),
    )
    .unwrap();
    assert_ckks_error(
        "dense repack",
        &DFTMatrix::<FFT64Portable, Encode, Repack, _>::try_from_factor_operands(&module, dense, factors())
            .err()
            .expect("an invalid matrix was accepted"),
        CKKSCompositionError::InvalidPlan {
            plan: "DFTPlan",
            reason: "dense DFT requires Split instead of Repack".into(),
        },
    );
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
        let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), 16usize.into(), storage_k.into());
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
                rotations: Default::default(),
                giant_steps: vec![LinearTransformationGiantStep { rot: 0, diagonals }],
            }],
        )
    };
    let k = meta.k.as_usize();
    assert!(build(vec![diagonal(0, k, false), diagonal(1, k, false)]).is_ok());
    let refused = |label: &str, diagonals, message: &str| {
        assert_ckks_error(
            label,
            &build(diagonals).err().expect("an invalid matrix was accepted"),
            CKKSCompositionError::InvalidOperand {
                op: "try_from_factor_operands",
                reason: message.into(),
            },
        );
    };
    // A repeated (giant, baby) pair would overwrite a prepared diagonal.
    refused(
        "repeated diagonal",
        vec![diagonal(0, k, false), diagonal(0, k, false)],
        "DFT factor repeats the diagonal (giant rot 0, baby 0)",
    );
    // Storage wider than the precision selects would be truncated by preparation.
    refused(
        "wide storage",
        vec![diagonal(0, 32, false), diagonal(1, 32, false)],
        &format!(
            "DFT diagonal is encoded across 32 bits but its precision {k} selects {} limbs",
            k.div_ceil(16)
        ),
    );
    refused(
        "slot kind",
        vec![diagonal(0, k, false), diagonal(1, k, true)],
        "DFT factor diagonals do not share one encoded width and slot kind",
    );
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
        fn lt_log_sparsity(&self, log_n: usize) -> usize {
            log_n.saturating_sub(self.log_n())
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
            rotations: Default::default(),
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
    // Preparation would change the encoded width from 14 to 16 bits.
    let message = "DFT diagonal is encoded across 14 bits but its precision 14 selects 1 limbs";
    let error = DFTMatrix::<FFT64Portable, Encode, Standard, _>::try_from_factor_operands(&module, plan, factors(14))
        .err()
        .expect("a 14-bit encoding was accepted");
    for (op, error) in [
        ("try_from_factor_operands", error),
        (
            "try_with_factor_operands",
            matrix
                .try_with_factor_operands(&module, factors(14))
                .err()
                .expect("a 14-bit encoding was accepted"),
        ),
    ] {
        assert_ckks_error(
            op,
            &error,
            CKKSCompositionError::InvalidOperand {
                op,
                reason: message.into(),
            },
        );
    }
    assert!(matrix.try_with_factor_operands(&module, factors(16)).is_ok());
}

use super::OverrideBackend;
use super::carry::assert_query_takes_extra;
use poulpy_ckks::api::LtDiagonalMeta;
use poulpy_ckks::layouts::{CKKSModuleAlloc, DFTMatrixPrepared};
use poulpy_ckks::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos};
use poulpy_core::DiagonalProd;
use poulpy_core::layouts::{
    GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LinearTransformation,
    SetGLWEPlaintextInfos,
};
use poulpy_hal::layouts::ScratchArena;
use std::cell::{Cell, RefCell};
const DFT_EXTRA_DEFAULT: usize = 1 << 25;
const DFT_PREPARE_DEFAULT: usize = 512;
thread_local! {
    static DFT_EXTRA: Cell<usize> = const { Cell::new(DFT_EXTRA_DEFAULT) };
    static DFT_PREPARE_EXTRA: Cell<usize> = const { Cell::new(DFT_PREPARE_DEFAULT) };
    static DFT_QUERIES: Cell<usize> = const { Cell::new(0) };
    /// Transform kind, destination width and source width of each query, in bits.
    static DFT_QUERY_LAYOUTS: RefCell<Vec<(poulpy_ckks::layouts::DFTType, usize, usize)>> = const { RefCell::new(Vec::new()) };
    static DFT_CALLS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl poulpy_ckks::oep::CKKSDFTImpl for OverrideBackend {
    fn ckks_prepare_dft_matrix_tmp_bytes<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: poulpy_core::layouts::GLWEInfos,
    {
        DFT_PREPARE_EXTRA.get() + poulpy_ckks::reference::dft::ckks_prepare_dft_matrix_tmp_bytes(module, dft)
    }

    fn ckks_dft_tmp_bytes<Dir, Fmt, P, Dst, Src, K>(
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
        poulpy_ckks::reference::dft::ckks_dft_tmp_bytes(module, dst, src, dft, key) + DFT_EXTRA.get()
    }

    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        module: &Module<Self>,
        res: &mut DFTMatrixPrepared<Self, Dir, Fmt>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>,
    {
        super::carry::with_marked_bytes(DFT_PREPARE_EXTRA.get(), scratch, |rest| {
            poulpy_ckks::reference::dft::ckks_prepare_dft_matrix(module, res, dft, rest)
        })
    }

    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        let _ = (module, ct, dft, keys);
        let extra = DFT_EXTRA.get();
        let (mut workspace, _) = scratch.borrow().take_region(extra);
        <Self as poulpy_hal::layouts::Backend>::copy_host_to_view(&mut workspace, &vec![0x3C; extra]);
        DFT_CALLS.set(DFT_CALLS.get() + 1);
        Err(anyhow::anyhow!("DFT override probe").into())
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
impl<BE: poulpy_hal::layouts::Backend> GetTensorKey<BE> for NoAutomorphismKey {
    fn lookup_tensor_key(
        &self,
        _: poulpy_core::layouts::TorusPrecision,
    ) -> poulpy_core::Result<poulpy_core::layouts::prepared::GLWETensorKeyPreparedBackendRef<'_, BE>> {
        panic!("DFT dispatch probe must not request keys")
    }
}

#[test]
fn dft_format_wrapper_reenters_selected_evaluation() {
    use poulpy_core::layouts::{LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), 16usize.into(), meta.k);
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
            rotations: Default::default(),
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
        k_digit: 16usize.into(),
    };
    let bytes = module.ckks_dft_tmp_bytes(&dst, &src, &matrix, &key);
    assert_eq!(
        bytes,
        poulpy_ckks::reference::dft::ckks_dft_tmp_bytes(&module, &dst, &src, &matrix, &key) + DFT_EXTRA_DEFAULT
    );
    DFT_CALLS.set(0);
    let error = with_scratch::<OverrideBackend, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_repack(&mut dst, &src, &matrix, &NoAutomorphismKey, s)
    })
    .unwrap_err();
    assert_ckks_internal("ckks_slots_to_coeffs_repack", &error, "DFT override probe");
    assert_eq!(DFT_CALLS.get(), 1);
}

impl<F: poulpy_ckks::api::CKKSEncodingScalar> crate::ckks_encoding::CKKSEncodingTransform<F> for OverrideBackend {
    type Fft = crate::ckks_encoding::EncodingFFTTable<F>;
}
crate::impl_ckks_encoding!(OverrideBackend);
poulpy_ckks::impl_ckks_encapsulated_mod_up_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_mod_up_first_factor_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_bootstrapping_reference_full!(OverrideBackend);
unsafe impl<F: poulpy_ckks::api::CKKSEncodingScalar + poulpy_ckks::layouts::DftScalar> poulpy_ckks::oep::CKKSDFTMatrixImpl<F>
    for OverrideBackend
{
    fn ckks_new_dft_matrix_tmp_bytes<Dir: poulpy_ckks::layouts::DftDirection, Fmt: poulpy_ckks::layouts::DftFormat>(
        module: &Module<Self>,
        plan: &DFTPlan,
    ) -> Result<usize> {
        poulpy_ckks::reference::dft::ckks_new_dft_matrix_tmp_bytes::<Dir, Fmt, Self, F>(module, plan)
    }

    fn ckks_new_dft_matrix<Dir: poulpy_ckks::layouts::DftDirection, Fmt: poulpy_ckks::layouts::DftFormat>(
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
    type Context = BootstrappingContext<OverrideBackend, f64>;
    let compile = |plan: &BootstrappingPlan| {
        let unprepared = with_scratch::<OverrideBackend, _>(Context::compile_unprepared_tmp_bytes(&module, plan).unwrap(), |s| {
            Context::compile_unprepared(&module, 16usize.into(), plan, s)
        })
        .unwrap();
        // Every matrix preparation adds the probe's workspace, so their maximum moves one for one with it.
        assert_query_takes_extra(
            "prepare_tmp_bytes",
            |bytes| DFT_PREPARE_EXTRA.set(bytes),
            || unprepared.prepare_tmp_bytes(&module),
        );
        DFT_PREPARE_EXTRA.set(DFT_PREPARE_DEFAULT);
        with_scratch::<OverrideBackend, _>(unprepared.prepare_tmp_bytes(&module), |s| unprepared.prepare(&module, s)).unwrap()
    };
    let context = compile(&plan);
    let key = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 16usize.into(),
        k_digit: 16usize.into(),
    };
    let keys = BootstrappingKeysLayout {
        rotation_keys: Default::default(),
        first_factor_key: None,
        conversion_keys: None,
        automorphism_key: poulpy_core::layouts::GLWEAutomorphismKeyLayout {
            n: key.n,
            base2k: key.base2k,
            k_aux: key.k_aux,
            rank: key.rank,
            dnum: key.dnum,
            k_digit: key.k_digit,
        },
        tensor_key: key,
        encapsulation: None,
    };
    let src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    let dst = module.ckks_ciphertext_alloc(16usize.into(), 256usize.into());
    assert_query_takes_extra(
        "ckks_bootstrap_tmp_bytes",
        |bytes| DFT_EXTRA.set(bytes),
        || module.ckks_bootstrap_tmp_bytes(&dst, &src, &context, &keys),
    );
    DFT_EXTRA.set(DFT_EXTRA_DEFAULT);
    DFT_QUERIES.set(0);
    DFT_QUERY_LAYOUTS.with_borrow_mut(Vec::clear);
    module.ckks_bootstrap_tmp_bytes(&dst, &src, &context, &keys);
    assert_eq!(DFT_QUERIES.get(), 2);
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
    let context = compile(&plan);
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
    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), 16usize.into(), meta.k);
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
            rotations: Default::default(),
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
        k_digit: 16usize.into(),
    };
    let bytes = poulpy_ckks::reference::dft::ckks_dft_tmp_bytes(&module, &ct, &ct, &matrix, &key);
    with_scratch::<OverrideBackend, _>(bytes, |s| {
        poulpy_ckks::reference::dft::ckks_dft_evaluate_assign(&module, &mut ct, &matrix, &NoAutomorphismKey, s)
    })
    .unwrap();
    assert_eq!(ct.log_budget(), 20);
}

#[test]
fn dft_preparation_uses_selected_scratch() {
    let module = Module::<OverrideBackend>::new(64);
    let plan = plan(poulpy_ckks::layouts::DFTType::Encode, DFTOutputFormat::Standard);
    type Ops = Module<OverrideBackend>;
    let matrix = with_scratch::<OverrideBackend, _>(
        <Ops as CKKSDFTMatrixOps<OverrideBackend, f64>>::ckks_new_dft_matrix_tmp_bytes::<Encode, Standard>(&module, &plan)
            .unwrap(),
        |s| {
            <Ops as CKKSDFTMatrixOps<OverrideBackend, f64>>::ckks_new_dft_matrix::<Encode, Standard>(
                &module,
                16usize.into(),
                &plan,
                s,
            )
        },
    )
    .unwrap();
    let bytes = module.ckks_prepare_dft_matrix_tmp_bytes(&matrix);
    assert_eq!(
        bytes,
        DFT_PREPARE_DEFAULT + poulpy_ckks::reference::dft::ckks_prepare_dft_matrix_tmp_bytes(&module, &matrix)
    );
    let prepared = with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_prepare_dft_matrix_alloc(&matrix, s)).unwrap();
    for (factor, prepared) in matrix.factor_operands().iter().zip(prepared.factor_operands()) {
        assert_eq!(factor.index(), prepared.index());
    }
}
