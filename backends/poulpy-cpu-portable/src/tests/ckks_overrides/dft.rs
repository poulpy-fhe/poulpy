use crate::FFT64Portable;
use poulpy_ckks::CKKSModuleInfos;
use poulpy_ckks::api::{
    CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSDFTOps, CKKSEncodingScalar, CKKSLinearTransformationOps, LtDiagonalMeta,
};
use poulpy_ckks::layouts::{
    BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCiphertextAlloc, CKKSDFTFactors, CKKSEvalModInfos,
    CKKSModRaiseInfos, CKKSPlaintextAlloc, CKKSPlaintextOwned, DFTMatrix, DFTMatrixPrepared, DFTOutputFormat, DFTPlan, DFTType,
    Decode, DftDirection, DftFormat, DftScalar, Encode, EncodedLut, Repack, Standard,
};
use poulpy_ckks::oep::{CKKSBootstrappingImpl, CKKSDFTImpl, CKKSDFTMatrixImpl};
use poulpy_ckks::reference::dft::CKKSDFTReference;
use poulpy_ckks::test_suite::{
    CKKSTestParams,
    helpers::{assert_ckks_error, assert_ckks_internal},
    parity::helpers::with_scratch,
};
use poulpy_ckks::{CKKSCompositionError, CKKSInfos, CKKSResult, CoeffsMeta, SlotsKind};
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedBackendRef, GLWEInfos, GLWETensorKeyLayout,
    GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LinearTransformation, LinearTransformationPrepared,
    SetBSGSMeta, SetGLWEPlaintextInfos, SetK, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

fn plan(kind: DFTType, format: DFTOutputFormat) -> DFTPlan {
    DFTPlan::new(kind, vec![(1, 1), (2, 2)], format, CoeffsMeta::from_delta_budget(12, 2)).unwrap()
}

/// A matrix of the encoding plan on `module`, built on exactly its query.
fn new_matrix(module: &Module<FFT64Portable>, plan: &DFTPlan) -> DFTMatrix<FFT64Portable, Encode, Standard> {
    type Ops = Module<FFT64Portable>;
    let bytes =
        <Ops as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_dft_matrix_new_tmp_bytes::<Encode, Standard>(module, plan).unwrap();
    with_scratch::<FFT64Portable, _>(bytes, |s| {
        <Ops as CKKSDFTMatrixOps<FFT64Portable, f64>>::ckks_dft_matrix_new::<Encode, Standard>(module, 16usize.into(), plan, s)
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
            with_scratch::<FFT64Portable, _>(
                <poulpy_hal::layouts::Module<FFT64Portable> as CKKSLinearTransformationOps<
                    FFT64Portable,
                    poulpy_hal::layouts::Standard,
                >>::ckks_linear_transformation_rhs_prepare_tmp_bytes(module, first),
                |s| module.ckks_linear_transformation_rhs_prepare(&mut prepared, factor, s),
            )
            .unwrap();
            prepared
        })
        .collect()
}

#[test]
fn downstream_dft_preparation_preserves_plan_and_factors() {
    let module = Module::<FFT64Portable>::new(64);
    let plan = plan(DFTType::Encode, DFTOutputFormat::Standard);
    let matrix = new_matrix(&module, &plan);
    let custom = matrix
        .try_with_factor_operands(&module, prepared_factors(&module, &matrix))
        .unwrap();
    let reference = with_scratch::<FFT64Portable, _>(module.ckks_dft_matrix_prepare_tmp_bytes(&matrix), |s| {
        module.ckks_dft_matrix_prepare_alloc(&matrix, s)
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
    let refused = |label: &str, result: CKKSResult<()>, op: &'static str, reason: &str| {
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
    use poulpy_ckks::layouts::{CKKSPlaintextAlloc, DFTType};
    use poulpy_core::layouts::{LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<FFT64Portable>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let dft_plan = || DFTPlan::new(DFTType::Encode, vec![(1, 1)], DFTOutputFormat::Standard, meta).unwrap();
    // A diagonal of the plan's precision, stored across `storage_k` bits.
    let diagonal = |baby: i64, storage_k: usize, real: bool| {
        let mut pt = module.ckks_plaintext_alloc_from_slots::<poulpy_hal::layouts::Standard>(
            module.ckks_max_slots::<poulpy_hal::layouts::Standard>(),
            16usize.into(),
            storage_k.into(),
        );
        pt.set_meta(meta.meta);
        SetK::set_k(&mut pt, meta.k);
        if real {
            pt.set_slots(SlotsKind::Real);
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
        fn lt_slots(&self) -> SlotsKind {
            SlotsKind::Complex
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
use poulpy_ckks::reference::bootstrapping::{CKKSBootstrapEvalModReference, CKKSEvalLutReference, CKKSModRaiseReference};
use poulpy_ckks::test_suite::parity::{run_bootstrap, run_refused_bootstraps};
use poulpy_ckks::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos};
use poulpy_core::GLWEDiagonalProd;
use std::cell::{Cell, RefCell};
const DFT_EXTRA_DEFAULT: usize = 1 << 25;
const DFT_PREPARE_DEFAULT: usize = 512;
const RAISE_EXTRA_DEFAULT: usize = 448;
const BOOTSTRAP_EVAL_MOD_EXTRA_DEFAULT: usize = 576;
const EVAL_LUT_EXTRA_DEFAULT: usize = 704;
const PROBE_DFT_EXTRA: usize = 832;
thread_local! {
    static DFT_EXTRA: Cell<usize> = const { Cell::new(DFT_EXTRA_DEFAULT) };
    static DFT_PREPARE_EXTRA: Cell<usize> = const { Cell::new(DFT_PREPARE_DEFAULT) };
    static DFT_QUERIES: Cell<usize> = const { Cell::new(0) };
    /// Transform kind, destination width and source width of each query, in bits.
    static DFT_QUERY_LAYOUTS: RefCell<Vec<(poulpy_ckks::layouts::DFTType, usize, usize)>> = const { RefCell::new(Vec::new()) };
    static DFT_CALLS: Cell<usize> = const { Cell::new(0) };
    /// Whether the evaluation hook counts and runs the reference instead of failing.
    static DFT_FORWARD: Cell<bool> = const { Cell::new(false) };
    static RAISE_CALLS: Cell<usize> = const { Cell::new(0) };
    static BOOTSTRAP_EVAL_MOD_CALLS: Cell<usize> = const { Cell::new(0) };
    static EVAL_LUT_CALLS: Cell<usize> = const { Cell::new(0) };
    static RAISE_EXTRA: Cell<usize> = const { Cell::new(RAISE_EXTRA_DEFAULT) };
    static BOOTSTRAP_EVAL_MOD_EXTRA: Cell<usize> = const { Cell::new(BOOTSTRAP_EVAL_MOD_EXTRA_DEFAULT) };
    static EVAL_LUT_EXTRA: Cell<usize> = const { Cell::new(EVAL_LUT_EXTRA_DEFAULT) };
}
unsafe impl CKKSDFTImpl for OverrideBackend {
    fn ckks_dft_matrix_prepare_tmp_bytes<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: GLWEInfos,
    {
        DFT_PREPARE_EXTRA.get() + poulpy_ckks::reference::dft::ckks_dft_matrix_prepare_tmp_bytes_reference(module, dft)
    }

    fn ckks_dft_tmp_bytes<Dir, Fmt, D, Dst, Src, K>(module: &Module<Self>, dst: &Dst, src: &Src, dft: &D, key: &K) -> usize
    where
        D: CKKSDFTFactors<Self, Dir, Fmt>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: GGLWEInfos,
    {
        DFT_QUERIES.set(DFT_QUERIES.get() + 1);
        DFT_QUERY_LAYOUTS
            .with_borrow_mut(|layouts| layouts.push((dft.dft_plan().kind(), dst.max_k().as_usize(), src.max_k().as_usize())));
        poulpy_ckks::reference::dft::ckks_dft_tmp_bytes_reference(module, dst, src, dft, key) + DFT_EXTRA.get()
    }

    fn ckks_dft_matrix_prepare<Dir, Fmt, P>(
        module: &Module<Self>,
        res: &mut DFTMatrixPrepared<Self, Dir, Fmt>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<Self>,
    {
        super::carry::with_marked_bytes(DFT_PREPARE_EXTRA.get(), scratch, |rest| {
            poulpy_ckks::reference::dft::ckks_dft_matrix_prepare_reference(module, res, dft, rest)
        })
    }

    fn ckks_dft_matrix_eval_assign<Dir, Fmt, D, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &D,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        D: CKKSDFTFactors<Self, Dir, Fmt>,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        if DFT_FORWARD.get() {
            DFT_CALLS.set(DFT_CALLS.get() + 1);
            return super::carry::with_marked_bytes(DFT_EXTRA.get(), scratch, |rest| {
                CKKSDFTReference::ckks_dft_matrix_eval_assign_reference(module, ct, dft, keys, rest)
            });
        }
        let extra = DFT_EXTRA.get();
        let (mut workspace, _) = scratch.borrow().take_region(extra);
        <Self as Backend>::copy_host_to_view(&mut workspace, &vec![0x3C; extra]);
        DFT_CALLS.set(DFT_CALLS.get() + 1);
        Err(anyhow::anyhow!("DFT override probe").into())
    }
}

struct NoAutomorphismKey;
impl<BE: Backend> GetAutomorphismKey<BE, ::poulpy_hal::layouts::Standard> for NoAutomorphismKey {
    fn lookup_automorphism_key(
        &self,
        _: i64,
        _: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE, ::poulpy_hal::layouts::Standard>> {
        panic!("DFT dispatch probe must not request keys")
    }
}

#[test]
fn dft_format_wrapper_reenters_selected_evaluation() {
    use poulpy_core::layouts::{LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let mut pt = module.ckks_plaintext_alloc_from_slots(
        module.ckks_max_slots::<poulpy_hal::layouts::Standard>(),
        16usize.into(),
        meta.k,
    );
    pt.set_meta(meta.meta);
    let matrix = DFTMatrix::<OverrideBackend, Decode, Repack, _>::try_from_factor_operands(
        &module,
        DFTPlan::new(DFTType::Decode, vec![(2, 1)], DFTOutputFormat::RepackImagAsReal, meta).unwrap(),
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
    let key = GLWETensorKeyLayout {
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
        poulpy_ckks::reference::dft::ckks_dft_tmp_bytes_reference(&module, &dst, &src, &matrix, &key) + DFT_EXTRA_DEFAULT
    );
    DFT_CALLS.set(0);
    let error = with_scratch::<OverrideBackend, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_repack(&mut dst, &src, &matrix, &NoAutomorphismKey, s)
    })
    .unwrap_err();
    assert_ckks_internal("ckks_slots_to_coeffs_repack", &error, "DFT override probe");
    assert_eq!(DFT_CALLS.get(), 1);
}

impl<F: CKKSEncodingScalar> crate::ckks_encoding::CKKSEncodingTransform<F> for OverrideBackend {
    type Fft = crate::ckks_encoding::EncodingFFTTable<F>;
}
crate::impl_ckks_encoding!(OverrideBackend);
poulpy_ckks::impl_ckks_encapsulated_mod_up_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_mod_up_first_factor_reference_full!(OverrideBackend);
unsafe impl CKKSBootstrappingImpl for OverrideBackend {
    fn ckks_mod_raise_tmp_bytes<R, A, P>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<Self>,
    {
        RAISE_EXTRA.get() + CKKSModRaiseReference::ckks_mod_raise_tmp_bytes_reference(module, res, a, plan, keys_layout)
    }

    fn ckks_mod_raise<R, A, P, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<Self>,
        K: BootstrappingLinearKeys<Self>,
    {
        RAISE_CALLS.set(RAISE_CALLS.get() + 1);
        super::carry::with_marked_bytes(RAISE_EXTRA.get(), scratch, |rest| {
            CKKSModRaiseReference::ckks_mod_raise_reference(module, res, a, plan, keys, rest)
        })
    }

    fn ckks_bootstrap_eval_mod_tmp_bytes<R, A, P, F>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
    {
        BOOTSTRAP_EVAL_MOD_EXTRA.get()
            + CKKSBootstrapEvalModReference::ckks_bootstrap_eval_mod_tmp_bytes_reference(module, res, a, plan, keys_layout)
    }

    fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: Sync,
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
        K: BootstrappingKeys<Self> + Sync,
        K::TensorKey: Sync,
    {
        BOOTSTRAP_EVAL_MOD_CALLS.set(BOOTSTRAP_EVAL_MOD_CALLS.get() + 1);
        super::carry::with_marked_bytes(BOOTSTRAP_EVAL_MOD_EXTRA.get(), scratch, |rest| {
            CKKSBootstrapEvalModReference::ckks_bootstrap_eval_mod_reference(module, res, a, plan, keys, rest)
        })
    }

    fn ckks_eval_lut_tmp_bytes<R, A, P, F>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
    {
        EVAL_LUT_EXTRA.get() + CKKSEvalLutReference::ckks_eval_lut_tmp_bytes_reference(module, res, a, plan, luts, keys_layout)
    }

    fn ckks_eval_lut<R, A, P, F, K>(
        module: &Module<Self>,
        res: &mut [R],
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
        K: BootstrappingKeys<Self>,
    {
        EVAL_LUT_CALLS.set(EVAL_LUT_CALLS.get() + 1);
        super::carry::with_marked_bytes(EVAL_LUT_EXTRA.get(), scratch, |rest| {
            CKKSEvalLutReference::ckks_eval_lut_reference(module, res, a, plan, luts, keys, rest)
        })
    }
}
unsafe impl<F: CKKSEncodingScalar + DftScalar> CKKSDFTMatrixImpl<F> for OverrideBackend {
    fn ckks_dft_matrix_new_tmp_bytes<Dir: DftDirection, Fmt: DftFormat>(module: &Module<Self>, plan: &DFTPlan) -> Result<usize> {
        poulpy_ckks::reference::dft::ckks_dft_matrix_new_tmp_bytes_reference::<Dir, Fmt, Self, F>(module, plan)
    }

    fn ckks_dft_matrix_new<Dir: DftDirection, Fmt: DftFormat>(
        module: &Module<Self>,
        base2k: Base2K,
        plan: &DFTPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<DFTMatrix<Self, Dir, Fmt>> {
        poulpy_ckks::reference::dft::ckks_dft_matrix_new_reference::<Dir, Fmt, Self, F>(module, base2k, plan, scratch)
    }
}

/// The bootstrap queries take the selected DFT's workspace and each step hook's extra one for one.
#[test]
fn bootstrap_sizing_includes_selected_dft_workspace() {
    use poulpy_ckks::api::CKKSBootstrappingOps;
    use poulpy_ckks::layouts::DFTType;
    use poulpy_ckks::layouts::eval_mod::EvalModPlan;
    use poulpy_ckks::layouts::{
        BootstrappingContextPrepared, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
    };
    use poulpy_ckks::polynomial::SplitStrategy;
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let eval_mod = EvalModPlan::complex_exponential(1, 1, 0, SplitStrategy::MinDepth, meta, 16);
    let plan = BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques::default(),
        plan(DFTType::Encode, DFTOutputFormat::SplitRealAndImag),
        eval_mod,
        plan(DFTType::Decode, DFTOutputFormat::SplitRealAndImag),
    )
    .unwrap();
    type Context = BootstrappingContextPrepared<OverrideBackend, f64>;
    let compile = |plan: &BootstrappingPlan| {
        let bytes = Context::compile_unprepared_tmp_bytes(&module, plan).unwrap();
        let unprepared =
            with_scratch::<OverrideBackend, _>(bytes, |s| Context::compile_unprepared(&module, 16usize.into(), plan, s)).unwrap();
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
    let key = GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 16usize.into(),
        k_digit: 16usize.into(),
    };
    let rotation_key = GLWEAutomorphismKeyLayout {
        n: key.n,
        base2k: key.base2k,
        k_aux: key.k_aux,
        rank: key.rank,
        dnum: key.dnum,
        k_digit: key.k_digit,
    };
    let keys = BootstrappingKeysLayout {
        rotation_keys: plan
            .galois_elements(6, 128)
            .into_iter()
            .chain([-1])
            .map(|p| (p, rotation_key))
            .collect(),
        first_factor_key: None,
        conversion_keys: None,
        tensor_key: key,
        encapsulation: None,
    };
    let src = module.ckks_ciphertext_alloc::<poulpy_hal::layouts::Standard>(16usize.into(), 64usize.into());
    let dst = module.ckks_ciphertext_alloc::<poulpy_hal::layouts::Standard>(16usize.into(), 256usize.into());
    assert_query_takes_extra(
        "ckks_bootstrap_tmp_bytes",
        |bytes| DFT_EXTRA.set(bytes),
        || module.ckks_bootstrap_tmp_bytes(&dst, &src, &context, &keys),
    );
    DFT_EXTRA.set(DFT_EXTRA_DEFAULT);
    let assert_steps_take_extras = |context: &Context| {
        for (extra, default) in [
            (&RAISE_EXTRA, RAISE_EXTRA_DEFAULT),
            (&BOOTSTRAP_EVAL_MOD_EXTRA, BOOTSTRAP_EVAL_MOD_EXTRA_DEFAULT),
        ] {
            assert_query_takes_extra(
                "ckks_bootstrap_tmp_bytes",
                |bytes| extra.set(bytes),
                || module.ckks_bootstrap_tmp_bytes(&dst, &src, context, &keys),
            );
            extra.set(default);
        }
    };
    assert_steps_take_extras(&context);
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
    assert_steps_take_extras(&context);

    let host = Module::<HostBytesBackend>::new(64);
    let lut = EncodedLut::binary(
        &host,
        3.0f64,
        1.0,
        8,
        4,
        1,
        16usize.into(),
        CoeffsMeta::from_delta_budget(16, 16),
        SplitStrategy::MinDepth,
    )
    .unwrap();
    let functional = BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques::default(),
        self::plan(DFTType::Encode, DFTOutputFormat::SplitRealAndImag),
        EvalModPlan::complex_exponential(1, 1, 0, SplitStrategy::MinDepth, meta, 16),
        self::plan(DFTType::Decode, DFTOutputFormat::SplitRealAndImag),
    )
    .unwrap()
    .with_functional_bootstrap(&lut)
    .unwrap();
    let context = compile(&functional);
    let luts = [lut.transfer_to(&module)];
    for (extra, default) in [(&RAISE_EXTRA, RAISE_EXTRA_DEFAULT), (&EVAL_LUT_EXTRA, EVAL_LUT_EXTRA_DEFAULT)] {
        assert_query_takes_extra(
            "ckks_functional_bootstrap_tmp_bytes",
            |bytes| extra.set(bytes),
            || module.ckks_functional_bootstrap_tmp_bytes(&dst, &src, &context, &luts, &keys),
        );
        extra.set(default);
    }
}

#[test]
fn dft_scratch_covers_factor_copy_with_working_buffer() {
    use poulpy_core::layouts::{LinearTransformationDiagonal, LinearTransformationGiantStep};
    let module = Module::<OverrideBackend>::new(64);
    let meta = CoeffsMeta::from_delta_budget(12, 2);
    let mut pt = module.ckks_plaintext_alloc_from_slots(
        module.ckks_max_slots::<poulpy_hal::layouts::Standard>(),
        16usize.into(),
        meta.k,
    );
    pt.set_meta(meta.meta);
    let matrix = DFTMatrix::<OverrideBackend, Encode, Standard, _>::try_from_factor_operands(
        &module,
        DFTPlan::new(DFTType::Encode, vec![(1, 1)], DFTOutputFormat::Standard, meta).unwrap(),
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
    let key = GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    };
    let bytes = poulpy_ckks::reference::dft::ckks_dft_tmp_bytes_reference(&module, &ct, &ct, &matrix, &key);
    with_scratch::<OverrideBackend, _>(bytes, |s| {
        poulpy_ckks::reference::dft::ckks_dft_matrix_eval_assign_reference(&module, &mut ct, &matrix, &NoAutomorphismKey, s)
    })
    .unwrap();
    assert_eq!(ct.log_budget(), 20);
}

#[test]
fn dft_preparation_uses_selected_scratch() {
    let module = Module::<OverrideBackend>::new(64);
    let plan = plan(DFTType::Encode, DFTOutputFormat::Standard);
    type Ops = Module<OverrideBackend>;
    let matrix = with_scratch::<OverrideBackend, _>(
        <Ops as CKKSDFTMatrixOps<OverrideBackend, f64>>::ckks_dft_matrix_new_tmp_bytes::<Encode, Standard>(&module, &plan)
            .unwrap(),
        |s| {
            <Ops as CKKSDFTMatrixOps<OverrideBackend, f64>>::ckks_dft_matrix_new::<Encode, Standard>(
                &module,
                16usize.into(),
                &plan,
                s,
            )
        },
    )
    .unwrap();
    let bytes = module.ckks_dft_matrix_prepare_tmp_bytes(&matrix);
    assert_eq!(
        bytes,
        DFT_PREPARE_DEFAULT + poulpy_ckks::reference::dft::ckks_dft_matrix_prepare_tmp_bytes_reference(&module, &matrix)
    );
    let prepared = with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_dft_matrix_prepare_alloc(&matrix, s)).unwrap();
    for (factor, prepared) in matrix.factor_operands().iter().zip(prepared.factor_operands()) {
        assert_eq!(factor.index(), prepared.index());
    }
}

/// The derived `ckks_bootstrap` and `ckks_functional_bootstrap`, each on exactly its query poisoned, enter each step
/// hook behind its marked region and give the base backend's outputs; a bootstrap the entry checks refuse enters none.
#[test]
fn bootstrap_enters_each_step_hook() {
    let params = CKKSTestParams {
        n: 64,
        hw: 48,
        base2k: 12,
        ..poulpy_ckks::test_suite::BASE19_PARAMS_F64
    };
    let module = Module::<OverrideBackend>::new(64);
    let counters = [
        &RAISE_CALLS,
        &BOOTSTRAP_EVAL_MOD_CALLS,
        &EVAL_LUT_CALLS,
        &DFT_CALLS,
        &super::EVAL_MOD_CALLS,
    ];
    let counts = || counters.map(|counter| counter.get());
    counters.iter().for_each(|counter| counter.set(0));
    DFT_FORWARD.set(true);
    DFT_EXTRA.set(PROBE_DFT_EXTRA);
    super::EVAL_MOD_FORWARD.set(true);
    let got = run_bootstrap::<_, f64>(params, &module);
    // Seven raises, five EvalMod steps and two LUT steps; two transforms each, a third under EvalRound+ (two cases);
    // EvalMod per half, once on real slots without a bypass, and none under the binary LUT.
    assert_eq!(counts(), [7, 5, 2, 16, 9]);
    counters.iter().for_each(|counter| counter.set(0));
    let refused = run_refused_bootstraps::<_, f64>(params, &module);
    assert_eq!(counts(), [0; 5], "a refused bootstrap entered a hook");
    DFT_FORWARD.set(false);
    DFT_EXTRA.set(DFT_EXTRA_DEFAULT);
    super::EVAL_MOD_FORWARD.set(false);
    for (error, op, reason) in [
        (
            &refused[0],
            "ckks_bootstrap",
            "supports rank-1 ciphertexts only, got rank 2 -> 1",
        ),
        (
            &refused[1],
            "ckks_functional_bootstrap",
            "LUT outputs must share one rank-1 layout",
        ),
    ] {
        assert_ckks_error(
            op,
            error,
            CKKSCompositionError::InvalidOperand {
                op,
                reason: reason.into(),
            },
        );
    }
    let want = run_bootstrap::<FFT64Portable, f64>(params, &Module::<FFT64Portable>::new(64));
    assert_eq!(got, want, "the probe's bootstraps differ from the base backend's");
}
