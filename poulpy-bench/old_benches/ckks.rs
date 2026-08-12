use std::{collections::HashMap, hint::black_box};

use criterion::{BenchmarkId, Criterion};
use poulpy_ckks::{
    CKKSMeta, SetCKKSInfos,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSConjugateOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
    api::{
        CKKSLinearTransformationOps, Diagonal, GiantStep, LinearTransformation, LinearTransformationBabySteps,
        LinearTransformationPrepared, LinearTransformationStrategy,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    oep::CKKSImpl,
};
use poulpy_core::{
    EncryptionLayout,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank, SetGaloisElement, TorusPrecision,
    },
    oep::{
        AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl, GGSWExternalProductImpl,
        GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEExternalProductImpl, GLWEKeyswitchImpl,
        GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl, GLWENormalizeImpl, GLWEPackImpl,
        GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl, LinearTransformationImpl,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, GaloisElement, Module, ScratchArena, ScratchOwned, ZnxViewMut},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

const N: usize = 1 << 16;
const BASE2K: usize = 52;
const K: usize = BASE2K * 24;
const LOG_DELTA: usize = 40;
const DSIZE: usize = 6;
const DNUM: usize = 4;
const MANY_TERMS: usize = 8;
const ROTATION: i64 = 1;

/// One point of the `ckks_mul` size sweep.
///
/// The number of limbs (`k = limbs * base2k`) and the gadget split (`dsize`,
/// `dnum`) are scaled down with `n`, so the benchmark shape stays representative
/// across sizes (smaller rings support smaller moduli / fewer limbs). `dnum` is
/// derived as `⌈k / (dsize * base2k)⌉`, matching `tsk_layout`.
#[derive(Clone, Copy)]
struct CkksMulParams {
    n: usize,
    base2k: usize,
    k: usize,
    log_delta: usize,
    dsize: usize,
    /// Short parameter label used in the criterion benchmark id (e.g. `logn=16`).
    label: &'static str,
}

/// `ckks_mul` size sweep: n = 2^14 .. 2^16 with limbs 6/12/24 and dsize 1/3/6.
/// At n = 2^16 this reproduces the legacy hardcoded shape (24 limbs, dsize 6,
/// dnum 4). Radix-4 NTT fusion is active only for n ≥ 2^15.
const CKKS_MUL_SWEEP: &[CkksMulParams] = &[
    CkksMulParams {
        n: 1 << 14,
        base2k: 52,
        k: 52 * 6,
        log_delta: 40,
        dsize: 1,
        label: "logn=14",
    },
    CkksMulParams {
        n: 1 << 15,
        base2k: 52,
        k: 52 * 12,
        log_delta: 40,
        dsize: 3,
        label: "logn=15",
    },
    CkksMulParams {
        n: 1 << 16,
        base2k: 52,
        k: 52 * 24,
        log_delta: 40,
        dsize: 6,
        label: "logn=16",
    },
];

pub trait CkksBenchBackend:
    Backend<OwnedBuf = Vec<u8>>
    + CKKSImpl<Self>
    + GLWEKeyswitchImpl<Self>
    + GLWEAddImpl<Self>
    + GLWESubImpl<Self>
    + GLWENegateImpl<Self>
    + GLWECopyImpl<Self>
    + GGLWEKeyswitchImpl<Self>
    + GGSWKeyswitchImpl<Self>
    + LWEKeyswitchImpl<Self>
    + GLWEExternalProductImpl<Self>
    + GGLWEExternalProductImpl<Self>
    + GGSWExternalProductImpl<Self>
    + GLWETensoringImpl<Self>
    + GLWEMulConstImpl<Self>
    + GLWEMulPlainImpl<Self>
    + GLWERotateImpl<Self>
    + GLWEMulXpMinusOneImpl<Self>
    + GLWEShiftImpl<Self>
    + GLWENormalizeImpl<Self>
    + GLWETraceImpl<Self>
    + GLWEPackImpl<Self>
    + GGSWRotateImpl<Self>
    + DecryptionImpl<Self>
    + ConversionImpl<Self>
    + AutomorphismImpl<Self>
    + HalModuleImpl<Self>
    + HalVecZnxImpl<Self>
    + HalVecZnxBigImpl<Self>
    + HalVecZnxDftImpl<Self>
    + HalSvpImpl<Self>
    + HalVmpImpl<Self>
    + HalConvolutionImpl<Self>
    + LinearTransformationImpl<Self>
where
    Self: Sized,
    Module<Self>: ModuleNew<Self>
        + ModuleCoreAlloc<OwnedBuf = Self::OwnedBuf>
        + GLWETensorKeyPreparedFactory<Self>
        + GLWEAutomorphismKeyPreparedFactory<Self>
        + CKKSAddOps<Self>
        + CKKSSubOps<Self>
        + CKKSNegOps<Self>
        + CKKSPow2Ops<Self>
        + CKKSMulOps<Self>
        + CKKSRotateOps<Self>
        + CKKSConjugateOps<Self>
        + CKKSAddManyOps<Self>
        + CKKSMulAddOps<Self>
        + CKKSMulSubOps<Self>
        + CKKSDotProductOps<Self>
        + CKKSLinearTransformationOps<Self>
        + CnvPVecAlloc<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
{
}

impl<BE> CkksBenchBackend for BE
where
    BE: Backend<OwnedBuf = Vec<u8>>
        + CKKSImpl<BE>
        + GLWEKeyswitchImpl<BE>
        + GLWEAddImpl<BE>
        + GLWESubImpl<BE>
        + GLWENegateImpl<BE>
        + GLWECopyImpl<BE>
        + GGLWEKeyswitchImpl<BE>
        + GGSWKeyswitchImpl<BE>
        + LWEKeyswitchImpl<BE>
        + GLWEExternalProductImpl<BE>
        + GGLWEExternalProductImpl<BE>
        + GGSWExternalProductImpl<BE>
        + GLWETensoringImpl<BE>
        + GLWEMulConstImpl<BE>
        + GLWEMulPlainImpl<BE>
        + GLWERotateImpl<BE>
        + GLWEMulXpMinusOneImpl<BE>
        + GLWEShiftImpl<BE>
        + GLWENormalizeImpl<BE>
        + GLWETraceImpl<BE>
        + GLWEPackImpl<BE>
        + GGSWRotateImpl<BE>
        + DecryptionImpl<BE>
        + ConversionImpl<BE>
        + AutomorphismImpl<BE>
        + HalModuleImpl<BE>
        + HalVecZnxImpl<BE>
        + HalVecZnxBigImpl<BE>
        + HalVecZnxDftImpl<BE>
        + HalSvpImpl<BE>
        + HalVmpImpl<BE>
        + HalConvolutionImpl<BE>
        + LinearTransformationImpl<BE>,
    Module<BE>: ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSMulOps<BE>
        + CKKSRotateOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddManyOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulSubOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
}

struct CkksBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    ct_a: CKKSCiphertext<Vec<u8>>,
    ct_b: CKKSCiphertext<Vec<u8>>,
    ct_dst: CKKSCiphertext<Vec<u8>>,
    pt: CKKSPlaintext<Vec<u8>>,
    cst: CKKSPlaintext<Vec<u8>>,
    const_full: CKKSPlaintext<Vec<u8>>,
    tsk: poulpy_core::layouts::GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
}

fn ckks_layout() -> GLWELayout {
    GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(K as u32),
        rank: Rank(1),
    }
}

fn ckks_ct_meta() -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta: LOG_DELTA,
    }
}

fn ckks_pt_meta() -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta: LOG_DELTA,
    }
}

fn tsk_layout() -> GLWETensorKeyLayout {
    let (dnum, k_aux) = crate::params::key_dnum_k_aux((K + DSIZE * BASE2K) as u32, BASE2K as u32, DSIZE as u32);
    GLWETensorKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(dnum),
    }
}

fn atk_layout() -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
    let (dnum, k_aux) = crate::params::key_dnum_k_aux((K + DSIZE * BASE2K) as u32, BASE2K as u32, DSIZE as u32);
    debug_assert_eq!(dnum, DNUM as u32);
    EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(dnum),
    })
    .unwrap()
}

fn reset_dst(dst: &mut CKKSCiphertext<Vec<u8>>) {
    dst.data_mut().raw_mut().fill(0);
    dst.set_meta_checked(ckks_ct_meta()).unwrap();
}

/// Resolves a [`LinearTransformationStrategy`] to the concrete baby-step count
/// `n1` used to factor the diagonals. `Direct` is `n1 = 1` (one giant step per
/// diagonal, no baby sharing). For the structure-aware optimum, build the
/// strategy with [`LinearTransformationStrategy::optimal`].
fn strategy_giant_step(strategy: LinearTransformationStrategy) -> usize {
    match strategy {
        LinearTransformationStrategy::Direct => 1,
        LinearTransformationStrategy::Bsgs { giant_step } => giant_step,
    }
    .max(1)
}

/// Builds an (unprepared) linear transformation over `diag_indices` whose BSGS
/// schedule follows `strategy`. The diagonal plaintexts are left zero — only the
/// schedule shape matters for benchmarking.
fn build_linear_transform<BE>(
    module: &Module<BE>,
    diag_indices: &[usize],
    strategy: LinearTransformationStrategy,
) -> LinearTransformation<CKKSPlaintext<Vec<u8>>>
where
    BE: CkksBenchBackend,
{
    let n1 = strategy_giant_step(strategy);
    let baby_steps: Vec<i64> = (0..n1).map(|k| k as i64).collect();
    let n2 = diag_indices.iter().copied().max().map_or(0, |i| (i / n1) + 1);
    let mut giant_steps: Vec<GiantStep<CKKSPlaintext<Vec<u8>>>> = (0..n2)
        .map(|j| GiantStep {
            rot: (n1 * j) as i64,
            diagonals: Vec::new(),
        })
        .collect();

    for &i in diag_indices {
        let j = i / n1;
        let k = i % n1;
        let mut plaintext = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), TorusPrecision(BASE2K as u32));
        plaintext.set_meta(ckks_pt_meta());
        giant_steps[j].diagonals.push(Diagonal {
            baby: k as i64,
            plaintext,
        });
    }

    LinearTransformation { baby_steps, giant_steps }
}

fn prepare_linear_transform<BE>(
    module: &Module<BE>,
    lt: &LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformationPrepared<BE>
where
    BE: CkksBenchBackend,
{
    // Phase 1: size the right-operand cache from the BSGS index and the
    // plaintext shape; phase 2: encode the diagonals into it.
    let first_pt = lt
        .giant_steps
        .iter()
        .flat_map(|gs| gs.diagonals.iter())
        .map(|d| &d.plaintext)
        .next()
        .expect("linear transformation has no diagonals");
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first_pt);
    module.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);
    prepared
}

/// Allocates and populates the prepared left operand (the baby rotations of
/// `src`) covering `baby_steps`.
fn prepare_babies<BE>(
    module: &Module<BE>,
    baby_steps: &[i64],
    src: &CKKSCiphertext<Vec<u8>>,
    atks: &HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformationBabySteps<BE>
where
    BE: CkksBenchBackend,
{
    let mut babies = LinearTransformationBabySteps::alloc(module, baby_steps, src);
    module
        .ckks_prepare_linear_transformation_baby_steps(&mut babies, src, atks, scratch)
        .expect("baby-step preparation failed (missing automorphism key?)");
    babies
}

fn insert_linear_transform_keys<BE>(
    module: &Module<BE>,
    atks: &mut HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    rotations: impl IntoIterator<Item = i64>,
) where
    BE: CkksBenchBackend,
{
    let atk_layout = atk_layout();
    for rotation in rotations {
        if rotation == 0 {
            continue;
        }
        // The linear-transformation eval and baby-prep look keys up by Galois
        // element (`get_automorphism_key(galois_element(rot))`), so the map must
        // be keyed by Galois element, not by the raw slot rotation.
        let gal_el = module.galois_element(rotation);
        if atks.contains_key(&gal_el) {
            continue;
        }
        let mut key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
        key.set_p(gal_el);
        atks.insert(gal_el, key);
    }
}

/// Distinct non-zero slot rotations a transform actually references: the babies
/// used by non-empty giant steps plus those giant steps' rotations. Replaces the
/// removed `LinearTransformation::required_rotations`; the engine now exposes
/// `galois_elements` (already mapped to Galois elements), but the key-setup path
/// here wants raw rotations.
fn required_rotations<P>(lt: &LinearTransformation<P>) -> Vec<i64> {
    let mut rotations: std::collections::BTreeSet<i64> = lt
        .giant_steps
        .iter()
        .flat_map(|gs| gs.diagonals.iter())
        .map(|d| d.baby)
        .filter(|&r| r != 0)
        .collect();
    rotations.extend(
        lt.giant_steps
            .iter()
            .filter(|gs| !gs.diagonals.is_empty())
            .map(|gs| gs.rot)
            .filter(|&r| r != 0),
    );
    rotations.into_iter().collect()
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}

fn setup<BE>() -> CkksBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    let module = Module::<BE>::new(N as u64);
    let ct_layout = ckks_layout();
    let tsk_layout = tsk_layout();
    let atk_layout = atk_layout();
    let meta = ckks_ct_meta();

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), TorusPrecision(K as u32));
    pt.set_meta(meta);
    let mut cst = module.ckks_pt_coeffs_alloc(2, Base2K(BASE2K as u32), TorusPrecision(K as u32));
    cst.set_meta(meta);
    let mut const_full = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), TorusPrecision(K as u32));
    const_full.set_meta(meta);

    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
    let mut atks = HashMap::new();
    let mut rotate_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    rotate_key.set_p(module.galois_element(ROTATION));
    atks.insert(ROTATION, rotate_key);
    let mut conjugate_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    conjugate_key.set_p(-1);
    atks.insert(-1, conjugate_key);

    let scratch_bytes = module
        .ckks_add_tmp_bytes()
        .max(module.ckks_sub_tmp_bytes())
        .max(module.ckks_neg_tmp_bytes())
        .max(module.ckks_mul_pow2_tmp_bytes())
        .max(module.ckks_div_pow2_tmp_bytes())
        .max(module.ckks_add_pt_vec_tmp_bytes())
        .max(module.ckks_sub_pt_vec_tmp_bytes())
        .max(module.ckks_add_pt_const_tmp_bytes())
        .max(module.ckks_sub_pt_const_tmp_bytes())
        .max(module.ckks_mul_tmp_bytes(&ct_a, &ct_a, &ct_a, &tsk))
        .max(module.ckks_square_tmp_bytes(&ct_a, &ct_a, &tsk))
        .max(module.ckks_mul_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_rotate_tmp_bytes(&ct_a, atks.get(&ROTATION).unwrap()))
        .max(module.ckks_conjugate_tmp_bytes(&ct_a, atks.get(&-1).unwrap()))
        .max(module.ckks_add_many_tmp_bytes())
        .max(module.ckks_mul_add_ct_tmp_bytes(&ct_dst, &ct_dst, &ct_dst, &tsk))
        .max(module.ckks_mul_sub_ct_tmp_bytes(&ct_dst, &ct_dst, &ct_dst, &tsk))
        .max(module.ckks_mul_add_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_sub_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_add_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_mul_sub_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_dot_product_ct_tmp_bytes(MANY_TERMS, &ct_dst, &ct_dst, &ct_dst, &tsk))
        .max(module.ckks_dot_product_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_dot_product_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full));

    CkksBenchSetup {
        module,
        scratch: ScratchOwned::<BE>::alloc(scratch_bytes),
        ct_a,
        ct_b,
        ct_dst,
        pt,
        cst,
        const_full,
        tsk,
        atks,
    }
}

/// Benchmarks the prepared evaluation of a single linear transformation defined
/// by its non-zero diagonal `diag_indices` and a BSGS `strategy`.
///
/// Self-contained: builds the transform, its automorphism keys, scratch, and
/// prepares both operands once (outside the measured loop), then measures only
/// `ckks_eval_linear_transformation_into` (resident `P = PreparedDiagonal`). A one-line shape summary of
/// the pruned BSGS schedule is printed before benchmarking.
fn bench_lt_case<BE>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    module: &Module<BE>,
    ct_src: &CKKSCiphertext<Vec<u8>>,
    label: &str,
    diag_indices: &[usize],
    strategy: LinearTransformationStrategy,
) where
    BE: CkksBenchBackend,
{
    let meta_ct = ckks_ct_meta();
    let meta_pt = ckks_pt_meta();
    let lt = build_linear_transform::<BE>(module, diag_indices, strategy);

    let mut atks = HashMap::new();
    insert_linear_transform_keys(module, &mut atks, required_rotations(&lt));

    // Scratch must cover the largest of RHS-prepare, LHS-prepare, and eval.
    let sizing_key = atks.values().next();
    let mut pt_infos = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), TorusPrecision(BASE2K as u32));
    pt_infos.set_meta(meta_pt);
    let scratch_bytes = module
        .ckks_prepare_linear_transformation_rhs_tmp_bytes(&pt_infos)
        .max(
            sizing_key
                .map(|key| module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(ct_src, key))
                .unwrap_or(0),
        )
        .max(
            sizing_key
                .map(|key| module.ckks_eval_linear_transformation_tmp_bytes(ct_src, key))
                .unwrap_or(0),
        );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let prepared = prepare_linear_transform(module, &lt, &mut scratch.borrow());
    let babies = prepare_babies(module, prepared.baby_steps(), ct_src, &atks, &mut scratch.borrow());

    let index = lt.index();
    let mut rotations: std::collections::BTreeSet<i64> = index.baby_steps.iter().copied().filter(|&r| r != 0).collect();
    rotations.extend(index.giant_steps.iter().copied().filter(|&r| r != 0));
    eprintln!(
        "ckks_linear_transformation/{label}: diagonals={} babies={} giants={} rotations={} keys={} scratch={}",
        index.index.iter().map(|babies| babies.len()).sum::<usize>(),
        index.baby_steps.len(),
        index.giant_steps.len(),
        rotations.len(),
        atks.len(),
        format_bytes(scratch_bytes),
    );

    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ckks_layout());
    ct_dst.set_meta_checked(meta_ct).unwrap();

    group.bench_function(label, |b| {
        b.iter(|| {
            reset_dst(&mut ct_dst);
            module
                .ckks_eval_linear_transformation_into(
                    &mut ct_dst,
                    black_box(ct_src),
                    black_box(&babies),
                    black_box(&prepared),
                    &atks,
                    &mut scratch.borrow(),
                )
                .unwrap();
        })
    });
}

pub fn bench_ckks_add<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let mut group = c.benchmark_group(format!("ckks_add_into::{label}"));
    group.bench_function("add_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.ct_b), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_assign(&mut s.ct_dst, black_box(&s.ct_a), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_pt_vec", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_const", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_const_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    0,
                    black_box(&s.cst),
                    0,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.finish();
}

pub fn bench_ckks_sub<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let mut group = c.benchmark_group(format!("ckks_sub_into::{label}"));
    group.bench_function("sub_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.ct_b), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_assign(&mut s.ct_dst, black_box(&s.ct_a), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_pt_vec", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_pt_const", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_const_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    0,
                    black_box(&s.cst),
                    0,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.finish();
}

pub fn bench_ckks_unary<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let mut group = c.benchmark_group(format!("ckks_unary::{label}"));
    group.bench_function("neg", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_neg_into(&mut s.ct_dst, black_box(&s.ct_a), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("neg_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module.ckks_neg_assign(&mut s.ct_dst).unwrap();
        })
    });
    group.bench_function("mul_pow2", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pow2_into(&mut s.ct_dst, black_box(&s.ct_a), 3, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_pow2_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pow2_assign(&mut s.ct_dst, 3, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("div_pow2", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_div_pow2_into(&mut s.ct_dst, black_box(&s.ct_a), 3, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("div_pow2_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module.ckks_div_pow2_assign(&mut s.ct_dst, 3).unwrap();
        })
    });
    group.finish();
}

fn mul_ckks_layout(p: &CkksMulParams) -> GLWELayout {
    GLWELayout {
        n: Degree(p.n as u32),
        base2k: Base2K(p.base2k as u32),
        k: TorusPrecision(p.k as u32),
        rank: Rank(1),
    }
}

fn mul_ckks_ct_meta(p: &CkksMulParams) -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta: p.log_delta,
    }
}

fn mul_tsk_layout(p: &CkksMulParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = crate::params::key_dnum_k_aux((p.k + p.dsize * p.base2k) as u32, p.base2k as u32, p.dsize as u32);
    GLWETensorKeyLayout {
        n: Degree(p.n as u32),
        base2k: Base2K(p.base2k as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        dsize: Dsize(p.dsize as u32),
        dnum: Dnum(dnum),
    }
}

fn reset_dst_meta(dst: &mut CKKSCiphertext<Vec<u8>>, meta: CKKSMeta) {
}

/// Lean per-size setup for the `ckks_mul` sweep: only the operands, plaintexts,
/// tensor key and scratch the six multiplication functions need (no automorphism
/// keys), built for the given [`CkksMulParams`].
fn mul_setup<BE>(p: &CkksMulParams) -> CkksBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    let module = Module::<BE>::new(p.n as u64);
    let ct_layout = mul_ckks_layout(p);
    let tsk_layout = mul_tsk_layout(p);
    let meta = mul_ckks_ct_meta(p);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(Base2K(p.base2k as u32), TorusPrecision(p.k as u32));
    pt.set_meta(meta);
    let mut cst = module.ckks_pt_coeffs_alloc(2, Base2K(p.base2k as u32), TorusPrecision(p.k as u32));
    cst.set_meta(meta);
    let mut const_full = module.ckks_pt_vec_alloc(Base2K(p.base2k as u32), TorusPrecision(p.k as u32));
    const_full.set_meta(meta);

    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let scratch_bytes = module
        .ckks_mul_tmp_bytes(&ct_a, &ct_a, &ct_a, &tsk)
        .max(module.ckks_square_tmp_bytes(&ct_a, &ct_a, &tsk))
        .max(module.ckks_mul_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full));

    CkksBenchSetup {
        module,
        scratch: ScratchOwned::<BE>::alloc(scratch_bytes),
        ct_a,
        ct_b,
        ct_dst,
        pt,
        cst,
        const_full,
        tsk,
        atks: HashMap::new(),
    }
}

pub fn bench_ckks_mul<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut group = c.benchmark_group(format!("ckks_mul_into::{label}"));
    for p in CKKS_MUL_SWEEP {
        let mut s = mul_setup::<BE>(p);
        let meta = mul_ckks_ct_meta(p);

        group.bench_function(BenchmarkId::new("mul_ct", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_mul_into(
                        &mut s.ct_dst,
                        black_box(&s.ct_a),
                        black_box(&s.ct_b),
                        &s.tsk,
                        &mut s.scratch.borrow(),
                    )
                    .unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("mul_ct_assign", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_mul_assign(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, &mut s.scratch.borrow())
                    .unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("square", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_square_into(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, &mut s.scratch.borrow())
                    .unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("square_assign", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_square_assign(&mut s.ct_dst, &s.tsk, &mut s.scratch.borrow())
                    .unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("mul_pt_vec", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_mul_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                    .unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("mul_const", p.label), |b| {
            b.iter(|| {
                reset_dst_meta(&mut s.ct_dst, meta);
                s.module
                    .ckks_mul_pt_const_into(
                        &mut s.ct_dst,
                        black_box(&s.ct_a),
                        black_box(&s.const_full),
                        0,
                        &mut s.scratch.borrow(),
                    )
                    .unwrap();
            })
        });
    }
    group.finish();
}

pub fn bench_ckks_automorphism<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let mut group = c.benchmark_group(format!("ckks_automorphism::{label}"));
    group.bench_function("rotate", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_rotate_into(&mut s.ct_dst, black_box(&s.ct_a), ROTATION, &s.atks, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("rotate_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            let _ = s
                .module
                .ckks_rotate_assign(&mut s.ct_dst, ROTATION, &s.atks, &mut s.scratch.borrow());
        })
    });
    group.bench_function("conjugate", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_conjugate_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    s.atks.get(&-1).unwrap(),
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("conjugate_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_conjugate_assign(&mut s.ct_dst, s.atks.get(&-1).unwrap(), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.finish();
}

pub fn bench_ckks_composite<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let many_a: Vec<&CKKSCiphertext<Vec<u8>>> = (0..MANY_TERMS).map(|_| &s.ct_a).collect();
    let many_b: Vec<&CKKSCiphertext<Vec<u8>>> = (0..MANY_TERMS).map(|_| &s.ct_b).collect();
    let pts: Vec<&_> = (0..MANY_TERMS).map(|_| &s.pt).collect();
    let const_fulls: Vec<&_> = (0..MANY_TERMS).map(|_| &s.const_full).collect();
    let pt_coeffs: Vec<usize> = vec![0; MANY_TERMS];

    let mut group = c.benchmark_group(format!("ckks_composite::{label}"));
    group.bench_function("add_many_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_many(&mut s.ct_dst, black_box(many_a.as_slice()), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_add_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_ct_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.ct_b),
                    &s.tsk,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_sub_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_ct_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.ct_b),
                    &s.tsk,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_add_pt_vec", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_vec", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_add_const", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_const_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.const_full),
                    0,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_const", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_const_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.const_full),
                    0,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_pt_vec_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_vec(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(pts.as_slice()),
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_ct_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_ct(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(many_b.as_slice()),
                    &s.tsk,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_const_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_const(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(const_fulls.as_slice()),
                    black_box(pt_coeffs.as_slice()),
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.finish();
}

pub fn bench_ckks_linear_transformation<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let module = Module::<BE>::new(N as u64);
    let mut ct_src = module.ckks_ciphertext_alloc_from_glwe_infos(&ckks_layout());
    ct_src.set_meta_checked(ckks_ct_meta()).unwrap();
    let mut group = c.benchmark_group(format!("ckks_linear_transformation::{label}"));

    use LinearTransformationStrategy::Bsgs;

    // Each case is a (label, non-zero diagonal indexes, BSGS strategy) triple.
    bench_lt_case(
        &mut group,
        &module,
        &ct_src,
        "bsgs_256",
        &(0..256).collect::<Vec<_>>(),
        Bsgs { giant_step: 16 },
    );

    group.finish();
}
