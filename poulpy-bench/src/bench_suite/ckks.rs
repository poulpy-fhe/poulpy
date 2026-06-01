use std::{collections::HashMap, hint::black_box};

use criterion::Criterion;
use poulpy_ckks::{
    CKKSMeta,
    api::{
        Diagonal, GiantStep, LinearTransformation, LinearTransformationOps, PreparedLinearTransformationLhs,
        PreparedLinearTransformationRhs,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSConjugateOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
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

const N: usize = 1 << 15;
const BASE2K: usize = 52;
const K: usize = 728;
const LOG_DELTA: usize = 40;
const DSIZE: usize = 1;
const MANY_TERMS: usize = 8;
const ROTATION: i64 = 1;
const LT_INTERLEAVED_DIAG_COUNT: usize = 256;
const LT_INTERLEAVED_DIAG_STRIDE: usize = 64;
const LT_INTERLEAVED_BSGS_N1: usize = 1024;

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
        + LinearTransformationOps<Self>
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
        + LinearTransformationOps<BE>
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

struct LinearTransformBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    ct_src: CKKSCiphertext<Vec<u8>>,
    ct_dst: CKKSCiphertext<Vec<u8>>,
    many_dsts: Vec<CKKSCiphertext<Vec<u8>>>,
    // Raw (unprepared) transforms only. Each benchmark prepares the operands it
    // needs locally; the setup keeps just the shared inputs.
    sparse_direct: LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    sparse_bsgs: LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    medium_bsgs: LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    dense_bsgs: LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    interleaved_bsgs_256: LinearTransformation<CKKSPlaintext<Vec<u8>>>,
    many_lts: [LinearTransformation<CKKSPlaintext<Vec<u8>>>; 2],
    atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    prep_scratch_bytes: usize,
    eval_scratch_bytes: usize,
    scratch_bytes: usize,
}

fn ckks_layout() -> GLWELayout {
    GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(K as u32),
        rank: Rank(1),
    }
}

fn ckks_meta() -> CKKSMeta {
    CKKSMeta {
        log_delta: LOG_DELTA,
        log_budget: K - LOG_DELTA,
    }
}

fn tsk_layout() -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision((K + DSIZE * BASE2K) as u32),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(K.div_ceil(DSIZE * BASE2K) as u32),
    }
}

fn atk_layout() -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
    let k = K + DSIZE * BASE2K;
    EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(k as u32),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(k.div_ceil(DSIZE * BASE2K) as u32),
    })
    .unwrap()
}

fn reset_dst(dst: &mut CKKSCiphertext<Vec<u8>>) {
    dst.data_mut().raw_mut().fill(0);
    dst.set_meta_checked(ckks_meta()).unwrap();
}

fn build_linear_transform<BE>(
    module: &Module<BE>,
    diag_indices: &[usize],
    n1: usize,
) -> LinearTransformation<CKKSPlaintext<Vec<u8>>>
where
    BE: CkksBenchBackend,
{
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
        let plaintext = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), ckks_meta());
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
) -> PreparedLinearTransformationRhs<BE>
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
    let mut prepared = PreparedLinearTransformationRhs::alloc_from_index(module, &lt.index(), first_pt);
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
) -> PreparedLinearTransformationLhs<BE>
where
    BE: CkksBenchBackend,
{
    let mut babies = PreparedLinearTransformationLhs::alloc(module, baby_steps, src);
    module
        .ckks_prepare_linear_transformation_lhs(&mut babies, src, atks, scratch)
        .expect("baby-step preparation failed (missing automorphism key?)");
    babies
}

fn interleaved_linear_transform_diagonals() -> Vec<usize> {
    (0..LT_INTERLEAVED_DIAG_COUNT)
        .map(|i| i * LT_INTERLEAVED_DIAG_STRIDE)
        .collect()
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
        if rotation == 0 || atks.contains_key(&rotation) {
            continue;
        }
        let mut key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
        key.set_p(module.galois_element(rotation));
        atks.insert(rotation, key);
    }
}

/// Diagonal / baby / giant / rotation counts of a transform, for the metadata
/// dump. Derived from the raw transform so the setup need not keep prepared
/// objects around.
struct LtShape {
    diagonals: usize,
    babies: usize,
    giants: usize,
    rotations: usize,
}

/// Shape of the transform evaluated directly (no BSGS pruning).
fn unprepared_shape<P>(lt: &LinearTransformation<P>) -> LtShape {
    LtShape {
        diagonals: lt.giant_steps.iter().map(|gs| gs.diagonals.len()).sum(),
        babies: lt.baby_steps.len(),
        giants: lt.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty()).count(),
        rotations: lt.required_rotations().len(),
    }
}

/// Shape of the pruned BSGS schedule the prepared evaluator actually walks
/// (matches the prepared right-operand cache built from `lt.index()`).
fn pruned_shape<P>(lt: &LinearTransformation<P>) -> LtShape {
    let index = lt.index();
    let mut rotations: std::collections::BTreeSet<i64> = index.baby_steps.iter().copied().filter(|&r| r != 0).collect();
    rotations.extend(index.giant_steps.iter().copied().filter(|&r| r != 0));
    LtShape {
        diagonals: index.index.iter().map(|babies| babies.len()).sum(),
        babies: index.baby_steps.len(),
        giants: index.giant_steps.len(),
        rotations: rotations.len(),
    }
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

fn print_linear_transform_metadata<BE>(label: &str, s: &LinearTransformBenchSetup<BE>)
where
    BE: CkksBenchBackend,
{
    eprintln!(
        "ckks_linear_transformation::{label}: scratch={} eval_tmp_bound={} prepare_tmp_bound={} key_count={}",
        format_bytes(s.scratch_bytes),
        format_bytes(s.eval_scratch_bytes),
        format_bytes(s.prep_scratch_bytes),
        s.atks.len(),
    );
    eprintln!("case,diagonals,babies,giants,required_rotations");
    let row = |case: &str, sh: LtShape| {
        eprintln!("{case},{},{},{},{}", sh.diagonals, sh.babies, sh.giants, sh.rotations);
    };
    row("one_shot_sparse_direct_shape_2", unprepared_shape(&s.sparse_direct));
    row("one_shot_sparse_bsgs_2", unprepared_shape(&s.sparse_bsgs));
    row("prepared_sparse_direct_2", pruned_shape(&s.sparse_direct));
    row("prepared_sparse_bsgs_2", pruned_shape(&s.sparse_bsgs));
    row("prepared_medium_bsgs_5", pruned_shape(&s.medium_bsgs));
    row("one_shot_medium_bsgs_5", unprepared_shape(&s.medium_bsgs));
    row("prepared_dense_bsgs_16", pruned_shape(&s.dense_bsgs));
    row(
        "prepared_interleaved_bsgs_256_stride64",
        pruned_shape(&s.interleaved_bsgs_256),
    );
    row("one_shot_dense_bsgs_16", unprepared_shape(&s.dense_bsgs));

    let many_shapes: Vec<LtShape> = s.many_lts.iter().map(pruned_shape).collect();
    let many_diagonals: usize = many_shapes.iter().map(|sh| sh.diagonals).sum();
    let many_giants: usize = many_shapes.iter().map(|sh| sh.giants).sum();
    let mut many_baby_set = std::collections::BTreeSet::new();
    let mut many_rotation_set = std::collections::BTreeSet::new();
    for lt in &s.many_lts {
        let index = lt.index();
        many_baby_set.extend(index.baby_steps.iter().copied());
        many_rotation_set.extend(index.baby_steps.iter().copied().filter(|&r| r != 0));
        many_rotation_set.extend(index.giant_steps.iter().copied().filter(|&r| r != 0));
    }
    eprintln!(
        "many_prepared_2_transforms,{many_diagonals},{},{many_giants},{}",
        many_baby_set.len(),
        many_rotation_set.len()
    );
}

fn setup<BE>() -> CkksBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    let module = Module::<BE>::new(N as u64);
    let ct_layout = ckks_layout();
    let tsk_layout = tsk_layout();
    let atk_layout = atk_layout();
    let meta = ckks_meta();

    let mut ct_a = module.ckks_ciphertext_alloc_from_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let pt = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), meta);
    let cst = module.ckks_pt_coeffs_alloc(2, Base2K(BASE2K as u32), meta);
    let const_full = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), meta);

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
        .max(module.ckks_mul_tmp_bytes(&ct_a, &tsk))
        .max(module.ckks_square_tmp_bytes(&ct_a, &tsk))
        .max(module.ckks_mul_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_rotate_tmp_bytes(&ct_a, atks.get(&ROTATION).unwrap()))
        .max(module.ckks_conjugate_tmp_bytes(&ct_a, atks.get(&-1).unwrap()))
        .max(module.ckks_add_many_tmp_bytes())
        .max(module.ckks_mul_add_ct_tmp_bytes(&ct_dst, &tsk))
        .max(module.ckks_mul_sub_ct_tmp_bytes(&ct_dst, &tsk))
        .max(module.ckks_mul_add_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_sub_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt))
        .max(module.ckks_mul_add_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_mul_sub_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full))
        .max(module.ckks_dot_product_ct_tmp_bytes(MANY_TERMS, &ct_dst, &tsk))
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

fn setup_linear_transform<BE>() -> LinearTransformBenchSetup<BE>
where
    BE: CkksBenchBackend,
{
    let module = Module::<BE>::new(N as u64);
    let ct_layout = ckks_layout();
    let meta = ckks_meta();

    let mut ct_src = module.ckks_ciphertext_alloc_from_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_infos(&ct_layout);
    ct_src.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();
    let mut many_dsts = vec![
        module.ckks_ciphertext_alloc_from_infos(&ct_layout),
        module.ckks_ciphertext_alloc_from_infos(&ct_layout),
    ];
    for dst in &mut many_dsts {
        dst.set_meta_checked(meta).unwrap();
    }

    let sparse_direct = build_linear_transform::<BE>(&module, &[2, 5], 1);
    let sparse_bsgs = build_linear_transform::<BE>(&module, &[2, 5], 3);
    let medium_bsgs = build_linear_transform::<BE>(&module, &[0, 1, 2, 5, 7], 3);
    let dense_bsgs = build_linear_transform::<BE>(&module, &(0..16).collect::<Vec<_>>(), 4);
    let interleaved_diags = interleaved_linear_transform_diagonals();
    let interleaved_bsgs_256 = build_linear_transform::<BE>(&module, &interleaved_diags, LT_INTERLEAVED_BSGS_N1);
    let many_lts = [
        build_linear_transform::<BE>(&module, &[0, 3, 6], 3),
        build_linear_transform::<BE>(&module, &[1, 2, 5], 2),
    ];

    let mut atks = HashMap::new();
    for rotations in [
        sparse_direct.required_rotations(),
        sparse_bsgs.required_rotations(),
        medium_bsgs.required_rotations(),
        dense_bsgs.required_rotations(),
        interleaved_bsgs_256.required_rotations(),
        many_lts[0].required_rotations(),
        many_lts[1].required_rotations(),
    ] {
        insert_linear_transform_keys(&module, &mut atks, rotations);
    }

    // The setup only sizes the shared scratch; each benchmark prepares its own
    // operands. A representative key sizes the LHS-prepare and eval scratch (both
    // take a ciphertext shape and one automorphism key); the RHS-prepare scratch
    // depends only on the plaintext shape, identical across every diagonal.
    let sizing_key = atks.values().next();
    let pt_infos = module.ckks_pt_vec_alloc(Base2K(BASE2K as u32), meta);
    let prep_rhs_scratch_bytes = module.ckks_prepare_linear_transformation_rhs_tmp_bytes(&pt_infos);
    let prep_lhs_scratch_bytes = sizing_key
        .map(|key| module.ckks_prepare_linear_transformation_lhs_tmp_bytes(&ct_src, key))
        .unwrap_or(0);
    let prep_scratch_bytes = prep_rhs_scratch_bytes.max(prep_lhs_scratch_bytes);
    let eval_scratch_bytes = sizing_key
        .map(|key| module.ckks_eval_linear_transformation_tmp_bytes(&ct_src, key))
        .unwrap_or(0);
    let scratch_bytes = prep_scratch_bytes.max(eval_scratch_bytes);
    let scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    LinearTransformBenchSetup {
        module,
        scratch,
        ct_src,
        ct_dst,
        many_dsts,
        sparse_direct,
        sparse_bsgs,
        medium_bsgs,
        dense_bsgs,
        interleaved_bsgs_256,
        many_lts,
        atks,
        prep_scratch_bytes,
        eval_scratch_bytes,
        scratch_bytes,
    }
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

pub fn bench_ckks_mul<BE>(c: &mut Criterion, label: &str)
where
    BE: CkksBenchBackend,
{
    let mut s = setup::<BE>();
    let mut group = c.benchmark_group(format!("ckks_mul_into::{label}"));
    group.bench_function("mul_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
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
    group.bench_function("mul_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_assign(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("square", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_square_into(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("square_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_square_assign(&mut s.ct_dst, &s.tsk, &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_pt_vec", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pt_vec_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt), &mut s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_const", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
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
    let mut s = setup_linear_transform::<BE>();
    print_linear_transform_metadata(label, &s);
    let mut group = c.benchmark_group(format!("ckks_linear_transformation::{label}"));

    // One-shot cases measure the whole pipeline (alloc + prepare RHS/LHS + eval).
    group.bench_function("one_shot_sparse_direct_shape_2", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_eval_linear_transformation_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_src),
                    black_box(&s.sparse_direct),
                    &s.atks,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("one_shot_sparse_bsgs_2", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_eval_linear_transformation_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_src),
                    black_box(&s.sparse_bsgs),
                    &s.atks,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("one_shot_medium_bsgs_5", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_eval_linear_transformation_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_src),
                    black_box(&s.medium_bsgs),
                    &s.atks,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("one_shot_dense_bsgs_16", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_eval_linear_transformation_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_src),
                    black_box(&s.dense_bsgs),
                    &s.atks,
                    &mut s.scratch.borrow(),
                )
                .unwrap();
        })
    });

    // Prepared cases prepare their own operands once (outside the measured loop)
    // and measure only the repeated evaluation.
    for (case, lt) in [
        ("prepared_sparse_direct_2", &s.sparse_direct),
        ("prepared_sparse_bsgs_2", &s.sparse_bsgs),
        ("prepared_medium_bsgs_5", &s.medium_bsgs),
        ("prepared_dense_bsgs_16", &s.dense_bsgs),
        ("prepared_interleaved_bsgs_256_stride64", &s.interleaved_bsgs_256),
    ] {
        let prepared = prepare_linear_transform(&s.module, lt, &mut s.scratch.borrow());
        let babies = prepare_babies(&s.module, prepared.baby_steps(), &s.ct_src, &s.atks, &mut s.scratch.borrow());
        group.bench_function(case, |b| {
            b.iter(|| {
                reset_dst(&mut s.ct_dst);
                s.module
                    .ckks_eval_prepared_linear_transformation_into(
                        &mut s.ct_dst,
                        black_box(&s.ct_src),
                        black_box(&prepared),
                        black_box(&babies),
                        &s.atks,
                        &mut s.scratch.borrow(),
                    )
                    .unwrap();
            })
        });
    }

    // Many-transform case: prepare every right operand and a shared baby cache
    // over their union, then measure the batched evaluation.
    {
        let prepared_many: Vec<_> = s
            .many_lts
            .iter()
            .map(|lt| prepare_linear_transform(&s.module, lt, &mut s.scratch.borrow()))
            .collect();
        let mut union = std::collections::BTreeSet::new();
        for prepared in &prepared_many {
            union.extend(prepared.baby_steps().iter().copied());
        }
        let union_baby_steps: Vec<i64> = union.into_iter().collect();
        let babies = prepare_babies(&s.module, &union_baby_steps, &s.ct_src, &s.atks, &mut s.scratch.borrow());
        group.bench_function("many_prepared_2_transforms", |b| {
            b.iter(|| {
                for dst in &mut s.many_dsts {
                    reset_dst(dst);
                }
                s.module
                    .ckks_eval_many_prepared_linear_transformations_into(
                        &mut s.many_dsts,
                        black_box(&s.ct_src),
                        black_box(&prepared_many),
                        black_box(&babies),
                        &s.atks,
                        &mut s.scratch.borrow(),
                    )
                    .unwrap();
            })
        });
    }

    group.finish();
}
