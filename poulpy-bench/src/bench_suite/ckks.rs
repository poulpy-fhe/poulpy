use std::{collections::HashMap, hint::black_box};

use criterion::Criterion;
use poulpy_ckks::{
    CKKSMeta,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSConjugateOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};
use poulpy_core::{
    EncryptionLayout, ScratchArenaTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank, SetGaloisElement, TorusPrecision,
    },
    oep::{
        AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl, GGSWExternalProductImpl,
        GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEExternalProductImpl, GLWEKeyswitchImpl,
        GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl, GLWENormalizeImpl, GLWEPackImpl,
        GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
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
        + CKKSDotProductOps<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable + ScratchArenaTakeCore<'a, Self>,
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
        + HalConvolutionImpl<BE>,
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
        + CKKSDotProductOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
}

struct CkksBenchSetup<BE: CkksBenchBackend> {
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

fn setup<BE: CkksBenchBackend>() -> CkksBenchSetup<BE> {
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

pub fn bench_ckks_add<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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

pub fn bench_ckks_sub<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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

pub fn bench_ckks_unary<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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

pub fn bench_ckks_mul<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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

pub fn bench_ckks_automorphism<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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

pub fn bench_ckks_composite<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
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
