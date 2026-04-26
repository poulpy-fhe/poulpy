use std::{collections::HashMap, hint::black_box};

use criterion::Criterion;
use poulpy_ckks::{
    CKKSInfos, CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx, alloc_pt_znx},
    },
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSConjugateOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulManyOps, CKKSMulOps,
        CKKSMulSubOps, CKKSNegOps, CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};
use poulpy_core::{
    EncryptionLayout, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, Rank, SetGaloisElement, TorusPrecision,
    },
    oep::CoreImpl,
};
use poulpy_hal::{
    api::{ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DeviceBuf, GaloisElement, Module, Scratch, ScratchOwned, ZnxViewMut},
    oep::HalImpl,
};

const N: usize = 1 << 15;
const BASE2K: usize = 52;
const K: usize = 728;
const LOG_DECIMAL: usize = 40;
const DSIZE: usize = 1;
const MANY_TERMS: usize = 8;
const ROTATION: i64 = 1;

pub trait CkksBenchBackend: Backend + CKKSImpl<Self> + CoreImpl<Self> + HalImpl<Self>
where
    Self: Sized,
    Module<Self>: ModuleNew<Self>
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
        + CKKSMulManyOps<Self>
        + CKKSMulAddOps<Self>
        + CKKSMulSubOps<Self>
        + CKKSDotProductOps<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    Scratch<Self>: ScratchAvailable + ScratchTakeCore<Self>,
{
}

impl<BE> CkksBenchBackend for BE
where
    BE: Backend + CKKSImpl<BE> + CoreImpl<BE> + HalImpl<BE>,
    Module<BE>: ModuleNew<BE>
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
        + CKKSMulManyOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulSubOps<BE>
        + CKKSDotProductOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
}

struct CkksBenchSetup<BE: CkksBenchBackend> {
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    ct_a: CKKSCiphertext<Vec<u8>>,
    ct_b: CKKSCiphertext<Vec<u8>>,
    ct_dst: CKKSCiphertext<Vec<u8>>,
    pt_znx: poulpy_ckks::layouts::plaintext::CKKSPlaintextVecZnx<Vec<u8>>,
    pt_rnx: CKKSPlaintextVecRnx<f64>,
    cst_znx: CKKSPlaintextCstZnx,
    cst_rnx: CKKSPlaintextCstRnx<f64>,
    tsk: poulpy_core::layouts::GLWETensorKeyPrepared<DeviceBuf<BE>, BE>,
    atks: HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>>,
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
        log_delta: LOG_DECIMAL,
        log_budget: K - LOG_DECIMAL,
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
    dst.meta = ckks_meta();
}

fn setup<BE: CkksBenchBackend>() -> CkksBenchSetup<BE> {
    let module = Module::<BE>::new(N as u64);
    let ct_layout = ckks_layout();
    let tsk_layout = tsk_layout();
    let atk_layout = atk_layout();
    let meta = ckks_meta();

    let mut ct_a = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    let mut ct_b = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    let mut ct_dst = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    ct_a.meta = meta;
    ct_b.meta = meta;
    ct_dst.meta = meta;

    let pt_znx = alloc_pt_znx(Degree(N as u32), Base2K(BASE2K as u32), meta);
    let pt_rnx = CKKSPlaintextVecRnx::<f64>::alloc(N).unwrap();
    let cst_rnx = CKKSPlaintextCstRnx::new(Some(1.25), Some(-0.5));
    let cst_znx = CKKSPlaintextCstZnx::new(
        Some(vec![0; meta.min_k(Base2K(BASE2K as u32)).as_usize().div_ceil(BASE2K)]),
        None,
        meta,
    );

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
        .max(module.ckks_add_pt_vec_znx_tmp_bytes())
        .max(module.ckks_sub_pt_vec_znx_tmp_bytes())
        .max(module.ckks_add_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_sub_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_add_pt_const_tmp_bytes())
        .max(module.ckks_sub_pt_const_tmp_bytes())
        .max(module.ckks_mul_tmp_bytes(&ct_layout, &tsk_layout))
        .max(module.ckks_square_tmp_bytes(&ct_layout, &tsk_layout))
        .max(module.ckks_mul_pt_vec_znx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_pt_const_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_rotate_tmp_bytes(&ct_layout, &atk_layout))
        .max(module.ckks_conjugate_tmp_bytes(&ct_layout, &atk_layout))
        .max(module.ckks_add_many_tmp_bytes())
        .max(module.ckks_mul_many_tmp_bytes(MANY_TERMS, &ct_layout, &tsk_layout))
        .max(module.ckks_mul_add_ct_tmp_bytes(&ct_layout, &tsk_layout))
        .max(module.ckks_mul_sub_ct_tmp_bytes(&ct_layout, &tsk_layout))
        .max(module.ckks_mul_add_pt_vec_znx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_sub_pt_vec_znx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_add_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_sub_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_add_pt_const_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_mul_sub_pt_const_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_dot_product_ct_tmp_bytes(MANY_TERMS, &ct_layout, &tsk_layout))
        .max(module.ckks_dot_product_pt_vec_znx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_dot_product_pt_vec_rnx_tmp_bytes(&ct_layout, &ct_layout, &meta))
        .max(module.ckks_dot_product_pt_const_tmp_bytes(&ct_layout, &ct_layout, &meta));

    CkksBenchSetup {
        module,
        scratch: ScratchOwned::<BE>::alloc(scratch_bytes),
        ct_a,
        ct_b,
        ct_dst,
        pt_znx,
        pt_rnx,
        cst_znx,
        cst_rnx,
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
                .ckks_add_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.ct_b), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_assign(&mut s.ct_dst, black_box(&s.ct_a), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_pt_vec_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_vec_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_pt_vec_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_vec_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.pt_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("add_const_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_const_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.cst_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("add_const_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_pt_const_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.cst_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
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
                .ckks_sub_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.ct_b), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_assign(&mut s.ct_dst, black_box(&s.ct_a), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_pt_vec_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_vec_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_pt_vec_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_vec_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.pt_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("sub_pt_const_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_const_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.cst_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("sub_pt_const_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_sub_pt_const_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.cst_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
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
                .ckks_neg_into(&mut s.ct_dst, black_box(&s.ct_a), s.scratch.borrow())
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
                .ckks_mul_pow2_into(&mut s.ct_dst, black_box(&s.ct_a), 3, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_pow2_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module.ckks_mul_pow2_assign(&mut s.ct_dst, 3, s.scratch.borrow()).unwrap();
        })
    });
    group.bench_function("div_pow2", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_div_pow2_into(&mut s.ct_dst, black_box(&s.ct_a), 3, s.scratch.borrow())
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
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_ct_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_assign(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("square", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_square_into(&mut s.ct_dst, black_box(&s.ct_a), &s.tsk, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("square_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_square_assign(&mut s.ct_dst, &s.tsk, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_pt_vec_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pt_vec_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_pt_vec_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pt_vec_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.pt_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_const_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pt_const_znx_into(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.cst_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_const_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_pt_const_rnx_into(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.cst_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
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
                .ckks_rotate_into(&mut s.ct_dst, black_box(&s.ct_a), ROTATION, &s.atks, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("rotate_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            let _ = s
                .module
                .ckks_rotate_assign::<_, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>>(
                    &mut s.ct_dst,
                    ROTATION,
                    &s.atks,
                    s.scratch.borrow(),
                );
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
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("conjugate_assign", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_conjugate_assign(&mut s.ct_dst, s.atks.get(&-1).unwrap(), s.scratch.borrow())
                .unwrap();
        })
    });
    group.finish();
}

pub fn bench_ckks_composite<BE: CkksBenchBackend>(c: &mut Criterion, label: &str) {
    let mut s = setup::<BE>();
    let many_a: Vec<&CKKSCiphertext<Vec<u8>>> = (0..MANY_TERMS).map(|_| &s.ct_a).collect();
    let many_b: Vec<&CKKSCiphertext<Vec<u8>>> = (0..MANY_TERMS).map(|_| &s.ct_b).collect();
    let pt_znxs: Vec<&_> = (0..MANY_TERMS).map(|_| &s.pt_znx).collect();
    let pt_rnxs: Vec<&_> = (0..MANY_TERMS).map(|_| &s.pt_rnx).collect();
    let cst_znxs: Vec<&_> = (0..MANY_TERMS).map(|_| &s.cst_znx).collect();
    let cst_rnxs: Vec<&_> = (0..MANY_TERMS).map(|_| &s.cst_rnx).collect();

    let mut group = c.benchmark_group(format!("ckks_composite::{label}"));
    group.bench_function("add_many_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_add_many(&mut s.ct_dst, black_box(many_a.as_slice()), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_many_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_many(&mut s.ct_dst, black_box(many_a.as_slice()), &s.tsk, s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_add_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_ct(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.ct_b),
                    &s.tsk,
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_sub_ct", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_ct(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.ct_b),
                    &s.tsk,
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_add_pt_vec_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_vec_znx(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_vec_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_vec_znx(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.pt_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_add_pt_vec_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_vec_rnx(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.pt_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_vec_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_vec_rnx(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.pt_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_add_const_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_const_znx(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.cst_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_const_znx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_const_znx(&mut s.ct_dst, black_box(&s.ct_a), black_box(&s.cst_znx), s.scratch.borrow())
                .unwrap();
        })
    });
    group.bench_function("mul_add_const_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_add_pt_const_rnx(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.cst_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("mul_sub_pt_const_rnx", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_mul_sub_pt_const_rnx(
                    &mut s.ct_dst,
                    black_box(&s.ct_a),
                    black_box(&s.cst_rnx),
                    ckks_meta(),
                    s.scratch.borrow(),
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
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_pt_vec_znx_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_vec_znx(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(pt_znxs.as_slice()),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_pt_vec_rnx_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_vec_rnx(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(pt_rnxs.as_slice()),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_const_znx_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_const_znx(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(cst_znxs.as_slice()),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.bench_function("dot_product_const_rnx_8", |b| {
        b.iter(|| {
            reset_dst(&mut s.ct_dst);
            s.module
                .ckks_dot_product_pt_const_rnx(
                    &mut s.ct_dst,
                    black_box(many_a.as_slice()),
                    black_box(cst_rnxs.as_slice()),
                    ckks_meta(),
                    s.scratch.borrow(),
                )
                .unwrap();
        })
    });
    group.finish();
}
