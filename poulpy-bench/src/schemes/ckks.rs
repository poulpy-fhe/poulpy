use std::{collections::HashMap, hint::black_box};

use crate::schemes::params::CkksBenchParams;
use criterion::{Bencher, measurement::Measurement};
use poulpy_ckks::{
    CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSEncodingOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops, CKKSRotateOps, CKKSSubOps},
    layouts::{CKKSEncodingBuffer, CKKSModuleAlloc},
};
use poulpy_core::{
    EncryptionLayout,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout,
        GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, Rank, SetGaloisElement, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, GaloisElement, Module, ScratchOwned},
};

const ROTATION: i64 = 1;
const CONJUGATE: i64 = -1;
const POW2_BITS: usize = 3;

fn ckks_layout(cp: &CkksBenchParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n as u32),
        base2k: Base2K(cp.base2k as u32),
        k: TorusPrecision(cp.k as u32),
        rank: Rank(1),
    }
}

fn ckks_ct_meta(cp: &CkksBenchParams) -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta: cp.log_delta,
        slots: cp.slots,
    }
}

fn mul_tsk_layout(p: &CkksBenchParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = crate::core::params::key_dnum_k_aux((p.k + p.dsize * p.base2k) as u32, p.base2k as u32, p.dsize as u32);
    GLWETensorKeyLayout {
        n: Degree(p.n as u32),
        base2k: Base2K(p.base2k as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        dsize: Dsize(p.dsize as u32),
        dnum: Dnum(dnum),
    }
}

fn atk_layout(cp: &CkksBenchParams) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
    let (dnum, k_aux) =
        crate::core::params::key_dnum_k_aux((cp.k + cp.dsize * cp.base2k) as u32, cp.base2k as u32, cp.dsize as u32);
    EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(cp.n as u32),
        base2k: Base2K(cp.base2k as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        dsize: Dsize(cp.dsize as u32),
        dnum: Dnum(dnum),
    })
    .unwrap()
}

pub fn runner_ckks_add_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSAddOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_add_into(&mut ct_dst, &ct_a, &ct_b, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_mul_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSMulOps<BE> + GLWETensorKeyPreparedFactory<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let tsk_layout = mul_tsk_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_tmp_bytes(&ct_a, &ct_a, &ct_a, &tsk));

    bencher.iter(|| {
        module
            .ckks_mul_into(&mut ct_dst, &ct_a, &ct_b, &tsk, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_rotate_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSRotateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let atk_layout = atk_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut atks = HashMap::new();
    let ak = module.alloc_tensor_key_prepared_from_infos(&atk_layout);
    let mut rotate_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    rotate_key.set_p(module.galois_element(ROTATION));
    atks.insert(module.galois_element(ROTATION), rotate_key);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_rotate_tmp_bytes(&ct_a, &ak));

    bencher.iter(|| {
        module
            .ckks_rotate_into(&mut ct_dst, &ct_a, ROTATION, &atks, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_conjugate_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSConjugateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let atk_layout = atk_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut conjugate_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    conjugate_key.set_p(CONJUGATE);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_conjugate_tmp_bytes(&ct_a, &conjugate_key));

    bencher.iter(|| {
        module
            .ckks_conjugate_into(&mut ct_dst, &ct_a, &conjugate_key, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_add_pt_vec_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSAddOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_pt_vec_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_add_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_add_pt_const_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSAddOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut cst = module.ckks_pt_coeffs_alloc(2, Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    cst.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_pt_const_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_add_pt_const_into(&mut ct_dst, &ct_a, 0, &cst, 0, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_sub_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSSubOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_b = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_b.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_sub_into(&mut ct_dst, &ct_a, &ct_b, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_sub_pt_vec_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSSubOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_pt_vec_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_sub_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_sub_pt_const_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSSubOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut cst = module.ckks_pt_coeffs_alloc(2, Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    cst.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_pt_const_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_sub_pt_const_into(&mut ct_dst, &ct_a, 0, &cst, 0, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_neg_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSNegOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_neg_tmp_bytes());

    bencher.iter(|| {
        module.ckks_neg_into(&mut ct_dst, &ct_a, &mut scratch.borrow()).unwrap();
        black_box(());
    });
}

pub fn runner_ckks_mul_pow2_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSPow2Ops<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pow2_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_mul_pow2_into(&mut ct_dst, &ct_a, POW2_BITS, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_div_pow2_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSPow2Ops<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_div_pow2_tmp_bytes());

    bencher.iter(|| {
        module
            .ckks_div_pow2_into(&mut ct_dst, &ct_a, POW2_BITS, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_square_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSMulOps<BE> + GLWETensorKeyPreparedFactory<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let tsk_layout = mul_tsk_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_square_tmp_bytes(&ct_a, &ct_a, &tsk));

    bencher.iter(|| {
        module
            .ckks_square_into(&mut ct_dst, &ct_a, &tsk, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_mul_pt_vec_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSMulOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt));

    bencher.iter(|| {
        module
            .ckks_mul_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

pub fn runner_ckks_mul_pt_const_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSMulOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut const_full = module.ckks_pt_vec_alloc(Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    const_full.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full));

    bencher.iter(|| {
        module
            .ckks_mul_pt_const_into(&mut ct_dst, &ct_a, &const_full, 0, &mut scratch.borrow())
            .unwrap();
        black_box(());
    });
}

fn ckks_encoding_values(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i as f64 + 1.0) / len as f64).collect()
}

pub fn runner_ckks_encode_slots_assign_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(cp.base2k.into(), cp.k.into());
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let mut slots = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_encode_slots_assign_into(&mut pt, &mut slots).unwrap();
        black_box(());
    });
}

pub fn runner_ckks_decode_slots_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let mut seed = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);
    module.ckks_encode_slots_assign_into(&mut pt, &mut seed).unwrap();

    let mut slots = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_decode_slots_into(&pt, &mut slots).unwrap();
        black_box(());
    });
}

pub fn runner_ckks_encode_coeffs_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let coeffs = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_encode_coeffs_into(&mut pt, &coeffs).unwrap();
        black_box(());
    });
}

pub fn runner_ckks_decode_coeffs_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let seed = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);
    module.ckks_encode_coeffs_into(&mut pt, &seed).unwrap();

    let mut coeffs = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_decode_coeffs_into(&pt, &mut coeffs).unwrap();
        black_box(());
    });
}
