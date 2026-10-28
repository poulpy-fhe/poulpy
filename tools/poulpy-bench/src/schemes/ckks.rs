use poulpy_ckks::CKKSModuleInfos;
use poulpy_ckks::Scale;
use poulpy_hal::api::ModuleSynchronize;
use std::{collections::HashMap, hint::black_box};

use crate::schemes::params::CkksBenchParams;
use criterion::{Bencher, measurement::Measurement};
use poulpy_ckks::{
    GLWEPlaintextMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSDotProductOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps, LinearTransformationBabySteps, LinearTransformationLayout,
        LinearTransformationPrepared, LinearTransformationStrategy,
    },
    layouts::{CKKSEncodingBuffer, CKKSModuleAlloc},
};
use poulpy_core::{
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        Base2K, Degree, Dnum, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, GetAutomorphismKey, GetTensorKey,
        LWEInfos, Rank, SetGaloisElement, TorusPrecision, k_digit_of,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPrepared,
            GLWETensorKeyPreparedBackendRef,
        },
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, GaloisElement, Module, PrepareHint, ScratchOwned},
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

fn ckks_ct_meta(cp: &CkksBenchParams) -> GLWEPlaintextMeta {
    GLWEPlaintextMeta {
        scale: Scale::Log(cp.log_delta),
        slots: cp.slots,
        log_sparsity: 0,
    }
}

fn mul_tsk_layout(p: &CkksBenchParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = crate::core::params::key_dnum_k_aux((p.k + p.dsize * p.base2k) as u32, p.base2k as u32, p.dsize as u32);
    GLWETensorKeyLayout {
        n: Degree(p.n as u32),
        base2k: Base2K(p.base2k as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        k_digit: k_digit_of(p.base2k as u32, p.dsize as u32),
        dnum: Dnum(dnum),
    }
}

fn atk_layout(cp: &CkksBenchParams) -> GLWEAutomorphismKeyLayout {
    let (dnum, k_aux) =
        crate::core::params::key_dnum_k_aux((cp.k + cp.dsize * cp.base2k) as u32, cp.base2k as u32, cp.dsize as u32);
    GLWEAutomorphismKeyLayout {
        n: Degree(cp.n as u32),
        base2k: Base2K(cp.base2k as u32),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(1),
        k_digit: k_digit_of(cp.base2k as u32, cp.dsize as u32),
        dnum: Dnum(dnum),
    }
}

pub fn runner_ckks_add_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSAddOps<BE>,
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

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_add_into(&mut ct_dst, &ct_a, &ct_b, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_mul_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSMulOps<BE> + GLWETensorKeyPreparedFactory<BE>,
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
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_rotate_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSRotateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
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
    let mut rotate_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
    rotate_key.set_p(module.galois_element(ROTATION));
    atks.insert(module.galois_element(ROTATION), rotate_key);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_rotate_tmp_bytes(&ct_a, &atk_layout));

    bencher.iter(|| {
        module
            .ckks_rotate_into(ROTATION, &mut ct_dst, &ct_a, &atks, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_conjugate_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSConjugateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
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
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_add_pt_vec_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSAddOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_pt_vec_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_add_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_add_pt_const_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSAddOps<BE>,
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

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_add_pt_const_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_add_pt_const_into(&mut ct_dst, 0, &ct_a, &cst, 0, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_sub_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSSubOps<BE>,
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

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_sub_into(&mut ct_dst, &ct_a, &ct_b, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_sub_pt_vec_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSSubOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_pt_vec_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_sub_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_sub_pt_const_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSSubOps<BE>,
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

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_sub_pt_const_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_sub_pt_const_into(&mut ct_dst, 0, &ct_a, &cst, 0, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_neg_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSNegOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_neg_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module.ckks_neg_into(&mut ct_dst, &ct_a, &mut scratch.borrow()).unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_mul_pow2_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSPow2Ops<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pow2_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_mul_pow2_into(POW2_BITS, &mut ct_dst, &ct_a, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_div_pow2_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSPow2Ops<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_div_pow2_tmp_bytes(ct_layout.size()));

    bencher.iter(|| {
        module
            .ckks_div_pow2_into(POW2_BITS, &mut ct_dst, &ct_a, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_square_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSMulOps<BE> + GLWETensorKeyPreparedFactory<BE>,
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
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_mul_pt_vec_into<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CkksBenchParams)
where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSMulOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    pt.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pt_vec_tmp_bytes(&ct_dst, &ct_a, &pt));

    bencher.iter(|| {
        module
            .ckks_mul_pt_vec_into(&mut ct_dst, &ct_a, &pt, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_mul_pt_const_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSMulOps<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let ct_layout = ckks_layout(cp);
    let meta = ckks_ct_meta(cp);

    let mut ct_a = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut ct_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    ct_a.set_meta_checked(meta).unwrap();
    ct_dst.set_meta_checked(meta).unwrap();

    let mut const_full = module.ckks_pt_vec_alloc(module.ckks_max_slots(), Base2K(cp.base2k as u32), TorusPrecision(cp.k as u32));
    const_full.set_meta(meta);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_mul_pt_const_tmp_bytes(&ct_dst, &ct_a, &const_full));

    bencher.iter(|| {
        module
            .ckks_mul_pt_const_into(&mut ct_dst, &ct_a, &const_full, 0, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

fn ckks_encoding_values(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i as f64 + 1.0) / len as f64).collect()
}

pub fn runner_ckks_encode_slots_assign_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), cp.base2k.into(), cp.k.into());
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let mut slots = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_encode_slots_assign_into(&mut pt, &mut slots).unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_decode_slots_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let mut seed = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);
    module.ckks_encode_slots_assign_into(&mut pt, &mut seed).unwrap();

    let mut slots = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_decode_slots_into(&pt, &mut slots).unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_encode_coeffs_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let coeffs = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_encode_coeffs_into(&mut pt, &coeffs).unwrap();
        module.synchronize();
        black_box(());
    });
}

pub fn runner_ckks_decode_coeffs_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE> + ModuleSynchronize<BE> + CKKSEncodingOps<BE, f64>,
{
    let module = Module::<BE>::new(cp.n as u64);

    let mut pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), cp.base2k.into(), 127usize.into()); // TODO: should this be a benchmark parameter?
    pt.set_meta(ckks_ct_meta(cp));

    let values = ckks_encoding_values(cp.n);
    let seed = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);
    module.ckks_encode_coeffs_into(&mut pt, &seed).unwrap();

    let mut coeffs = CKKSEncodingBuffer::<BE::OwnedBuf, f64>::from_host::<BE>(&values);

    bencher.iter(|| {
        module.ckks_decode_coeffs_into(&pt, &mut coeffs).unwrap();
        module.synchronize();
        black_box(());
    });
}

/// The automorphism keys and the tensor key of a transform of ciphertext diagonals.
struct TransformKeys<'a, BE: Backend> {
    atks: &'a HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    tsk: &'a GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
}

impl<BE: Backend> GetAutomorphismKey<BE> for TransformKeys<'_, BE> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.atks.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend> GetTensorKey<BE> for TransformKeys<'_, BE> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.tsk.lookup_tensor_key(k)
    }
}

/// Diagonals of the ciphertext-diagonal transform, every one of `0..CT_LT_DIAGONALS`.
const CT_LT_DIAGONALS: i64 = 128;
/// Its BSGS giant step: 16 baby rotations and 8 giant steps.
const CT_LT_GIANT_STEP: usize = 16;

/// `ckks_prepare_linear_transformation_baby_steps` then `ckks_eval_linear_transformation_into`: a matrix of
/// `CT_LT_DIAGONALS` encrypted diagonals, resident, times an encrypted vector.
pub fn runner_ckks_eval_ct_linear_transformation_into<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSMulOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CnvPVecAlloc<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
{
    ckks_ct_linear_transformation::<BE, M, false>(bencher, cp);
}

/// The same product as [`runner_ckks_eval_ct_linear_transformation_into`] composed: each baby rotation rotated and
/// prepared on its own, one `ckks_dot_product_prepared` per giant step, then rotated and added.
pub fn runner_ckks_eval_ct_linear_transformation_composed<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSMulOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CnvPVecAlloc<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
{
    ckks_ct_linear_transformation::<BE, M, true>(bencher, cp);
}

fn ckks_ct_linear_transformation<BE: Backend<ZnxWord = i64>, M: Measurement, const COMPOSED: bool>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSMulOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CnvPVecAlloc<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);
    let ct_layout = ckks_layout(cp);
    let atk_layout = atk_layout(cp);
    let meta = ckks_ct_meta(cp);
    let mut src = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    src.set_meta_checked(meta).unwrap();
    dst.set_meta_checked(meta).unwrap();
    let tsk = module.alloc_tensor_key_prepared_from_infos(&mul_tsk_layout(cp));

    let layout = LinearTransformationLayout {
        indexes: (0..CT_LT_DIAGONALS).collect(),
        slots: cp.n / 2,
        strategy: LinearTransformationStrategy::Bsgs {
            giant_step: CT_LT_GIANT_STEP,
        },
    };
    let plan = layout.index();
    let mut atks = HashMap::new();
    for p in plan.galois_elements(module.cyclotomic_order()) {
        let mut key = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
        key.set_p(p);
        atks.insert(p, key);
    }
    let keys = TransformKeys { atks: &atks, tsk: &tsk };

    if !COMPOSED {
        // The diagonals as ciphertexts of `src`'s layout and scale, resident.
        let mut lt = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(&module, &plan, &src);
        lt.set_log_scale(cp.log_delta);
        let mut babies = LinearTransformationBabySteps::alloc(&module, &plan.baby_steps, &src);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&src, &plan, &atk_layout)
                .max(module.ckks_eval_linear_transformation_tmp_bytes(&src, &src, &plan, &atk_layout, &tsk)),
        );
        bencher.iter(|| {
            module
                .ckks_prepare_linear_transformation_baby_steps(&mut babies, &src, &atks, &mut scratch.borrow())
                .unwrap();
            module
                .ckks_eval_linear_transformation_into(&mut dst, &src, &babies, &lt, &keys, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
            black_box(());
        });
        return;
    }

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ckks_prepare_tmp_bytes(&src));
    let rights: Vec<Vec<_>> = plan
        .index
        .iter()
        .map(|babies| {
            babies
                .iter()
                .map(|_| {
                    let mut right = module.ckks_prepared_right_alloc_from_infos(&src, PrepareHint::Reuse);
                    module
                        .ckks_prepare_right_into(&mut right, &src, &mut scratch.borrow())
                        .unwrap();
                    right
                })
                .collect()
        })
        .collect();
    let mut lefts: Vec<_> = plan
        .baby_steps
        .iter()
        .map(|_| module.ckks_prepared_left_alloc_from_infos(&src, PrepareHint::Reuse))
        .collect();
    let mut rotated = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut term = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let baby_index = |baby: i64| plan.baby_steps.iter().position(|&b| b == baby).unwrap();
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ckks_prepare_tmp_bytes(&src)
            .max(module.ckks_rotate_tmp_bytes(&src, &atk_layout))
            .max(module.ckks_dot_product_ct_tmp_bytes(CT_LT_GIANT_STEP, &src, &src, &src, &tsk))
            .max(module.ckks_add_tmp_bytes(ct_layout.size())),
    );
    bencher.iter(|| {
        for (left, &baby) in lefts.iter_mut().zip(&plan.baby_steps) {
            if baby == 0 {
                module.ckks_prepare_left_into(left, &src, &mut scratch.borrow()).unwrap();
            } else {
                module
                    .ckks_rotate_into(baby, &mut rotated, &src, &atks, &mut scratch.borrow())
                    .unwrap();
                module.ckks_prepare_left_into(left, &rotated, &mut scratch.borrow()).unwrap();
            }
        }
        for (g, (&rot, babies)) in plan.giant_steps.iter().zip(&plan.index).enumerate() {
            let a: Vec<_> = babies.iter().map(|&baby| &lefts[baby_index(baby)]).collect();
            let b: Vec<_> = rights[g].iter().collect();
            let out = if g == 0 { &mut dst } else { &mut term };
            module
                .ckks_dot_product_prepared(out, &a, &b, &tsk, &mut scratch.borrow())
                .unwrap();
            if rot != 0 {
                module.ckks_rotate_assign(rot, out, &atks, &mut scratch.borrow()).unwrap();
            }
            if g != 0 {
                module.ckks_add_assign(&mut dst, &term, &mut scratch.borrow()).unwrap();
            }
        }
        module.synchronize();
        black_box(());
    });
}
