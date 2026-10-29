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
    layouts::{CKKSCiphertextOwned, CKKSEncodingBuffer, CKKSModuleAlloc},
};
use poulpy_core::{
    GLWEMaskFill, GLWENormalize,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        Base2K, Degree, Dnum, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, LWEInfos, LinearTransformation,
        LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationPlan, ModuleCoreAlloc, Rank,
        SetGaloisElement, TorusPrecision, TransformKeys, k_digit_of,
        prepared::{GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared},
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, GaloisElement, Module, PrepareHint, ScratchOwned},
    source::Source,
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

/// Diagonals of the ciphertext-diagonal transform, every one of `0..CT_LT_DIAGONALS`.
const CT_LT_DIAGONALS: i64 = 128;
/// Its BSGS giant step: 16 baby rotations and 8 giant steps.
const CT_LT_GIANT_STEP: usize = 16;

/// The operands of a transform of `CT_LT_DIAGONALS` encrypted diagonals on an encrypted vector, uniform data at their
/// layouts: the source, the transform's diagonals, its plan, and the prepared automorphism keys of the plan and
/// tensor key.
#[allow(clippy::type_complexity)]
fn ct_lt_operands<BE: Backend<ZnxWord = i64>>(
    module: &Module<BE>,
    cp: &CkksBenchParams,
) -> (
    CKKSCiphertextOwned<BE>,
    LinearTransformation<CKKSCiphertextOwned<BE>>,
    LinearTransformationPlan,
    HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
)
where
    Module<BE>: GLWEMaskFill<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = ckks_layout(cp);
    let (atk_layout, tsk_layout) = (atk_layout(cp), mul_tsk_layout(cp));
    let mut source = Source::new([0u8; 32]);
    let ciphertext = |source: &mut Source| {
        let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
        module.fill_glwe_from_source(&mut ct, source);
        ct.set_meta_checked(ckks_ct_meta(cp)).unwrap();
        ct
    };
    let src = ciphertext(&mut source);
    let plan = LinearTransformationLayout {
        indexes: (0..CT_LT_DIAGONALS).collect(),
        slots: module.ckks_max_slots(),
        strategy: LinearTransformationStrategy::Bsgs {
            giant_step: CT_LT_GIANT_STEP,
        },
    }
    .index();
    let lt = LinearTransformation {
        baby_steps: plan.baby_steps.clone(),
        giant_steps: plan
            .giant_steps
            .iter()
            .zip(&plan.index)
            .map(|(&rot, babies)| LinearTransformationGiantStep {
                rot,
                diagonals: babies
                    .iter()
                    .map(|&baby| LinearTransformationDiagonal {
                        baby,
                        plaintext: ciphertext(&mut source),
                    })
                    .collect(),
            })
            .collect(),
        rotations: plan.rotations,
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_automorphism_key_prepare_tmp_bytes(&atk_layout)
            .max(module.glwe_tensor_key_prepare_tmp_bytes(&tsk_layout)),
    );
    let atks = plan
        .galois_elements(module.cyclotomic_order())
        .into_iter()
        .map(|p| {
            let mut key = module.glwe_automorphism_key_alloc_from_infos(&atk_layout);
            fill_by_digit(module, &mut key, 1, &mut source);
            key.set_p(p);
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_layout);
            module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut scratch.borrow());
            (p, prepared)
        })
        .collect();
    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
    fill_by_digit(module, &mut tsk, 1, &mut source);
    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
    module.glwe_tensor_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());
    (src, lt, plan, atks, tsk_prepared)
}

/// `ckks_eval_linear_transformation_into` of `CT_LT_DIAGONALS` resident encrypted diagonals on an encrypted vector,
/// its baby rotations prepared.
pub fn runner_ckks_eval_linear_transformation_into_ct<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWEMaskFill<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);
    let (src, lt, plan, atks, tsk) = ct_lt_operands(&module, cp);
    let keys = TransformKeys {
        automorphism: &atks,
        tensor: &tsk,
    };
    let first = lt.first_diagonal_plaintext().unwrap();
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(&module, &plan, first);
    let mut babies = LinearTransformationBabySteps::alloc(&module, &plan.baby_steps, &src);
    let mut dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ckks_layout(cp));
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ckks_prepare_linear_transformation_rhs_tmp_bytes(first)
            .max(module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&src, &plan, &atk_layout(cp)))
            .max(module.ckks_eval_linear_transformation_tmp_bytes(&src, first, &plan, &tsk, &atk_layout(cp))),
    );
    module
        .ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_prepare_linear_transformation_baby_steps(&mut babies, &src, &atks, &mut scratch.borrow())
        .unwrap();
    bencher.iter(|| {
        module
            .ckks_eval_linear_transformation_into(&mut dst, &src, &babies, &prepared, &keys, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
        black_box(());
    });
}

/// The product of [`runner_ckks_eval_linear_transformation_into_ct`] composed from the same prepared rotations and
/// diagonals: one `ckks_dot_product_prepared` per giant step, rotated and added, then normalized.
pub fn runner_ckks_eval_linear_transformation_into_ct_composed<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CkksBenchParams,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + CKKSMulOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + GLWENormalize<BE>
        + GLWEMaskFill<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let module = Module::<BE>::new(cp.n as u64);
    let (src, lt, plan, atks, tsk) = ct_lt_operands(&module, cp);
    let ct_layout = ckks_layout(cp);
    let mut rotated = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut term = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut dst = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ckks_prepare_tmp_bytes(&src)
            .max(module.ckks_rotate_tmp_bytes(&src, &atk_layout(cp)))
            .max(module.ckks_dot_product_ct_tmp_bytes(CT_LT_GIANT_STEP, &src, &src, &src, &tsk))
            .max(module.ckks_add_tmp_bytes(ct_layout.size()))
            .max(module.glwe_normalize_tmp_bytes()),
    );
    let lefts: HashMap<i64, _> = plan
        .baby_steps
        .iter()
        .map(|&baby| {
            let mut left = module.ckks_prepared_left_alloc_from_infos(&src, PrepareHint::Reuse);
            if baby != 0 {
                module
                    .ckks_rotate_into(baby, &mut rotated, &src, &atks, &mut scratch.borrow())
                    .unwrap();
            }
            let ct = if baby == 0 { &src } else { &rotated };
            module.ckks_prepare_left_into(&mut left, ct, &mut scratch.borrow()).unwrap();
            (baby, left)
        })
        .collect();
    let rights: Vec<Vec<_>> = lt
        .giant_steps
        .iter()
        .map(|gs| {
            gs.diagonals
                .iter()
                .map(|d| {
                    let mut right = module.ckks_prepared_right_alloc_from_infos(&d.plaintext, PrepareHint::Reuse);
                    module
                        .ckks_prepare_right_into(&mut right, &d.plaintext, &mut scratch.borrow())
                        .unwrap();
                    right
                })
                .collect()
        })
        .collect();
    let steps: Vec<(i64, Vec<_>, Vec<_>)> = lt
        .giant_steps
        .iter()
        .zip(&rights)
        .map(|(gs, rights)| {
            (
                gs.rot,
                gs.diagonals.iter().map(|d| &lefts[&d.baby]).collect(),
                rights.iter().collect(),
            )
        })
        .collect();
    bencher.iter(|| {
        for (g, (rot, a, b)) in steps.iter().enumerate() {
            let out = if g == 0 { &mut dst } else { &mut term };
            module
                .ckks_dot_product_prepared(out, a, b, &tsk, &mut scratch.borrow())
                .unwrap();
            if *rot != 0 {
                module.ckks_rotate_assign(*rot, out, &atks, &mut scratch.borrow()).unwrap();
            }
            if g != 0 {
                module.ckks_add_assign(&mut dst, &term, &mut scratch.borrow()).unwrap();
            }
        }
        module.glwe_normalize_assign(&mut dst, &mut scratch.borrow());
        module.synchronize();
        black_box(());
    });
}
