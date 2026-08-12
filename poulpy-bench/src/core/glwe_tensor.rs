use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use poulpy_core::{
    GLWETensoring,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEInfos, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchArenaTakeBasic, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize,
        VecZnxIdftApplyTmpA, VecZnxSubAssignBackend,
    },
    layouts::{
        Backend, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, HostDataMut, Module,
        ScratchOwned, VecZnx, VecZnxBigReborrowBackendMut, VecZnxDftReborrowBackendMut, VecZnxReborrowBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut, vec_znx_big_backend_ref_from_mut,
    },
};

use crate::params::{CoreParams, key_dnum_k_aux};

#[inline]
fn msb_mask_bottom_limb(base2k: usize, k: usize) -> i64 {
    match k % base2k {
        0 => !0i64,
        r => (!0i64) << (base2k - r),
    }
}

#[inline]
fn normalize_input_limb_bound_with_offset(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    res_offset: i64,
) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }

    full_size.min((res_size * res_base2k + offset_bits as usize).div_ceil(in_base2k))
}

fn glwe_layout(cp: &CoreParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    }
}

fn tensor_key_layout(cp: &CoreParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    GLWETensorKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    }
}

/// Relinearization (the keyswitch phase of `ckks_mul`). The tensor key is left
/// zeroed: the op is data-independent, so this times the real kernel path.
pub fn runner_glwe_tensor_relinearize<BE: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let tsk_infos = tensor_key_layout(cp);

    let module = Module::<BE>::new(cp.n as u64);

    let mut res = module.glwe_alloc_from_infos(&glwe_infos);
    let tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_relinearize_tmp_bytes(&res, &tensor, &tsk));

    bencher.iter(|| {
        module.glwe_tensor_relinearize(&mut res, &tensor, &tsk, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_apply<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let b = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    bencher.iter(|| {
        module.glwe_tensor_apply(0, &mut tensor, &a, &b, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_prepare_left<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let a_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), a.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.max_size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_left_tmp_bytes(a.max_size(), a.max_size()));

    bencher.iter(|| {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
            a_mask,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_glwe_tensor_prepare_right<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(cp.n as u64);

    let b = module.glwe_alloc_from_infos(&glwe_infos);
    let b_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), b.max_k().as_usize());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.max_size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_right_tmp_bytes(b.max_size(), b.max_size()));

    bencher.iter(|| {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
            b_mask,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_glwe_tensor_diag_lane<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>:
        ModuleNew<BE> + GLWETensoring<BE> + Convolution<BE> + CnvPVecAlloc<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    let glwe_infos = glwe_layout(cp);
    let n: usize = cp.n as usize;
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let b = module.glwe_alloc_from_infos(&glwe_infos);
    let tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let diag_dft_size = normalize_input_limb_bound_with_offset(
        a.max_size() + b.max_size() - cnv_offset_hi,
        tensor.max_size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.max_size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.max_size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.max_size(), a.max_size())
            .max(module.cnv_prepare_right_tmp_bytes(b.max_size(), b.max_size())),
    );
    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
            a_mask,
            &mut prep_scratch.borrow(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
            b_mask,
            &mut prep_scratch.borrow(),
        );
    }

    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    bencher.iter(|| {
        let scratch = scratch.borrow();
        let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(&module, 1, diag_dft_size);
        module.cnv_apply_dft(
            cnv_offset_hi,
            &mut res_dft,
            0,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch,
        );
        let (mut res_big, scratch) = scratch.take_vec_znx_big_scratch(&module, 1, diag_dft_size);
        {
            let mut res_big_backend =
                <poulpy_hal::layouts::VecZnxBig<BE::BufMut<'_>, BE> as VecZnxBigReborrowBackendMut<BE>>::reborrow_backend_mut(
                    &mut res_big,
                );
            let mut res_dft_backend =
                <poulpy_hal::layouts::VecZnxDft<BE::BufMut<'_>, BE> as VecZnxDftReborrowBackendMut<BE>>::reborrow_backend_mut(
                    &mut res_dft,
                );
            module.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
        }
        let (mut tmp, mut scratch) = scratch.take_vec_znx_scratch(n, 1, tensor.max_size());
        module.vec_znx_big_normalize(
            &mut tmp,
            base2k,
            cnv_offset_lo,
            0,
            &vec_znx_big_backend_ref_from_mut(&res_big),
            base2k,
            0,
            &mut scratch,
        );
        black_box(());
    });
}

pub fn runner_glwe_tensor_pairwise_lane<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxSubAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    let glwe_infos = glwe_layout(cp);
    let n: usize = cp.n as usize;
    let cols: usize = (glwe_infos.rank() + 1).into();
    if cols < 2 {
        // This op needs rank >= 1 (cols >= 2)
        panic!("runner_glwe_tensor_pairwise_lane: rank must be >= 1 (cols >= 2)");
    }

    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let b = module.glwe_alloc_from_infos(&glwe_infos);
    let tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let pairwise_dft_size = normalize_input_limb_bound_with_offset(
        a.max_size() + b.max_size() - cnv_offset_hi,
        tensor.max_size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.max_size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.max_size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.max_size(), a.max_size())
            .max(module.cnv_prepare_right_tmp_bytes(b.max_size(), b.max_size())),
    );
    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
            a_mask,
            &mut prep_scratch.borrow(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
            b_mask,
            &mut prep_scratch.borrow(),
        );
    }

    let mut diag_terms = module.vec_znx_alloc(cols, tensor.max_size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    {
        for i in 0..cols {
            let scratch = scratch.borrow();
            let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(&module, 1, pairwise_dft_size);
            module.cnv_apply_dft(
                cnv_offset_hi,
                &mut res_dft,
                0,
                &a_prep.to_backend_ref(),
                i,
                &b_prep.to_backend_ref(),
                i,
                &mut scratch,
            );
            let (mut res_big, scratch) = scratch.take_vec_znx_big_scratch(&module, 1, pairwise_dft_size);
            {
                let mut res_big_backend = <poulpy_hal::layouts::VecZnxBig<BE::BufMut<'_>, BE> as VecZnxBigReborrowBackendMut<
                    BE,
                >>::reborrow_backend_mut(&mut res_big);
                let mut res_dft_backend = <poulpy_hal::layouts::VecZnxDft<BE::BufMut<'_>, BE> as VecZnxDftReborrowBackendMut<
                    BE,
                >>::reborrow_backend_mut(&mut res_dft);
                module.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let (mut tmp, mut scratch) = scratch.take_vec_znx_scratch(n, 1, tensor.max_size());
            module.vec_znx_big_normalize(
                &mut tmp,
                base2k,
                cnv_offset_lo,
                0,
                &vec_znx_big_backend_ref_from_mut(&res_big),
                base2k,
                0,
                &mut scratch,
            );
            for j in 0..diag_terms.size().min(tmp.size()) {
                diag_terms.at_mut(i, j).copy_from_slice(tmp.at(0, j));
            }
            for j in tmp.size()..diag_terms.size() {
                diag_terms.at_mut(i, j).fill(0);
            }
        }
    }

    bencher.iter(|| {
        let scratch = scratch.borrow();
        let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(&module, 1, pairwise_dft_size);
        module.cnv_pairwise_apply_dft(
            cnv_offset_hi,
            &mut res_dft,
            0,
            &a_prep.to_backend_ref(),
            &b_prep.to_backend_ref(),
            0,
            1,
            &mut scratch,
        );
        let (mut res_big, scratch) = scratch.take_vec_znx_big_scratch(&module, 1, pairwise_dft_size);
        {
            let mut res_big_backend =
                <poulpy_hal::layouts::VecZnxBig<BE::BufMut<'_>, BE> as VecZnxBigReborrowBackendMut<BE>>::reborrow_backend_mut(
                    &mut res_big,
                );
            let mut res_dft_backend =
                <poulpy_hal::layouts::VecZnxDft<BE::BufMut<'_>, BE> as VecZnxDftReborrowBackendMut<BE>>::reborrow_backend_mut(
                    &mut res_dft,
                );
            module.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
        }
        let (mut tmp, mut scratch) = scratch.take_vec_znx_scratch(n, 1, tensor.max_size());
        module.vec_znx_big_normalize(
            &mut tmp,
            base2k,
            cnv_offset_lo,
            0,
            &vec_znx_big_backend_ref_from_mut(&res_big),
            base2k,
            0,
            &mut scratch,
        );
        let mut tmp_mut = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut tmp);
        let diag_terms_ref = <VecZnx<BE::OwnedBuf> as poulpy_hal::layouts::VecZnxToBackendRef<BE>>::to_backend_ref(&diag_terms);
        module.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, 0);
        module.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, 1);
        black_box(());
    });
}

pub fn runner_glwe_tensor_square_apply<BE: Backend<OwnedBuf = Vec<u8>>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    bencher.iter(|| {
        module.glwe_tensor_square_apply(0, &mut tensor, &a, &mut scratch.borrow());
        black_box(());
    });
}
