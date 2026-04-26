use std::hint::black_box;

use criterion::Criterion;
use poulpy_core::{
    GLWETensoring, ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWETensor, LWEInfos},
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic,
        VecZnxBigNormalize, VecZnxCopy, VecZnxIdftApplyConsume, VecZnxSubAssign,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned, VecZnx},
};

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

pub fn bench_glwe_tensor_apply<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let n: usize = glwe_infos.n().into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    let group_name = format!("glwe_tensor_apply::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_tensor_apply(
                0,
                &mut tensor,
                &a,
                a.max_k().as_usize(),
                &b,
                b.max_k().as_usize(),
                scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_apply_add_assign<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let n: usize = glwe_infos.n().into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    let group_name = format!("glwe_tensor_apply_add_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_tensor_apply_add_assign(
                0,
                &mut tensor,
                &a,
                a.max_k().as_usize(),
                &b,
                b.max_k().as_usize(),
                scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_prepare_left<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let a_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), a.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_left_tmp_bytes(a.size(), a.size()));

    let group_name = format!("glwe_tensor_prepare_left::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.cnv_prepare_left(&mut a_prep, a.data(), a_mask, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_prepare_right<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), b.max_k().as_usize());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_right_tmp_bytes(b.size(), b.size()));

    let group_name = format!("glwe_tensor_prepare_right::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.cnv_prepare_right(&mut b_prep, b.data(), b_mask, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_diag_lane<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable + ScratchTakeBasic,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let diag_dft_size = normalize_input_limb_bound_with_offset(
        a.size() + b.size() - cnv_offset_hi,
        tensor.size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.size(), a.size())
            .max(module.cnv_prepare_right_tmp_bytes(b.size(), b.size())),
    );
    module.cnv_prepare_left(&mut a_prep, a.data(), a_mask, prep_scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, b.data(), b_mask, prep_scratch.borrow());

    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    let group_name = format!("glwe_tensor_diag_lane::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            let scratch = scratch.borrow();
            let (mut res_dft, scratch) = scratch.take_vec_znx_dft(&module, 1, diag_dft_size);
            module.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, 0, &b_prep, 0, scratch);
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(&mut tmp, base2k, cnv_offset_lo, 0, &res_big, base2k, 0, scratch);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_pairwise_lane<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy
        + VecZnxSubAssign,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable + ScratchTakeBasic,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    if cols < 2 {
        return;
    }

    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let pairwise_dft_size = normalize_input_limb_bound_with_offset(
        a.size() + b.size() - cnv_offset_hi,
        tensor.size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.size(), a.size())
            .max(module.cnv_prepare_right_tmp_bytes(b.size(), b.size())),
    );
    module.cnv_prepare_left(&mut a_prep, a.data(), a_mask, prep_scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, b.data(), b_mask, prep_scratch.borrow());

    let mut diag_terms = VecZnx::alloc(n, cols, tensor.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    {
        let scratch = scratch.borrow();
        for i in 0..cols {
            let (mut res_dft, scratch) = scratch.take_vec_znx_dft(&module, 1, pairwise_dft_size);
            module.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, i, scratch);
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(&mut tmp, base2k, cnv_offset_lo, 0, &res_big, base2k, 0, scratch);
            module.vec_znx_copy(&mut diag_terms, i, &tmp, 0);
        }
    }

    let group_name = format!("glwe_tensor_pairwise_lane::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            let scratch = scratch.borrow();
            let (mut res_dft, scratch) = scratch.take_vec_znx_dft(&module, 1, pairwise_dft_size);
            module.cnv_pairwise_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, &b_prep, 0, 1, scratch);
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(&mut tmp, base2k, cnv_offset_lo, 0, &res_big, base2k, 0, scratch);
            module.vec_znx_sub_assign(&mut tmp, 0, &diag_terms, 0);
            module.vec_znx_sub_assign(&mut tmp, 0, &diag_terms, 1);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_square_apply<BE: Backend>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let n: usize = glwe_infos.n().into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    let group_name = format!("glwe_tensor_square_apply::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_tensor_square_apply(0, &mut tensor, &a, a.max_k().as_usize(), scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
