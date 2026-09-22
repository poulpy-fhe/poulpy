use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`crate::api::Cmux::cmux_tmp_bytes`].
pub fn cmux_tmp_bytes_reference<BE: Backend<ZnxWord = i64> + 'static, R, A, B>(
    module: &Module<BE>,
    res_infos: &R,
    a_infos: &A,
    selector_infos: &B,
) -> usize
where
    R: GLWEInfos,
    A: GLWEInfos,
    B: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let tmp_infos = GLWELayout {
        n: res_infos.n(),
        base2k: res_infos.base2k(),
        k: res_infos.k().max(a_infos.k()),
        rank: res_infos.rank(),
    };
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(&tmp_infos, &tmp_infos, selector_infos);
    let cols: usize = (selector_infos.rank() + 1).into();
    let res_dft: usize = module.bytes_of_vec_znx_dft(module.n(), cols, output_size);
    let res_big: usize = module.bytes_of_vec_znx_big(module.n(), cols, output_size);
    module.glwe_bytes_of_from_infos(res_infos)
        + module
            .glwe_bytes_of_from_infos(a_infos)
            .max(module.glwe_bytes_of_from_infos(&tmp_infos))
        + res_dft
        + res_big
        + module
            .glwe_external_product_internal_tmp_bytes(&tmp_infos, &tmp_infos, selector_infos)
            .max(module.vec_znx_big_normalize_tmp_bytes())
            .max(module.vec_znx_idft_apply_tmp_bytes())
            // Blind selection/rotation also use this bound to copy inputs
            // whose precision or radix need not match their accumulator.
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_copy_tmp_bytes(res_infos, &tmp_infos))
            .max(module.glwe_copy_tmp_bytes(&tmp_infos, res_infos))
            .max(module.glwe_copy_tmp_bytes(res_infos, res_infos))
            .max(module.glwe_copy_tmp_bytes(a_infos, a_infos))
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`crate::api::Cmux::cmux`].
pub fn cmux_reference<'k, BE, R, T, F>(
    module: &Module<BE>,
    res: &mut R,
    t: &T,
    f: &F,
    s: &GGSWPreparedBackendRef<'k, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    T: GLWEToBackendRef<BE>,
    F: GLWEToBackendRef<BE>,
    BE: Backend<ZnxWord = i64> + 'static + 'k,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let f_backend = f.to_backend_ref();

    let scratch = scratch.borrow();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let ggsw_base2k: usize = s.base2k().into();

    module.glwe_sub(res, t, f);
    let cols: usize = (res.rank() + 1).into();
    let (mut tmp_in, mut scratch_2) = scratch.take_glwe_scratch(res);
    module.glwe_copy(&mut tmp_in, res, &mut scratch_2);
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, res, s);
    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
    {
        let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(module.n(), cols, output_size);
        let mut tmp_res_big = tmp_res_big;
        module.glwe_external_product_dft(&mut res_dft, &tmp_in, s, &mut scratch_4.borrow());
        let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
        for col in 0..cols {
            module.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
        }
        (res_big, scratch_norm) = (tmp_res_big, scratch_4);
    }
    let mut res_big = res_big;
    // The false branch is borrowed immutably for the whole call. Reading it
    // directly also keeps workspace independent of its allocation capacity.
    for j in 0..cols {
        module.vec_znx_big_add_small_assign(&mut res_big, j, f_backend.data(), j);
        let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
        module.vec_znx_big_normalize(
            tmp_in.data_mut(),
            res_base2k,
            res_k,
            0,
            j,
            &res_big_ref,
            ggsw_base2k,
            j,
            &mut scratch_norm.borrow(),
        );
    }
    module.glwe_copy(res, &tmp_in, &mut scratch_norm);
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`crate::api::Cmux::cmux_assign_neg`].
pub fn cmux_assign_neg_reference<'k, BE, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    s: &GGSWPreparedBackendRef<'k, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    BE: Backend<ZnxWord = i64> + 'static + 'k,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let a_backend = a.to_backend_ref();

    assert_eq!(res.base2k(), a_backend.base2k());

    let scratch = scratch.borrow();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let ggsw_base2k: usize = s.base2k().into();
    let tmp_infos = GLWELayout {
        n: s.n(),
        base2k: res.base2k(),
        k: res.k().max(a_backend.k()),
        rank: res.rank(),
    };
    let (mut tmp, scratch_1) = scratch.take_glwe_scratch(&tmp_infos);
    let (mut res_prev, mut scratch_2) = scratch_1.take_glwe_scratch(res);
    module.glwe_copy(&mut res_prev, res, &mut scratch_2);
    module.glwe_sub(&mut tmp, a, res);
    let cols: usize = (res.rank() + 1).into();
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(&tmp_infos, &tmp_infos, s);
    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
    {
        let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(module.n(), cols, output_size);
        let mut tmp_res_big = tmp_res_big;
        module.glwe_external_product_dft(&mut res_dft, &tmp, s, &mut scratch_4.borrow());
        let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
        for col in 0..cols {
            module.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
        }
        (res_big, scratch_norm) = (tmp_res_big, scratch_4);
    }
    let mut res_big = res_big;
    let res_prev_ref = vec_znx_backend_ref_from_mut::<BE>(res_prev.data());
    for j in 0..cols {
        module.vec_znx_big_add_small_assign(&mut res_big, j, &res_prev_ref, j);
        let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
        module.vec_znx_big_normalize(
            tmp.data_mut(),
            res_base2k,
            res_k,
            0,
            j,
            &res_big_ref,
            ggsw_base2k,
            j,
            &mut scratch_norm.borrow(),
        );
    }
    module.glwe_copy(res, &tmp, &mut scratch_norm);
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`crate::api::Cmux::cmux_assign`].
pub fn cmux_assign_reference<'k, BE, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    s: &GGSWPreparedBackendRef<'k, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    BE: Backend<ZnxWord = i64> + 'static + 'k,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let a_backend = a.to_backend_ref();
    let scratch = scratch.borrow();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let ggsw_base2k: usize = s.base2k().into();
    module.glwe_sub_assign(res, a);
    let cols: usize = (res.rank() + 1).into();
    let (mut tmp, scratch_1) = scratch.take_glwe_scratch(res);
    let (mut tmp_a, mut scratch_2) = scratch_1.take_glwe_scratch(&a_backend);
    module.glwe_copy(&mut tmp, res, &mut scratch_2);
    module.glwe_copy(&mut tmp_a, a, &mut scratch_2);
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, res, s);
    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
    {
        let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(module.n(), cols, output_size);
        let mut tmp_res_big = tmp_res_big;
        module.glwe_external_product_dft(&mut res_dft, &tmp, s, &mut scratch_4.borrow());
        let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
        for col in 0..cols {
            module.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
        }
        (res_big, scratch_norm) = (tmp_res_big, scratch_4);
    }
    let mut res_big = res_big;
    let tmp_a_ref = vec_znx_backend_ref_from_mut::<BE>(tmp_a.data());
    for j in 0..cols {
        module.vec_znx_big_add_small_assign(&mut res_big, j, &tmp_a_ref, j);
        let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
        module.vec_znx_big_normalize(
            tmp.data_mut(),
            res_base2k,
            res_k,
            0,
            j,
            &res_big_ref,
            ggsw_base2k,
            j,
            &mut scratch_norm.borrow(),
        );
    }
    module.glwe_copy(res, &tmp, &mut scratch_norm);
}
