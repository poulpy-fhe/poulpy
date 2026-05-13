//! Reference implementations of the [`ConversionDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//!
//! Re-exported publicly through `crate::oep::conversion_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyRangeBackend, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxZeroBackend,
    },
    layouts::{
        Backend, ScratchArena, VecZnx, VecZnxBackendRef, VecZnxBigToBackendRef, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VecZnxToBackendRef,
    },
};

use crate::{
    GLWERotate, ScratchArenaTakeCore,
    default::{keyswitching::GGLWEProductDefault, operations::GLWECopyDefault},
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut,
        GLWEToBackendRef, GLWEViewMut, GLWEViewRef, LWEInfos, LWEToBackendMut, LWEToBackendRef, Rank, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{ConversionDefault, GLWEKeyswitchDefault},
};

pub fn lwe_sample_extract_default<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    assert!(res.n() <= a.n());
    assert_eq!(a.n(), module.n() as u32);
    assert!(res.base2k() == a.base2k());

    let min_size: usize = res.size().min(a.size());
    let n: usize = res.n().into();

    module.vec_znx_zero_backend(&mut res.data, 0);
    (0..min_size).for_each(|i| {
        module.vec_znx_copy_range_backend(&mut res.data, 0, i, 0, &a.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut res.data, 0, i, 1, &a.data, 1, i, 0, n);
    });
}

pub fn glwe_from_lwe_tmp_bytes_default<BE, M, R, A, K>(module: &M, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE> + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of(
        module.n().into(),
        key_infos.base2k(),
        lwe_infos.max_k().max(glwe_infos.max_k()),
        1u32.into(),
    );

    let lvl_1_ks: usize = module.glwe_keyswitch_tmp_bytes_default(glwe_infos, glwe_infos, key_infos);
    let lvl_1_a_conv: usize = if lwe_infos.base2k() == key_infos.base2k() {
        0
    } else {
        VecZnx::bytes_of(module.n(), 1, lwe_infos.size()) + module.vec_znx_normalize_tmp_bytes()
    };

    let lvl_1: usize = lvl_1_ks.max(lvl_1_a_conv);

    lvl_0 + lvl_1
}

pub fn glwe_from_lwe_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    lwe: &A,
    ksk: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ConversionDefault<BE>
        + ModuleN
        + GLWEKeyswitchDefault<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: LWEToBackendRef<BE> + LWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let res_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: res.rank(),
    };
    let lwe = lwe.to_backend_ref();

    assert_eq!(res_infos.n.as_u32(), module.n() as u32);
    assert_eq!(ksk.n(), module.n() as u32);
    assert!(lwe.n() <= module.n() as u32);
    assert!(
        scratch.available() >= module.glwe_from_lwe_tmp_bytes_default(&res_infos, &lwe, ksk),
        "scratch.available(): {} < GLWEFromLWE::glwe_from_lwe_tmp_bytes: {}",
        scratch.available(),
        module.glwe_from_lwe_tmp_bytes_default(&res_infos, &lwe, ksk)
    );

    let scratch = scratch.borrow();

    let (mut glwe, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: ksk.base2k(),
        k: lwe.max_k(),
        rank: 1u32.into(),
    });
    module.vec_znx_zero_backend(&mut glwe.data, 0);
    module.vec_znx_zero_backend(&mut glwe.data, 1);

    let n_lwe: usize = lwe.n().into();

    let mut scratch_1 = if lwe.base2k() == ksk.base2k() {
        for i in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut glwe.data, 0, i, 0, &lwe.data, 0, i, 0, 1);
            module.vec_znx_copy_range_backend(&mut glwe.data, 1, i, 0, &lwe.data, 0, i, 1, n_lwe);
        }
        scratch_1
    } else {
        let (mut a_conv, mut scratch_2) = scratch_1.take_vec_znx_scratch(module.n(), 1, lwe.size());
        module.vec_znx_zero_backend(&mut a_conv, 0);
        for j in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut a_conv, 0, j, 0, &lwe.data, 0, j, 0, 1);
        }

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            0,
            0,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );

        module.vec_znx_zero_backend(&mut a_conv, 0);
        for j in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut a_conv, 0, j, 0, &lwe.data, 0, j, 1, n_lwe);
        }

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            0,
            1,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );

        scratch_2
    };

    let mut res_backend = res.to_backend_mut();
    let glwe_ref = glwe_backend_ref_from_mut::<BE>(&glwe);
    let glwe_view = &glwe_ref;
    let mut res_view = &mut res_backend;
    module.glwe_keyswitch_default(&mut res_view, &glwe_view, ksk, key_size, &mut scratch_1)
}

pub fn lwe_from_glwe_tmp_bytes_default<BE, M, R, A, K>(module: &M, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE>,
    R: LWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let res_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: lwe_infos.base2k(),
        k: lwe_infos.max_k(),
        rank: Rank(1),
    };

    let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of(module.n().into(), lwe_infos.base2k(), lwe_infos.max_k(), 1u32.into());
    let lvl_1: usize = module.glwe_keyswitch_tmp_bytes_default(&res_infos, glwe_infos, key_infos);
    let lvl_2: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos);

    lvl_0 + lvl_1 + lvl_2
}

pub fn lwe_from_glwe_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    a_idx: usize,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ConversionDefault<BE>
        + ModuleN
        + GLWEKeyswitchDefault<BE>
        + GLWERotate<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let a_backend = a.to_backend_ref();

    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);
    assert!(res.n() <= module.n() as u32);
    assert!(
        scratch.available() >= module.lwe_from_glwe_tmp_bytes_default(res, a, key),
        "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
        scratch.available(),
        module.lwe_from_glwe_tmp_bytes_default(res, a, key)
    );

    let glwe_layout: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: Rank(1),
    };

    let scratch = scratch.borrow();
    let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.take_glwe_scratch(&glwe_layout);

    let a_backend_view = &a_backend;
    module.glwe_keyswitch_default(&mut tmp_glwe_rank_1, &a_backend_view, key, key_size, &mut scratch_1);
    if a_idx != 0 {
        module.glwe_rotate_assign(-(a_idx as i64), &mut tmp_glwe_rank_1, &mut scratch_1);
    }

    let mut res_backend = res.to_backend_mut();
    let tmp_glwe_rank_1_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
    let min_size: usize = res_backend.size().min(tmp_glwe_rank_1_ref.size());
    let n: usize = res_backend.n().into();

    module.vec_znx_zero_backend(&mut res_backend.data, 0);
    for i in 0..min_size {
        module.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 0, &tmp_glwe_rank_1_ref.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 1, &tmp_glwe_rank_1_ref.data, 1, i, 0, n);
    }
}

pub fn ggsw_from_gglwe_tmp_bytes_default<BE, M, R, A>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
where
    BE: Backend,
    M: ConversionDefault<BE>,
    R: GGSWInfos,
    A: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    module.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos)
}

pub fn ggsw_from_gglwe_default<'s, BE, M, R, A, T>(
    module: &M,
    res: &mut R,
    a: &A,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ConversionDefault<BE> + ModuleN + GLWECopyDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGLWEToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut res_backend = res.to_backend_mut();
    let a_backend = a.to_backend_ref();

    assert_eq!(res_backend.rank(), a_backend.rank_out());
    assert_eq!(res_backend.dnum(), a_backend.dnum());
    assert_eq!(res_backend.n(), module.n() as u32);
    assert_eq!(a_backend.n(), module.n() as u32);
    assert_eq!(tsk.n(), module.n() as u32);
    assert_eq!(res_backend.base2k(), a_backend.base2k());
    assert!(
        scratch.available() >= module.ggsw_from_gglwe_tmp_bytes_default(&res_backend, tsk),
        "scratch.available(): {} < GGSWFromGGLWE::ggsw_from_gglwe_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_from_gglwe_tmp_bytes_default(&res_backend, tsk)
    );

    for row in 0..res_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        let a_at = a_backend.at_view(row, 0);
        module.glwe_copy_default(&mut res_at, &a_at);
    }

    module.ggsw_expand_row_default(&mut res_backend, tsk, tsk_size, scratch)
}

pub fn ggsw_expand_rows_tmp_bytes_default<BE, M, R, A>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GGSWInfos,
    A: GGLWEInfos,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, tsk_infos.n());

    let tsk_base2k: usize = tsk_infos.base2k().into();

    let rank: usize = res_infos.rank().into();
    let cols: usize = rank + 1;

    let res_size: usize = res_infos.size();
    let a_size: usize = res_infos.max_k().as_usize().div_ceil(tsk_base2k);

    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols - 1, a_size) + VecZnx::bytes_of(module.n(), 1, a_size);
    let lvl_1_res_dft: usize = module.bytes_of_vec_znx_dft(cols, a_size);
    let lvl_1_gglwe_prod: usize = module.gglwe_product_dft_tmp_bytes_default(res_size, a_size, tsk_infos);
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(cols, res_size)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_1: usize = lvl_1_res_dft + lvl_1_gglwe_prod.max(lvl_1_big);
    let lvl_2: usize = module.vec_znx_normalize_tmp_bytes();

    lvl_0 + lvl_1.max(lvl_2)
}

pub fn ggsw_expand_row_default<'s, BE, M, R, T>(
    module: &M,
    res: &mut R,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ConversionDefault<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut res_backend = res.to_backend_mut();

    let res_base2k: usize = res_backend.base2k().into();
    let tsk_base2k: usize = tsk.base2k().into();

    assert!(
        scratch.available() >= module.ggsw_expand_rows_tmp_bytes_default(&res_backend, tsk),
        "scratch.available(): {} < GGSWExpandRows::ggsw_expand_rows_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_expand_rows_tmp_bytes_default(&res_backend, tsk)
    );

    let rank: usize = res_backend.rank().into();
    let cols: usize = rank + 1;

    let res_conv_size: usize = res_backend.max_k().as_usize().div_ceil(tsk_base2k);
    {
        let (mut a_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols - 1, res_conv_size);
        let (mut a_0, mut scratch_2) = scratch_1.take_vec_znx_scratch(module.n(), 1, res_conv_size);

        for row in 0..res_backend.dnum().as_usize() {
            {
                let glwe_mi_1: GLWEViewRef<'_, BE> = res_backend.at_view(row, 0);

                for i in 0..cols - 1 {
                    module.vec_znx_normalize(
                        &mut a_0,
                        tsk_base2k,
                        0,
                        0,
                        &glwe_mi_1.data,
                        res_base2k,
                        i + 1,
                        &mut scratch_2.borrow(),
                    );
                    let a_0_ref: VecZnxBackendRef<'_, BE> = a_0.to_backend_ref();
                    module.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_0_ref, 0);
                }
                module.vec_znx_normalize(
                    &mut a_0,
                    tsk_base2k,
                    0,
                    0,
                    &glwe_mi_1.data,
                    res_base2k,
                    0,
                    &mut scratch_2.borrow(),
                );
            }

            let a_0_ref: VecZnxBackendRef<'_, BE> = a_0.to_backend_ref();
            let a_dft_ref: VecZnxDftBackendRef<'_, BE> = a_dft.to_backend_ref();
            let mut scratch_row = scratch_2.borrow();
            ggsw_expand_rows_internal(
                module,
                row,
                &mut res_backend,
                &a_0_ref,
                &a_dft_ref,
                tsk,
                tsk_size,
                &mut scratch_row,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ggsw_expand_rows_internal<'a, 'b, R, M, T, BE: Backend>(
    module: &M,
    row: usize,
    res: &mut R,
    a_0: &VecZnxBackendRef<'a, BE>,
    a_dft: &VecZnxDftBackendRef<'b, BE>,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GGLWEProductDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GGSWAtViewMut<BE> + GGSWInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    let tsk: GGLWEToGGSWKeyPreparedBackendRef<'_, BE> = tsk.to_backend_ref();
    let cols: usize = res.rank().as_usize() + 1;

    for col in 1..cols {
        let scratch_row = scratch.borrow();
        let (mut res_dft, mut scratch_1) = scratch_row.take_vec_znx_dft_scratch(module, cols, tsk_size);
        for j in 0..cols {
            module.vec_znx_dft_zero(&mut res_dft, j);
        }

        {
            let mut scratch_prod = scratch_1.borrow();
            module.gglwe_product_dft_default(&mut res_dft, a_dft, tsk.at(col - 1), &mut scratch_prod);
        }

        let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, cols, res_dft.size());
        let res_dft_ref = res_dft.to_backend_ref();
        for j in 0..cols {
            scratch_2 = scratch_2.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, j, &res_dft_ref, j, scratch));
        }

        module.vec_znx_big_add_small_assign(&mut res_big, col, a_0, 0);
        let res_big_ref = res_big.to_backend_ref();

        let res_base2k: usize = res.base2k().as_usize();

        for j in 0..cols {
            let mut res_col: GLWEViewMut<'_, _> = res.at_view_mut(row, col);
            let scratch_norm = &mut scratch_2.borrow();
            module.vec_znx_big_normalize(
                &mut res_col.data,
                res_base2k,
                0,
                j,
                &res_big_ref,
                tsk.base2k().as_usize(),
                j,
                scratch_norm,
            );
        }
    }
}
