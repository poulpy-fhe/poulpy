//! Portable algorithms expressed with HAL operations.
//! Inter-family core operations dispatch through the selected backend hooks.

#![allow(private_bounds)]
use crate::api::{GLWEKeyswitch, GLWENormalize};

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxBigAddSmallAssign,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallAssign, VecZnxBigSubSmallNegateAssign, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxIdftNormalizeConsumeTmpBytes, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, ScratchArena, VecZnxBigToBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        prepared::{GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
    oep::GLWEAutomorphismReference,
    reference::keyswitching::{GLWEKeyswitchInternal, gglwe_product_output_size},
};

pub fn glwe_automorphism_tmp_bytes_reference<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GLWEKeyswitch<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxBigAutomorphismAssignTmpBytes
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeConsumeTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    // The plain and assign variants call the dispatched keyswitch, then rotate the
    // destination in place; the two stages do not overlap.
    let lvl_plain: usize = module
        .glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
        .max(module.vec_znx_automorphism_assign_tmp_bytes());

    // The accumulating variants never call glwe_keyswitch: they normalize into a
    // conv buffer at the key radix and drive glwe_keyswitch_internal themselves, so
    // their scratch is sized here rather than read off any keyswitch implementation.
    // Layout: res_dft | a_conv | max(normalize, ks_internal, res_big + compute).
    let cols: usize = res_infos.rank().as_usize() + 1;
    let mask_cols: usize = a_infos.rank().as_usize();
    let mut a_conv_infos: GLWELayout = GLWELayout {
        n: a_infos.n(),
        base2k: key_infos.base2k(),
        k: a_infos.k(),
        rank: a_infos.rank(),
    };
    // The product window is read at the input precision; the conv buffer is then
    // widened to its whole allocation, exactly as the bodies do.
    let output_size: usize = gglwe_product_output_size::<BE, _, _, _>(res_infos, &a_conv_infos, key_infos);
    a_conv_infos.k = a_conv_infos.max_k();
    let a_dft_size: usize = a_conv_infos.size();

    let lvl_dft: usize = module.bytes_of_vec_znx_dft(module.n(), cols, output_size);
    let lvl_conv: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
    let lvl_big: usize = module.bytes_of_vec_znx_big(module.n(), cols, output_size)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_automorphism_assign_tmp_bytes())
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_accumulate: usize = lvl_dft
        + lvl_conv
        + module
            .glwe_normalize_tmp_bytes()
            .max(module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, output_size, a_dft_size, key_infos))
            .max(lvl_big);

    lvl_plain.max(lvl_accumulate)
}

pub fn glwe_automorphism_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphismReference<BE> + GLWEKeyswitch<BE> + VecZnxAutomorphismAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, a, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, a, &key)
    );

    module.glwe_keyswitch(res, a, &key, scratch);
    let cols = res.rank().as_usize() + 1;
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_automorphism_assign(p, &mut res_ref.data, i, &mut scratch.borrow());
    }
}

pub fn glwe_automorphism_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphismReference<BE> + GLWEKeyswitch<BE> + VecZnxAutomorphismAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, res, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, res, &key)
    );

    module.glwe_keyswitch_assign(res, &key, scratch);

    let cols = res.rank().as_usize() + 1;
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_automorphism_assign(p, &mut res_ref.data, i, &mut scratch.borrow());
    }
}

pub fn glwe_automorphism_add_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, a, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, a, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &a_layout, &key);
    a_layout.k = a_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, &key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a_norm.data, 0);

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_add_small_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_automorphism_add_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, res, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, res, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &res_layout, &key);
    res_layout.k = res_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, &key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_norm.data, 0);

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_add_small_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_automorphism_sub_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, a, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, a, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &a_layout, &key);
    a_layout.k = a_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, &key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_automorphism_sub_negate_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, a, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, a, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &a_layout, &key);
    a_layout.k = a_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, &key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_negate_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_automorphism_sub_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, res, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, res, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &res_layout, &key);
    res_layout.k = res_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, &key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_automorphism_sub_negate_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismReference<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let p = key.p();
    let key = key.to_backend_ref();
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes_reference(res, res, &key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes_reference(res, res, &key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, &res_layout, &key);
    res_layout.k = res_layout.max_k();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, output_size);
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, &key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module.n(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(p, &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_negate_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &res_big_ref,
                key_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
    }
}
