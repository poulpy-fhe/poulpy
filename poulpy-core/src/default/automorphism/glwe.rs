//! Reference implementations of the [`GLWEAutomorphismDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//! Backends opt in to a method's default by forwarding from their `GLWEAutomorphismDefault`
//! impl: `glwe_automorphism_defaults::glwe_automorphism(self, …)`. A backend that lacks the
//! HAL ops simply doesn't call the helper and provides its own implementation in the trait
//! method body.
//!
//! These items are re-exported publicly through `crate::oep::glwe_automorphism_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxBigAddSmallAssign,
        VecZnxBigAutomorphismAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigSubSmallAssign,
        VecZnxBigSubSmallNegateAssign, VecZnxDftBytesOf, VecZnxIdftApply,
    },
    layouts::{Backend, ScratchArena, VecZnxBigToBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    default::{keyswitching::GLWEKeySwitchInternal, operations::GLWENormalizeDefault},
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, prepared::GGLWEPreparedToBackendRef,
    },
    oep::{GLWEAutomorphismDefault, GLWEKeyswitchDefault},
};

pub fn glwe_automorphism_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE> + VecZnxAutomorphismAssignTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    // The add/sub variants always normalize into a conv buffer before glwe_keyswitch_internal.
    // Their scratch layout is: res_dft | res_conv | max(ks_internal_tmp, res_big + compute).
    // Since glwe_keyswitch_tmp_bytes = dft + max(ks_internal, big + compute), the total is
    // lvl_conv + glwe_keyswitch_tmp_bytes, which also dominates the plain default/assign variants.
    let lvl_conv: usize = if res_infos.max_k() > a_infos.max_k() {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
    } else {
        GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos)
    };
    let lvl_ks: usize = module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
    let lvl_auto: usize = module.vec_znx_automorphism_assign_tmp_bytes();

    lvl_auto.max(lvl_conv + lvl_ks)
}

pub fn glwe_automorphism_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE> + GLWEKeyswitchDefault<BE> + VecZnxAutomorphismAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, a, key)
    );

    module.glwe_keyswitch(res, a, key, key_size, scratch);
    let cols = res.rank().as_usize() + 1;
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_automorphism_assign(key.p(), &mut res_ref.data, i, &mut scratch.borrow());
    }
}

pub fn glwe_automorphism_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE> + GLWEKeyswitchDefault<BE> + VecZnxAutomorphismAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, res, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, res, key)
    );

    module.glwe_keyswitch_assign(res, key, key_size, scratch);

    let cols = res.rank().as_usize() + 1;
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_automorphism_assign(key.p(), &mut res_ref.data, i, &mut scratch.borrow());
    }
}

pub fn glwe_automorphism_add_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, a, key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a_norm.data, 0);

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_add_small_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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

pub fn glwe_automorphism_add_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, res, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, res, key)
    );

    let key_size = key.size().min(key_size);

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize_default(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_norm.data, 0);

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_add_small_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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

pub fn glwe_automorphism_sub_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, a, key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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

pub fn glwe_automorphism_sub_negate_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, a, key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut a_layout = a.glwe_layout();
    a_layout.base2k = key.base2k();
    let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&a_layout);
    module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2);
    let a_norm = a_conv.to_backend_ref();

    {
        let mut scratch = scratch_2;
        module.glwe_keyswitch_internal(&mut res_dft, &a_conv, key, &mut scratch);
        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_negate_assign(&mut res_big, i, &a_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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

pub fn glwe_automorphism_sub_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, res, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, res, key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize_default(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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

pub fn glwe_automorphism_sub_negate_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEAutomorphismDefault<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        scratch.available() >= module.glwe_automorphism_tmp_bytes(res, res, key),
        "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_tmp_bytes(res, res, key)
    );

    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    let mut res_layout = res.glwe_layout();
    res_layout.base2k = key.base2k();
    let (mut res_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&res_layout);
    module.glwe_normalize_default(&mut res_conv, res, &mut scratch_2);
    module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_2);

    {
        let res_norm = res_conv.to_backend_ref();
        let (mut res_big, mut scratch) = scratch_2.borrow().take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
        }

        for i in 0..cols {
            scratch = scratch.apply_mut(|scratch| module.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch));
            module.vec_znx_big_sub_small_negate_assign(&mut res_big, i, &res_norm.data, i);
        }

        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for i in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_ref.data,
                res_base2k,
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
