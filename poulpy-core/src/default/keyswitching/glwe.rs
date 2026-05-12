use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, ScratchAvailable, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy,
        VecZnxDftZero, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedBackendRef, GLWEInfos, GLWEToBackendRef, LWEInfos, prepared::GGLWEPreparedToBackendRef},
};

impl<BE: Backend> GLWEKeySwitchInternal<BE> for Module<BE> where Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE> {}

pub(crate) trait GLWEKeySwitchInternal<BE: Backend>
where
    Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE>,
{
    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let cols: usize = (a_infos.rank() + 1).into();
        let a_size: usize = a_infos.size();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols - 1, a_size);
        let lvl_1: usize = self.gglwe_product_dft_tmp_bytes_default(res_infos.size(), a_size, key_infos);
        lvl_0 + lvl_1
    }

    fn glwe_keyswitch_internal<'s, 'r, A, K>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE> + ScratchAvailable,
        BE: 's,
    {
        let a = a.to_backend_ref();
        let key: GGLWEPreparedBackendRef<'_, BE> = key.to_backend_ref();
        assert_eq!(a.base2k(), key.base2k());
        assert!(
            scratch.available() >= self.glwe_keyswitch_internal_tmp_bytes(&key, &a, &key),
            "scratch.available(): {} < GLWEKeySwitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_keyswitch_internal_tmp_bytes(&key, &a, &key)
        );
        let cols: usize = (a.rank() + 1).into();
        let a_size: usize = a.size();
        scratch.scope(|scratch_phase| {
            let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(self, cols - 1, a_size);
            for col_i in 0..cols - 1 {
                self.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, &a.data, col_i + 1);
            }
            self.gglwe_product_dft_default(res, &a_dft.to_backend_ref(), &key, &mut scratch_1.borrow());
        });
    }
}

impl<BE: Backend> GGLWEProductDefault<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
{
}

pub(crate) trait GGLWEProductDefault<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>,
{
    fn gglwe_product_dft_tmp_bytes_default<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        let dsize: usize = key_infos.dsize().as_usize();

        if dsize == 1 {
            let lvl_0: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                key_infos.dnum().into(),
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );
            lvl_0
        } else {
            let dnum: usize = key_infos.dnum().into();
            let a_size: usize = a_size.div_ceil(dsize).min(dnum);
            let cols_out: usize = (key_infos.rank_out() + 1).into();
            let lvl_0: usize = self.bytes_of_vec_znx_dft(key_infos.rank_in().into(), a_size);
            let lvl_1: usize = self.bytes_of_vec_znx_dft(cols_out, key_infos.size());
            let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                dnum,
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );

            lvl_0 + lvl_1 + lvl_2
        }
    }

    fn gglwe_product_dft_default<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        let cols: usize = a.cols();
        let a_size: usize = a.size();
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key),
            "scratch.available(): {} < GGLWEProductDefault::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key)
        );

        if key.dsize() == 1 {
            self.vmp_apply_dft_to_dft(res, a, &key.data, 0, scratch);
        } else {
            let dsize: usize = key.dsize().into();
            let dnum: usize = key.dnum().into();
            let cols_out: usize = res.cols();

            for di in 0..dsize {
                let (mut ai_dft, mut scratch_1) =
                    scratch
                        .borrow()
                        .take_vec_znx_dft_scratch(self, cols, ((a_size + di) / dsize).min(dnum));

                res.set_size(res.max_size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_copy(dsize, dsize - di - 1, &mut ai_dft, j, a, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(res, &ai_dft.to_backend_ref(), &key.data, 0, &mut scratch_1.borrow());
                } else {
                    let (mut res_dft_tmp, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(self, cols_out, res.size());
                    self.vmp_apply_dft_to_dft(
                        &mut res_dft_tmp,
                        &ai_dft.to_backend_ref(),
                        &key.data,
                        di,
                        &mut scratch_2.borrow(),
                    );
                    for col in 0..cols_out {
                        self.vec_znx_dft_add_assign(res, col, &res_dft_tmp.to_backend_ref(), col);
                    }
                }
            }

            res.set_size(res.max_size());
        }
    }
}

// === Free-function defaults for GLWEKeyswitchDefault ===

use poulpy_hal::{
    api::{
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{VecZnxBigToBackendRef, VecZnxToBackendRef},
};

use crate::{
    default::operations::GLWENormalizeDefault,
    layouts::{GLWE, GLWELayout, GLWEToBackendMut},
    oep::GLWEKeyswitchDefault,
};

fn glwe_keyswitch_dft_fill<'s, 'r, BE, M, A>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    A: GLWEToBackendRef<BE>,
    M: GLWEKeySwitchInternal<BE> + GGLWEProductDefault<BE> + VecZnxDftApply<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE> + ScratchAvailable,
{
    let a = a.to_backend_ref();
    assert_eq!(a.base2k(), key.base2k());
    assert!(
        scratch.available() >= module.glwe_keyswitch_internal_tmp_bytes(key, &a, key),
        "scratch.available(): {} < GLWEKeySwitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_internal_tmp_bytes(key, &a, key)
    );
    let cols: usize = (a.rank() + 1).into();
    let a_size: usize = a.size();
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, cols - 1, a_size);
        for col_i in 0..cols - 1 {
            let a_data: &VecZnxBackendRef<'_, BE> = &a.data;
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a_data, col_i + 1);
        }
        let a_dft_ref = a_dft.to_backend_ref();
        module.gglwe_product_dft_default(res, &a_dft_ref, key, &mut scratch_1.borrow());
    });
}

#[allow(private_bounds)]
pub fn glwe_keyswitch_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let cols: usize = res_infos.rank().as_usize() + 1;
    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols, key_infos.size());
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(cols, key_infos.size());
    let lvl_1: usize = lvl_1_big
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_2: usize = if a_infos.base2k() != key_infos.base2k() {
        let small_term_tmp: usize = poulpy_hal::layouts::VecZnx::<Vec<u8>>::bytes_of(module.n(), 1, key_infos.size());
        let a_conv_infos: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.max_k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes_default()
            .max(module.glwe_keyswitch_internal_tmp_bytes(res_infos, &a_conv_infos, key_infos));
        let lvl_2_2: usize = lvl_1_big
            + small_term_tmp
            + module
                .vec_znx_idft_apply_tmp_bytes()
                .max(module.vec_znx_big_normalize_tmp_bytes())
                .max(module.vec_znx_normalize_tmp_bytes());
        lvl_2_0 + lvl_2_1.max(lvl_2_2)
    } else {
        lvl_1.max(module.glwe_keyswitch_internal_tmp_bytes(res_infos, a_infos, key_infos))
    };

    lvl_0 + lvl_2
}

#[allow(private_bounds)]
pub fn glwe_keyswitch_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEKeyswitchDefault<BE>
        + ModuleN
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(
        a.rank(),
        key.rank_in(),
        "a.rank(): {} != b.rank_in(): {}",
        a.rank(),
        key.rank_in()
    );
    assert_eq!(
        res.rank(),
        key.rank_out(),
        "res.rank(): {} != b.rank_out(): {}",
        res.rank(),
        key.rank_out()
    );

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes(res, a, key)
    );

    let key_size = key.size().min(key_size);

    let a_base2k: usize = a.base2k().into();
    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let key: GGLWEPreparedBackendRef<'_, BE> = key.to_backend_ref();

    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    for col in 0..res_dft.cols() {
        module.vec_znx_dft_zero(&mut res_dft, col);
    }

    let mut scratch = scratch_1;
    if a_base2k != key_base2k {
        scratch.scope(|scratch_phase| {
            let (mut a_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2.borrow());
            glwe_keyswitch_dft_fill(module, &mut res_dft, &a_conv, &key, &mut scratch_2);
        });
    } else {
        glwe_keyswitch_dft_fill(module, &mut res_dft, a, &key, &mut scratch.borrow());
    }

    let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, key_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for i in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch.borrow());
    }
    if a_base2k != key_base2k {
        let (mut res_small, mut scratch_2) = scratch.borrow().take_vec_znx_scratch(module.n(), 1, key_size);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            0,
            0,
            &a.to_backend_ref().data,
            a_base2k,
            0,
            &mut scratch_2.borrow(),
        );
        let res_small_ref = res_small.to_backend_ref();
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_small_ref, 0);
    } else {
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a.to_backend_ref().data, 0);
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

#[allow(private_bounds)]
pub fn glwe_keyswitch_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEKeyswitchDefault<BE>
        + ModuleN
        + GLWEKeySwitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(
        res.rank(),
        key.rank_in(),
        "res.rank(): {} != a.rank_in(): {}",
        res.rank(),
        key.rank_in()
    );
    assert_eq!(
        res.rank(),
        key.rank_out(),
        "res.rank(): {} != b.rank_out(): {}",
        res.rank(),
        key.rank_out()
    );

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes(res, res, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes(res, res, key)
    );

    let key_size = key.size().min(key_size);

    let res_base2k: usize = res.base2k().as_usize();
    let key_base2k: usize = key.base2k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    for col in 0..res_dft.cols() {
        module.vec_znx_dft_zero(&mut res_dft, col);
    }

    let (res_big, mut scratch) = if res_base2k != key_base2k {
        let scratch = scratch_1;
        let (mut res_conv, mut scratch_3) = scratch.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: key.base2k(),
            k: res.max_k(),
            rank: res.rank(),
        });
        module.glwe_normalize_default(&mut res_conv, res, &mut scratch_3.borrow());

        module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_3);

        let (mut res_big, mut scratch) = scratch_3.take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        let (mut res_small, mut scratch_2) = scratch.take_vec_znx_scratch(module.n(), 1, key_size);
        let res_ref = GLWEToBackendRef::<BE>::to_backend_ref(res);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            0,
            0,
            &res_ref.data,
            res_base2k,
            0,
            &mut scratch_2.borrow(),
        );
        let res_small_ref = res_small.to_backend_ref();
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_small_ref, 0);
        (res_big, scratch_2)
    } else {
        {
            let mut ks_scratch = scratch_1.borrow();
            module.glwe_keyswitch_internal(&mut res_dft, res, key, &mut ks_scratch);
        }
        let res_ref = GLWEToBackendRef::<BE>::to_backend_ref(res);
        let (mut res_big, mut scratch) = scratch_1.take_vec_znx_big_scratch(module, cols, key_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_ref.data, 0);
        (res_big, scratch)
    };
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
