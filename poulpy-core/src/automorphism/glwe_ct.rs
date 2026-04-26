use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchTakeBasic, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxBigAddSmallAssign,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigNormalize, VecZnxBigSubSmallAssign,
        VecZnxBigSubSmallNegateAssign, VecZnxNormalize,
    },
    layouts::{Backend, Module, Scratch, VecZnxBig},
};

pub use crate::api::GLWEAutomorphism;
use crate::{
    GLWEKeySwitchInternal, GLWEKeyswitch, GLWENormalize, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos},
};

pub(crate) trait GLWEAutomorphismDefault<BE: Backend>:
    Sized
    + GLWEKeyswitch<BE>
    + GLWEKeySwitchInternal<BE>
    + VecZnxNormalize<BE>
    + VecZnxAutomorphismAssign<BE>
    + VecZnxAutomorphismAssignTmpBytes
    + VecZnxBigAutomorphismAssign<BE>
    + VecZnxBigAutomorphismAssignTmpBytes
    + VecZnxBigSubSmallAssign<BE>
    + VecZnxBigSubSmallNegateAssign<BE>
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxBigNormalize<BE>
    + GLWENormalize<BE>
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
        let lvl_1: usize = self
            .vec_znx_automorphism_assign_tmp_bytes()
            .max(self.vec_znx_big_automorphism_assign_tmp_bytes());

        lvl_0.max(lvl_1)
    }

    fn glwe_automorphism_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        self.glwe_keyswitch(res, a, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_assign(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        self.glwe_keyswitch_assign(res, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_assign(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_add_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_add_small_assign(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_add_small_assign(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_add_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_add_small_assign(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_add_small_assign(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_assign(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_assign(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_negate_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_negate_assign(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_negate_assign(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_assign(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_assign(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_negate_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_negate_assign(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_assign(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_negate_assign(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }
}

impl<BE: Backend> GLWEAutomorphismDefault<BE> for Module<BE>
where
    Self: Sized
        + GLWEKeyswitch<BE>
        + GLWEKeySwitchInternal<BE>
        + VecZnxNormalize<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
