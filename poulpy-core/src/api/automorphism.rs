use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Scratch},
};

use crate::{
    ScratchTakeCore,
    api::GGSWExpandRows,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSW,
        GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos, GLWEToMut, GLWEToRef, GetGaloisElement, SetGaloisElement,
    },
};

pub trait GLWEAutomorphism<BE: Backend> {
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;
}

pub trait GGSWAutomorphism<BE: Backend>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        let lvl_0: usize = self
            .glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos));
        lvl_0
    }

    fn ggsw_automorphism<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut + GGSWInfos,
        A: GGSWToRef + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.dnum() <= a.dnum());
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes(res, a, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes(res, a, key, tsk)
        );

        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism(&mut res.at_mut(row, 0), &a.at(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }

    fn ggsw_automorphism_assign<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes(res, res, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes(res, res, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism_assign(&mut res.at_mut(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}

pub trait GLWEAutomorphismKeyAutomorphism<BE: Backend> {
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;
}
