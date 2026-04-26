use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, Scratch},
};

pub use crate::api::GGSWAutomorphism;
use crate::{
    GGSWExpandRows, ScratchTakeCore,
    automorphism::glwe_ct::GLWEAutomorphism,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedToRef, GGSW, GGSWInfos, GGSWToMut,
        GGSWToRef, GetGaloisElement,
    },
};

#[doc(hidden)]
pub trait GGSWAutomorphismDefault<BE: Backend>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_automorphism_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    fn ggsw_automorphism_default<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk)
        );

        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism(&mut res.at_mut(row, 0), &a.at(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }

    fn ggsw_automorphism_assign_default<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes_default(res, res, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes_default(res, res, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism_assign(&mut res.at_mut(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}

impl<BE: Backend> GGSWAutomorphismDefault<BE> for Module<BE> where Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE> {}
