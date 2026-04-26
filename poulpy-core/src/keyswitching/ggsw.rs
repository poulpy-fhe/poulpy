use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, Module, Scratch},
};

pub use crate::api::GGSWKeyswitch;
use crate::{
    GGSWExpandRows, ScratchTakeCore,
    keyswitching::GLWEKeyswitch,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGSW, GGSWInfos, GGSWToMut, GGSWToRef, LWEInfos},
};

#[doc(hidden)]
pub trait GGSWKeyswitchDefault<BE: Backend>
where
    Self: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        assert_eq!(key_infos.rank_in(), key_infos.rank_out());
        assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
        assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());
        assert_eq!(self.n() as u32, tsk_infos.n());

        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    fn ggsw_keyswitch_assign_default<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.ggsw_keyswitch_tmp_bytes_default(res, res, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_keyswitch_tmp_bytes_default(res, res, key, tsk)
        );

        for row in 0..res.dnum().into() {
            self.glwe_keyswitch_assign(&mut res.at_mut(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }

    fn ggsw_keyswitch_default<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.ggsw_keyswitch_tmp_bytes_default(res, a, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_keyswitch_tmp_bytes_default(res, a, key, tsk)
        );

        for row in 0..a.dnum().into() {
            self.glwe_keyswitch(&mut res.at_mut(row, 0), &a.at(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}

impl<BE: Backend> GGSWKeyswitchDefault<BE> for Module<BE> where Self: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE> {}
