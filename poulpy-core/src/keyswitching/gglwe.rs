use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch},
};

pub use crate::api::GGLWEKeyswitch;
use crate::{
    ScratchTakeCore,
    keyswitching::GLWEKeyswitch,
    layouts::{GGLWE, GGLWEInfos, GGLWEPreparedToRef, GGLWEToMut, GGLWEToRef, GLWESwitchingKey},
};

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {}

#[doc(hidden)]
pub trait GGLWEKeyswitchDefault<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch_default<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank: {} != a input rank: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank_in(),
            "res output rank: {} != b input rank: {}",
            a.rank_out(),
            b.rank_in()
        );
        assert_eq!(
            res.rank_out(),
            b.rank_out(),
            "res output rank: {} != b output rank: {}",
            res.rank_out(),
            b.rank_out()
        );
        assert!(res.dnum() <= a.dnum(), "res.dnum()={} > a.dnum()={}", res.dnum(), a.dnum());
        assert_eq!(res.dsize(), a.dsize(), "res dsize: {} != a dsize: {}", res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes_default(res, a, b)
        );

        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }
    }

    fn gglwe_keyswitch_assign_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(
            res.rank_out(),
            a.rank_out(),
            "res output rank: {} != a output rank: {}",
            res.rank_out(),
            a.rank_out()
        );
        assert!(
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes_default(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<BE: Backend> GGLWEKeyswitchDefault<BE> for Module<BE> where Self: GLWEKeyswitch<BE> {}
