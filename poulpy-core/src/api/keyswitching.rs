use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    ScratchTakeCore,
    api::{GGSWExpandRows, LWESampleExtract},
    layouts::{
        GGLWE, GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToMut,
        GGSWToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, LWE, LWEInfos, LWEToMut, LWEToRef, Rank, TorusPrecision,
    },
};

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;
}

pub trait GGLWEKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes(res, a, b)
        );

        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }
    }

    fn gglwe_keyswitch_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes(res, res, a),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

pub trait GGSWKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_keyswitch_assign<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

pub trait LWEKeySwitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract + ModuleN,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, key_infos.n());

        let max_k: TorusPrecision = a_infos.max_k().max(res_infos.max_k());

        let glwe_a_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: a_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_res_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: res_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_a_infos);
        let lvl_1: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_res_infos);
        let lvl_2: usize = self.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a: &LWE<&[u8]> = &a.to_ref();

        assert!(res.n().as_usize() <= self.n());
        assert!(a.n().as_usize() <= self.n());
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_keyswitch_tmp_bytes(res, a, ksk),
            "scratch.available(): {} < LWEKeySwitch::lwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.lwe_keyswitch_tmp_bytes(res, a, ksk)
        );

        let (mut glwe_in, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: a.max_k(),
            rank: Rank(1),
        });
        glwe_in.data.zero();

        let n_lwe: usize = a.n().into();

        for i in 0..a.size() {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        let (mut glwe_out, scratch_2) = scratch_1.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: Rank(1),
        });

        self.glwe_keyswitch(&mut glwe_out, &glwe_in, ksk, scratch_2);
        self.lwe_sample_extract(res, &glwe_out);
    }
}
