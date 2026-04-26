use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, DataMut, Scratch, VecZnxBig, VecZnxDft, ZnxZero},
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GGSW, GGSWInfos, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

pub trait GLWEExternalProduct<BE: Backend> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_assign<R, D>(&self, res: &mut R, a: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_external_product<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

pub trait GLWEExternalProductInternal<BE: Backend> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_internal<DR, A, G>(
        &self,
        res_dft: VecZnxDft<DR, BE>,
        a: &A,
        ggsw: &G,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        G: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable;
}

pub trait GGLWEExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        let lvl_0: usize = self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos);
        lvl_0
    }

    fn gglwe_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank_in: {} != a input rank_in: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank(),
            "a output rank_out: {} != b rank: {}",
            a.rank_out(),
            b.rank()
        );
        assert_eq!(
            res.rank_out(),
            b.rank(),
            "res output rank_out: {} != b rank: {}",
            res.rank_out(),
            b.rank()
        );
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes(res, a, b)
        );

        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();
        let b: &GGSWPrepared<&[u8], BE> = &b.to_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_external_product(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }

        for row in res.dnum().min(a.dnum()).into()..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                res.at_mut(row, col).data_mut().zero();
            }
        }
    }

    fn gglwe_external_product_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(
            res.rank_out(),
            a.rank(),
            "res output rank: {} != a rank: {}",
            res.rank_out(),
            a.rank()
        );
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes(res, res, a),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_external_product_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

pub trait GGSWExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + ModuleN,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        let lvl_0: usize = self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos);
        lvl_0
    }

    fn ggsw_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::GGSWToMut,
        A: crate::layouts::GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let b: &GGSWPrepared<&[u8], BE> = &b.to_ref();

        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
        assert_eq!(res.base2k(), a.base2k());

        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes(res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();

        for row in 0..min_dnum {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_external_product(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }

        for row in min_dnum..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                res.at_mut(row, col).data.zero();
            }
        }
    }

    fn ggsw_external_product_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes(res, res, a),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_external_product_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}
