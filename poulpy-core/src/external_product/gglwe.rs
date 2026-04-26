use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, Scratch, ZnxZero},
};

pub use crate::api::GGLWEExternalProduct;
use crate::{
    GLWEExternalProduct, ScratchTakeCore,
    layouts::{GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWPrepared, GLWEInfos, prepared::GGSWPreparedToRef},
};

#[doc(hidden)]
pub trait GGLWEExternalProductDefault<BE: Backend>
where
    Self: GLWEExternalProduct<BE>,
{
    fn gglwe_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product_default<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.gglwe_external_product_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes_default(res, a, b)
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

    fn gglwe_external_product_assign_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.gglwe_external_product_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes_default(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_external_product_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<BE: Backend> GGLWEExternalProductDefault<BE> for Module<BE> where Self: GLWEExternalProduct<BE> {}

// module-only API: external product is provided by `GGLWEExternalProduct` on `Module`.
