use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, Module, Scratch, ZnxZero},
};

pub use crate::api::GGSWExternalProduct;
use crate::{
    GLWEExternalProduct, ScratchTakeCore,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GGSWExternalProductDefault<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + ModuleN,
{
    fn ggsw_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product_default<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
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
            scratch.available() >= self.ggsw_external_product_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes_default(res, a, b)
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

    fn ggsw_external_product_assign_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes_default(res, res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_external_product_assign(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<BE: Backend> GGSWExternalProductDefault<BE> for Module<BE> where Self: GLWEExternalProduct<BE> + ModuleN {}
