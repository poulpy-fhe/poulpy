//! Same-layer compositions that preserve core backend dispatch.

mod gglwe {
    use crate::api::GLWEZero;

    use poulpy_hal::layouts::{Backend, ScratchArena};

    use crate::{
        api::GLWEExternalProduct,
        layouts::{
            GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GLWEInfos, GLWEViewMut, prepared::GGSWPreparedBackendRef,
        },
    };

    pub(crate) fn gglwe_external_product_tmp_bytes_derived<BE, M, R, A, B>(
        module: &M,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B,
    ) -> usize
    where
        BE: Backend,
        M: GLWEExternalProduct<BE>,
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        module.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    pub(crate) fn gglwe_external_product_derived<BE, M, R, A>(
        module: &M,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEExternalProduct<BE> + GLWEZero<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
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
            scratch.available() >= gglwe_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            gglwe_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();
        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            for row in 0..min_dnum {
                for col in 0..res_rank_in {
                    let mut res_at = res.at_view_mut(row, col);
                    let a_at = a.at_view(row, col);
                    module.glwe_external_product(&mut res_at, &a_at, b, &mut scratch.borrow());
                }
            }
        }

        if min_dnum < res_dnum {
            let mut res = res.to_backend_mut();
            for row in min_dnum..res_dnum {
                for col in 0..res_rank_in {
                    let mut ct: GLWEViewMut<'_, BE> = res.at_view_mut(row, col);
                    module.glwe_zero(&mut ct);
                }
            }
        }
    }

    pub(crate) fn gglwe_external_product_assign_derived<BE, M, R>(
        module: &M,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEExternalProduct<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        assert_eq!(
            res.rank_out(),
            a.rank(),
            "res output rank: {} != a rank: {}",
            res.rank_out(),
            a.rank()
        );
        assert!(
            scratch.available() >= gglwe_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, res, a),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            gglwe_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();
        let mut res = res.to_backend_mut();
        for row in 0..res_dnum {
            for col in 0..res_rank_in {
                let mut res_at = res.at_view_mut(row, col);
                module.glwe_external_product_assign(&mut res_at, a, &mut scratch.borrow());
            }
        }
    }
}
pub(crate) use gglwe::*;

mod ggsw {
    use crate::api::GLWEZero;

    use poulpy_hal::{
        api::ModuleN,
        layouts::{Backend, ScratchArena},
    };

    use crate::{
        api::GLWEExternalProduct,
        layouts::{
            GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, LWEInfos,
            prepared::GGSWPreparedBackendRef,
        },
    };

    pub(crate) fn ggsw_external_product_tmp_bytes_derived<BE, M, R, A, B>(
        module: &M,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B,
    ) -> usize
    where
        BE: Backend,
        M: GLWEExternalProduct<BE>,
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        module.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    pub(crate) fn ggsw_external_product_derived<BE, M, R, A>(
        module: &M,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEExternalProduct<BE> + GLWEZero<BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    {
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= ggsw_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            ggsw_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();
        for row in 0..min_dnum {
            for col in 0..res_rank {
                let mut res_at = res.at_view_mut(row, col);
                let a_at = a.at_view(row, col);
                module.glwe_external_product(&mut res_at, &a_at, b, &mut scratch.borrow());
            }
        }

        if min_dnum < res_dnum {
            for row in min_dnum..res_dnum {
                for col in 0..res_rank {
                    module.glwe_zero(&mut res.at_view_mut(row, col));
                }
            }
        }
    }

    pub(crate) fn ggsw_external_product_assign_derived<BE, M, R>(
        module: &M,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEExternalProduct<BE> + ModuleN,
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    {
        assert_eq!(res.n(), module.n() as u32);
        assert_eq!(a.n(), module.n() as u32);
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert!(
            scratch.available() >= ggsw_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, res, a),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            ggsw_external_product_tmp_bytes_derived::<BE, _, _, _, _>(module, res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();
        for row in 0..res_dnum {
            for col in 0..res_rank {
                module.glwe_external_product_assign(&mut res.at_view_mut(row, col), a, &mut scratch.borrow());
            }
        }
    }
}
pub(crate) use ggsw::*;
