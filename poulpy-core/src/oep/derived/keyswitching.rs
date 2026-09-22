//! Same-layer compositions that preserve core backend dispatch.

mod gglwe {
    use crate::api::GLWEKeyswitch;

    use poulpy_hal::layouts::{Backend, ScratchArena};

    use crate::layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    };

    pub(crate) fn gglwe_keyswitch_tmp_bytes_derived<BE, M, R, A, K>(
        module: &M,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        BE: Backend,
        M: GLWEKeyswitch<BE>,
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    pub(crate) fn gglwe_keyswitch_derived<BE, M, R, A>(
        module: &M,
        res: &mut R,
        a: &A,
        b: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEKeyswitch<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
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
            scratch.available() >= gglwe_keyswitch_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            gglwe_keyswitch_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, b)
        );

        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                let mut res_at = res.at_view_mut(row, col);
                let a_at = a.at_view(row, col);
                module.glwe_keyswitch(&mut res_at, &a_at, &b.to_backend_ref(), &mut scratch.borrow());
            }
        }
    }

    pub(crate) fn gglwe_keyswitch_assign_derived<BE, M, R>(
        module: &M,
        res: &mut R,
        a: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEKeyswitch<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        let mut res = res.to_backend_mut();

        assert_eq!(
            res.rank_out(),
            a.rank_out(),
            "res output rank: {} != a output rank: {}",
            res.rank_out(),
            a.rank_out()
        );
        assert!(
            scratch.available() >= gglwe_keyswitch_tmp_bytes_derived::<BE, _, _, _, _>(module, &res, &res, a),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            gglwe_keyswitch_tmp_bytes_derived::<BE, _, _, _, _>(module, &res, &res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                let mut res_at = res.at_view_mut(row, col);
                module.glwe_keyswitch_assign(&mut res_at, &a.to_backend_ref(), &mut scratch.borrow());
            }
        }
    }
}
pub(crate) use gglwe::*;

mod ggsw {
    use crate::api::GLWEKeyswitch;

    use poulpy_hal::{
        api::ModuleN,
        layouts::{Backend, ScratchArena},
    };

    use crate::{
        api::GGSWExpandRows,
        layouts::{
            GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, LWEInfos,
            prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
        },
    };

    pub(crate) fn ggsw_keyswitch_tmp_bytes_derived<BE, M, R, A, K, T>(
        module: &M,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        BE: Backend,
        M: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        assert_eq!(key_infos.rank_in(), key_infos.rank_out());
        assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
        assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());
        assert_eq!(module.n() as u32, res_infos.n());
        assert_eq!(module.n() as u32, a_infos.n());
        assert_eq!(module.n() as u32, key_infos.n());
        assert_eq!(module.n() as u32, tsk_infos.n());

        module
            .glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
            .max(module.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ggsw_keyswitch_derived<BE, M, R, A>(
        module: &M,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
    {
        let mut res_backend = res.to_backend_mut();
        let a_backend = a.to_backend_ref();

        assert!(res_backend.dnum() <= a_backend.dnum());
        assert_eq!(res_backend.dsize(), a_backend.dsize());
        assert_eq!(res_backend.base2k(), a_backend.base2k());
        assert!(
            scratch.available()
                >= ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &a_backend, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &a_backend, key, tsk)
        );

        for row in 0..res_backend.dnum().into() {
            let mut res_at = res_backend.at_view_mut(row, 0);
            let a_at = a_backend.at_view(row, 0);
            module.glwe_keyswitch(&mut res_at, &a_at, &key.to_backend_ref(), &mut scratch.borrow());
        }

        module.ggsw_expand_row(&mut res_backend, tsk, scratch)
    }

    pub(crate) fn ggsw_keyswitch_assign_derived<BE, M, R>(
        module: &M,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        let mut res_backend = res.to_backend_mut();

        assert!(
            scratch.available()
                >= ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &res_backend, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &res_backend, key, tsk)
        );

        for row in 0..res_backend.dnum().into() {
            let mut res_at = res_backend.at_view_mut(row, 0);
            module.glwe_keyswitch_assign(&mut res_at, &key.to_backend_ref(), &mut scratch.borrow());
        }

        module.ggsw_expand_row(&mut res_backend, tsk, scratch)
    }
}
pub(crate) use ggsw::*;
