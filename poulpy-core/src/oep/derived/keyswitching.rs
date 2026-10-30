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
        assert_eq!(
            res.k_digit(),
            a.k_digit(),
            "res k_digit: {} != a k_digit: {}",
            res.k_digit(),
            a.k_digit()
        );
        assert_eq!(
            res.k_offset(),
            a.k_offset(),
            "res k_offset: {} != a k_offset: {}",
            res.k_offset(),
            a.k_offset()
        );
        assert_eq!(res.base2k(), a.base2k());
        crate::layouts::assert_no_row_offset("gglwe_keyswitch", &b);
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
    use crate::layouts::operand_degree;

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
        operand_degree(module.n(), &[res_infos.n(), a_infos.n(), key_infos.n(), tsk_infos.n()]);

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
        assert_eq!(res_backend.k_digit(), a_backend.k_digit());
        assert_eq!(res_backend.base2k(), a_backend.base2k());
        assert!(
            scratch.available()
                >= ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &a_backend, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            ggsw_keyswitch_tmp_bytes_derived::<BE, _, _, _, _, _>(module, &res_backend, &a_backend, key, tsk)
        );

        crate::layouts::assert_no_row_offset("ggsw_keyswitch", &key);
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

mod glwe_dft {
    use poulpy_hal::{
        api::{ScratchArenaTakeVecZnxDft, VecZnxDftBytesOf},
        layouts::{Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftToBackendRef},
    };

    use crate::{
        layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendRef, prepared::GGLWEPreparedBackendRef},
        oep::GLWEKeyswitchImpl,
    };

    /// Scratch of [`glwe_keyswitch_dft_add_derived`]: the product, carved at an
    /// aligned offset, then the scratch of the product itself.
    pub(crate) fn glwe_keyswitch_dft_add_tmp_bytes_derived<BE, A, K>(
        module: &Module<BE>,
        res_size: usize,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        BE: GLWEKeyswitchImpl,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let product = module.vec_znx_dft_bytes_of(module.n(), (key_infos.rank_out() + 1).as_usize(), res_size);
        BE::scratch_aligned(product) + BE::glwe_keyswitch_dft_tmp_bytes(module, res_size, a_infos, key_infos)
    }

    /// `res += Σ_i decomp(a_i) · key_i`: the product of
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_dft`] into a temporary, added to `res`.
    pub(crate) fn glwe_keyswitch_dft_add_derived<BE, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: GLWEKeyswitchImpl,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let tmp_bytes = glwe_keyswitch_dft_add_tmp_bytes_derived(module, res.size(), a, key);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_dft_add_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        let cols = res.cols();
        let (mut product, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(res.n(), cols, res.size());
        BE::glwe_keyswitch_dft(module, &mut product, a, key, &mut scratch_1);
        let product = product.to_backend_ref();
        for col in 0..cols {
            BE::vec_znx_dft_add_assign(module, res, col, &product, col);
        }
    }
}
pub(crate) use glwe_dft::*;

mod gglwe_product {
    use poulpy_hal::layouts::{
        Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
    };

    use crate::{
        layouts::{assert_gglwe_product_idft_normalize, prepared::GGLWEPreparedBackendRef},
        oep::GGLWEProductDigitsStridedImpl,
    };

    /// [`GGLWEProductDigitsStridedImpl::gglwe_product_idft_normalize`] on the items in order,
    /// every item and the scratch checked before the first write.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn gglwe_products_idft_normalize_derived<BE: GGLWEProductDigitsStridedImpl>(
        module: &Module<BE>,
        res: &mut [VecZnxBackendMut<'_, BE>],
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut [VecZnxDftBackendMut<'_, BE>],
        a: &[VecZnxDftBackendRef<'_, BE>],
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        addends: &[&[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>]],
        body: &[Option<(&VecZnxBackendRef<'_, BE>, usize)>],
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        assert!(
            res_dft.len() == res.len()
                && a.len() == res.len()
                && (addends.is_empty() || addends.len() == res.len())
                && body.len() == res.len(),
            "gglwe_products_idft_normalize: items of different lengths"
        );
        let addend = |i: usize| addends.get(i).copied().unwrap_or(&[]);
        let mut need = 0;
        for i in 0..res.len() {
            assert_gglwe_product_idft_normalize(
                "gglwe_products_idft_normalize",
                &res[i],
                &res_dft[i],
                &a[i],
                key,
                addend(i),
                body[i],
            );
            need = need.max(BE::gglwe_product_idft_normalize_tmp_bytes(
                module,
                res[i].size(),
                res_dft[i].size(),
                a[i].size(),
                key,
            ));
        }
        assert!(
            scratch.available() >= need,
            "gglwe_products_idft_normalize: scratch below its query"
        );
        for (i, (res, res_dft)) in res.iter_mut().zip(res_dft.iter_mut()).enumerate() {
            BE::gglwe_product_idft_normalize(
                module,
                res,
                res_base2k,
                res_k,
                res_offset,
                res_dft,
                &a[i],
                key,
                term_count,
                addend(i),
                body[i],
                a_base2k,
                scratch,
            );
        }
    }
}
pub(crate) use gglwe_product::*;
