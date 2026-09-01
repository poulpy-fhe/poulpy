use std::mem::size_of;

#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
use crate::NTT3x42IfmaRayon;
use crate::NTT4x30Avx512;
#[cfg(feature = "enable-rayon")]
use crate::NTT4x30Avx512Rayon;
use poulpy_ckks::{
    CKKSCtBounds, CKKSMeta, CKKSResult, SetCKKSInfos,
    api::CKKSPow2Ops,
    default::bootstrapping::{ckks_encapsulated_mod_up_default, ckks_encapsulated_mod_up_tmp_bytes_default},
    oep::CKKSEncapsulatedModUpImpl,
};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    default::keyswitching::glwe::gglwe_product_output_size,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
    oep::GGLWEProductDigitsStridedImpl,
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApply,
    },
    execution::SerialTaskExecutor,
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VmpPMatBackendRef,
    },
};

trait ModUpBackend: Backend {
    #[allow(clippy::too_many_arguments)]
    fn product_known_zero_prefix(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

impl ModUpBackend for NTT4x30Avx512 {
    fn product_known_zero_prefix(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = <Self as GGLWEProductDigitsStridedImpl<Self>>::gglwe_product_digits_strided_tmp_bytes(
            module,
            res.size(),
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_digits_strided_avx_known_zero_prefix::<SerialTaskExecutor>(
            module,
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            zero_prefix,
            tmp,
        );
    }
}

#[cfg(feature = "enable-rayon")]
impl ModUpBackend for NTT4x30Avx512Rayon {
    fn product_known_zero_prefix(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::ntt4x30_avx512::vmp_apply_digits_strided_known_zero_prefix(
            module,
            res,
            a,
            dsize,
            zero_prefix,
            product_limbs,
            pmat,
            scratch,
        );
    }
}

#[cfg(feature = "enable-ifma")]
impl ModUpBackend for NTT3x42Ifma {
    fn product_known_zero_prefix(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            1,
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_digits_strided_ifma_known_zero_prefix::<SerialTaskExecutor>(
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            zero_prefix,
            tmp,
        );
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl ModUpBackend for NTT3x42IfmaRayon {
    fn product_known_zero_prefix(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::ntt3x42_ifma::vmp_apply_digits_strided_known_zero_prefix(
            module,
            res,
            a,
            dsize,
            zero_prefix,
            product_limbs,
            pmat,
            scratch,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encapsulated_mod_up<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: ModUpBackend + CKKSEncapsulatedModUpImpl<BE>,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSPow2Ops<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let k_large = dst.k().as_usize();
    let k_small = src.k().as_usize();
    if dst.base2k() != sparse_to_dense.base2k() || sparse_to_dense.dsize().as_usize() < 2 || k_large < k_small + scale_up {
        return ckks_encapsulated_mod_up_default(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch);
    }

    module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    let shift = k_large - k_small - scale_up;
    module.glwe_copy(dst, src);
    module.glwe_rsh(shift, dst, scratch);
    dst.set_meta(CKKSMeta {
        log_delta: src.log_delta() + scale_up,
        log_sparsity: src.log_sparsity(),
        slots: src.slots(),
    });
    dst.set_k(k_large.into());
    let zero_prefix = shift / dst.base2k().as_usize();

    let key = sparse_to_dense.to_backend_ref();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(dst, dst, &key);
    let output_cols = dst.rank().as_usize() + 1;
    let mask_cols = dst.rank().as_usize();
    let dsize = key.dsize().as_usize();
    let product_terms = key
        .n()
        .as_usize()
        .saturating_mul(key.dnum().as_usize())
        .saturating_mul(dsize)
        .saturating_mul(mask_cols.max(1));
    let accumulation_bits = if product_terms <= 1 {
        0
    } else {
        usize::BITS as usize - (product_terms - 1).leading_zeros() as usize
    };
    let base2k = key.base2k().as_usize();
    let product_limbs = base2k.saturating_mul(2).saturating_add(accumulation_bits).div_ceil(base2k);
    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, output_cols, output_size);

    {
        let dst_ref = dst.to_backend_ref();
        let a_size = dst_ref.size();
        let (mut a_dft, mut product_scratch) = scratch_1.borrow().take_vec_znx_dft_scratch(module, mask_cols, a_size);
        for col in 0..mask_cols {
            let mut suffix = a_dft.with_limb_range_mut(zero_prefix, a_size);
            module.vec_znx_dft_apply(1, zero_prefix, &mut suffix, col, dst_ref.data(), col + 1);
        }
        BE::product_known_zero_prefix(
            module,
            &mut res_dft,
            &a_dft.to_backend_ref(),
            dsize,
            zero_prefix,
            product_limbs,
            key.data(),
            &mut product_scratch,
        );
    }

    let dst_ref = dst.to_backend_ref();
    let (mut res_big, mut normalize_scratch) = scratch_1.take_vec_znx_big_scratch(module, output_cols, output_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..output_cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut normalize_scratch);
    }
    module.vec_znx_big_add_small_assign(&mut res_big, 0, dst_ref.data(), 0);
    drop(dst_ref);

    let res_big_ref = res_big.to_backend_ref();
    let mut dst_ref = dst.to_backend_mut();
    let base2k = dst_ref.base2k().as_usize();
    for col in 0..output_cols {
        module.vec_znx_big_normalize(
            dst_ref.data_mut(),
            base2k,
            0,
            col,
            &res_big_ref,
            base2k,
            col,
            &mut normalize_scratch.borrow(),
        );
    }
    Ok(())
}

macro_rules! impl_encapsulated_mod_up {
    ($be:ty) => {
        unsafe impl CKKSEncapsulatedModUpImpl<$be> for $be {
            fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
                module: &Module<$be>,
                dst_infos: &Dst,
                src_infos: &Src,
                dense_to_sparse_infos: &D2S,
                sparse_to_dense_infos: &S2D,
            ) -> usize
            where
                Dst: CKKSCtBounds,
                Src: CKKSCtBounds,
                D2S: GGLWEInfos,
                S2D: GGLWEInfos,
            {
                ckks_encapsulated_mod_up_tmp_bytes_default(
                    module,
                    dst_infos,
                    src_infos,
                    dense_to_sparse_infos,
                    sparse_to_dense_infos,
                )
            }

            fn ckks_encapsulated_mod_up<Dst, Src>(
                module: &Module<$be>,
                dst: &mut Dst,
                src: &mut Src,
                scale_up: usize,
                dense_to_sparse: &GGLWEPreparedBackendRef<'_, $be>,
                sparse_to_dense: &GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ScratchArena<'_, $be>,
            ) -> CKKSResult<()>
            where
                Dst: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + CKKSCtBounds + SetCKKSInfos,
                Src: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + CKKSCtBounds + SetCKKSInfos,
            {
                encapsulated_mod_up(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)
            }
        }
    };
}

impl_encapsulated_mod_up!(NTT4x30Avx512);
#[cfg(feature = "enable-rayon")]
impl_encapsulated_mod_up!(NTT4x30Avx512Rayon);
#[cfg(feature = "enable-ifma")]
impl_encapsulated_mod_up!(NTT3x42Ifma);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_encapsulated_mod_up!(NTT3x42IfmaRayon);
