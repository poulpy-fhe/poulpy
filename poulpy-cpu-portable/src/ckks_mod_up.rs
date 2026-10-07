//! Encapsulated ModUp of [`NTT4x30Portable`] with the limbs that the raise leaves at zero skipped.
//!
//! After the switch to the sparse key and the shift to the large modulus, the top `shift / base2k` limbs of the
//! ciphertext are zero. The switch back to the dense key transforms only the limbs below them, and its product
//! is told how many rows to skip instead of scanning for them.

use poulpy_ckks::{
    CKKSCtBounds, CKKSMeta, CKKSResult, SetCKKSInfos,
    oep::CKKSEncapsulatedModUpImpl,
    reference::bootstrapping::{ckks_encapsulated_mod_up_reference, ckks_encapsulated_mod_up_tmp_bytes_reference},
};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
    reference::keyswitching::glwe::gglwe_product_output_size,
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    execution::SerialTaskExecutor,
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDftToBackendRef},
};

use crate::NTT4x30Portable;
use crate::ntt4x30::{STRIDED_MAX_DSIZE, gglwe_product_digits_strided, gglwe_product_digits_strided_tmp_bytes};

type BE = NTT4x30Portable;

/// Whether the raise from `src` to `dst` takes the path of this module, and falls back to the reference otherwise.
fn takes_native_path<Dst, Src, S2D>(dst: &Dst, src: &Src, scale_up: Option<usize>, sparse_to_dense: &S2D) -> bool
where
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    S2D: GGLWEInfos,
{
    let dsize = sparse_to_dense.dsize().as_usize();
    dst.base2k() == sparse_to_dense.base2k()
        && (2..=STRIDED_MAX_DSIZE).contains(&dsize)
        && scale_up.is_none_or(|scale_up| dst.k().as_usize() >= src.k().as_usize() + scale_up)
}

/// Scratch of the switch back to the dense key: the product, then its inverse transform and normalization.
fn dense_switch_tmp_bytes<Dst, S2D>(module: &Module<BE>, dst: &Dst, sparse_to_dense: &S2D) -> usize
where
    Dst: CKKSCtBounds,
    S2D: GGLWEInfos,
{
    let n = module.n();
    let (mask_cols, output_cols) = (dst.rank().as_usize(), dst.rank().as_usize() + 1);
    let output_size = gglwe_product_output_size::<BE, _, _, _>(dst, dst, sparse_to_dense);
    let res_dft = BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, output_cols, output_size));
    let product = BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, mask_cols, dst.size()))
        + gglwe_product_digits_strided_tmp_bytes(mask_cols, dst.size());
    let normalize = BE::scratch_aligned(module.bytes_of_vec_znx_big(n, output_cols, output_size))
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    res_dft + product.max(normalize)
}

unsafe impl CKKSEncapsulatedModUpImpl for BE {
    fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
        module: &Module<BE>,
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
        let reference = ckks_encapsulated_mod_up_tmp_bytes_reference(
            module,
            dst_infos,
            src_infos,
            dense_to_sparse_infos,
            sparse_to_dense_infos,
        );
        // The scale is not known here: the size covers the native path whenever the layouts allow it.
        if takes_native_path(dst_infos, src_infos, None, sparse_to_dense_infos) {
            reference.max(dense_switch_tmp_bytes(module, dst_infos, sparse_to_dense_infos))
        } else {
            reference
        }
    }

    fn ckks_encapsulated_mod_up<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        if !takes_native_path(dst, src, Some(scale_up), sparse_to_dense) {
            return ckks_encapsulated_mod_up_reference(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch);
        }
        let k_large = dst.k().as_usize();
        let k_small = src.k().as_usize();

        module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
        let shift = k_large - k_small - scale_up;
        module.glwe_copy(dst, src, scratch);
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
        let (mut res_dft, mut scratch_1) = scratch
            .borrow()
            .take_vec_znx_dft_scratch(module.n(), output_cols, output_size);

        {
            let dst_ref = dst.to_backend_ref();
            let a_size = dst_ref.size();
            let zero_prefix = zero_prefix.min(a_size);
            let (mut a_dft, mut product_scratch) = scratch_1.borrow().take_vec_znx_dft_scratch(module.n(), mask_cols, a_size);
            // The limbs of the prefix are left as the scratch holds them: the product does not read them.
            for col in 0..mask_cols {
                let mut suffix = a_dft.with_limb_range_mut(zero_prefix, a_size);
                module.vec_znx_dft_apply(1, zero_prefix, &mut suffix, col, dst_ref.data(), col + 1);
            }
            gglwe_product_digits_strided::<_, SerialTaskExecutor>(
                &mut res_dft,
                &a_dft.to_backend_ref(),
                dsize,
                product_limbs,
                key.data(),
                Some(zero_prefix),
                &mut product_scratch,
            );
        }

        let (mut res_big, mut normalize_scratch) = scratch_1.take_vec_znx_big_scratch(module.n(), output_cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for col in 0..output_cols {
            module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut normalize_scratch);
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, dst.to_backend_ref().data(), 0);

        let res_big_ref = res_big.to_backend_ref();
        let mut dst_ref = dst.to_backend_mut();
        let base2k = dst_ref.base2k().as_usize();
        let k = dst_ref.k().as_usize();
        for col in 0..output_cols {
            module.vec_znx_big_normalize(
                dst_ref.data_mut(),
                base2k,
                k,
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
}
