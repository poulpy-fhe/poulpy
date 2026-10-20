//! Encapsulated ModUp of the packed NTT4x30 backends with the limbs that the raise leaves at zero skipped.
//!
//! After the switch to the sparse key and the shift to the large modulus, the top `shift` bits of the ciphertext
//! are zero. The switch back to the dense key transforms only the operand limbs below the ones this leaves at
//! zero, [`gadget_known_zero_prefix`], and its product is told how many rows to skip instead of scanning for
//! them. For digits that are whole limbs the operand is the ciphertext, otherwise its layout on the digit grid.

use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    oep::CKKSEncapsulatedModUpImpl,
    reference::bootstrapping::{ckks_encapsulated_mod_up_reference, ckks_encapsulated_mod_up_tmp_bytes_reference},
};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, digits_aligned,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
    reference::keyswitching::glwe::{
        gadget_dft_apply_tmp_bytes, gadget_input_size, gadget_known_zero_prefix, gglwe_product_limbs, gglwe_product_output_size,
        precision_limbs,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes,
    },
    execution::SerialTaskExecutor,
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VecZnxToBackendRef, VmpPMatBackendRef,
    },
};

use crate::NTT4x30Portable;
use crate::ntt4x30::{STRIDED_MAX_DSIZE, gglwe_product_digits_strided, gglwe_product_digits_strided_tmp_bytes};

/// The fused interleaved-digit product of a backend, as the ModUp of this module uses it.
///
/// Not a stable API: it lets the Rayon variant of the portable backend share the body below.
pub trait PackedModUp: Backend {
    /// Scratch (in bytes) of [`Self::product_known_zero_prefix`].
    fn product_tmp_bytes(a_cols: usize, a_size: usize) -> usize;

    /// The interleaved-digit product of `a` by `pmat`, for at most [`STRIDED_MAX_DSIZE`] digits, where the
    /// first `zero_prefix` limbs of `a` are known to be zero and are not read.
    #[allow(clippy::too_many_arguments)]
    fn product_known_zero_prefix(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

impl PackedModUp for NTT4x30Portable {
    fn product_tmp_bytes(a_cols: usize, a_size: usize) -> usize {
        gglwe_product_digits_strided_tmp_bytes(a_cols, a_size)
    }

    fn product_known_zero_prefix(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        zero_prefix: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        gglwe_product_digits_strided::<_, SerialTaskExecutor>(res, a, dsize, product_limbs, pmat, Some(zero_prefix), scratch);
    }
}

/// Whether the raise from `src` to `dst` takes the path of this module, and falls back to the reference otherwise.
fn takes_native_path<Dst, Src, S2D>(dst: &Dst, src: &Src, scale_up: Option<usize>, sparse_to_dense: &S2D) -> bool
where
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    S2D: GGLWEInfos,
{
    let dsize = sparse_to_dense.dsize().as_usize();
    // A strided key view is gathered by the reference body only.
    dst.base2k() == sparse_to_dense.base2k()
        && sparse_to_dense.stride() == 1
        && (2..=STRIDED_MAX_DSIZE).contains(&dsize)
        && scale_up.is_none_or(|scale_up| dst.k().as_usize() >= src.k().as_usize() + scale_up)
}

/// Scratch of the switch back to the dense key: the product, then its inverse transform and normalization.
fn dense_switch_tmp_bytes<BE, Dst, S2D>(module: &Module<BE>, dst: &Dst, sparse_to_dense: &S2D) -> usize
where
    BE: PackedModUp,
    Module<BE>: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxGadgetDecomposeTmpBytes,
    Dst: CKKSCtBounds,
    S2D: GGLWEInfos,
{
    let n = module.n();
    let (mask_cols, output_cols) = (dst.rank().as_usize(), dst.rank().as_usize() + 1);
    let output_size = gglwe_product_output_size::<BE, _, _, _>(dst, dst, sparse_to_dense);
    let res_dft = BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, output_cols, output_size));
    // Off-limb digits are laid out in a one-column buffer before their transform.
    let (base2k, digit_k, dnum) = (sparse_to_dense.base2k(), sparse_to_dense.digit_k(), sparse_to_dense.dnum());
    let operand_size = gadget_input_size(base2k, digit_k, dnum, dst.k(), dst.size());
    let layout = gadget_dft_apply_tmp_bytes::<BE, _>(module, base2k, digit_k, dnum, dst.k(), dst.size());
    let product = BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, mask_cols, operand_size))
        + layout.max(BE::product_tmp_bytes(mask_cols, operand_size));
    let normalize = BE::scratch_aligned(module.bytes_of_vec_znx_big(n, output_cols, output_size))
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    res_dft + product.max(normalize)
}

/// Scratch (in bytes) of [`encapsulated_mod_up`].
pub fn encapsulated_mod_up_tmp_bytes<BE, Dst, Src, D2S, S2D>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    dense_to_sparse_infos: &D2S,
    sparse_to_dense_infos: &S2D,
) -> usize
where
    BE: PackedModUp,
    Module<BE>: GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxGadgetDecomposeTmpBytes,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    S2D: GGLWEInfos,
{
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
}

/// The encapsulated ModUp, with the zero limbs of the raised ciphertext skipped where the layouts allow it.
pub fn encapsulated_mod_up<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: PackedModUp + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + poulpy_ckks::api::CKKSPow2Ops<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxGadgetDecompose<BE>
        + VecZnxIdftApply<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
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
        dst.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(src.log_delta() + scale_up),
            log_sparsity: src.log_sparsity(),
            slots: src.slots(),
        });
        dst.set_k(k_large.into());

        let key = sparse_to_dense.to_backend_ref();
        let output_size = gglwe_product_output_size::<BE, _, _, _>(dst, dst, &key);
        let output_cols = dst.rank().as_usize() + 1;
        let mask_cols = dst.rank().as_usize();
        let dsize = key.dsize().as_usize();
        let product_limbs = gglwe_product_limbs(&key, 1);
        let (mut res_dft, mut scratch_1) = scratch
            .borrow()
            .take_vec_znx_dft_scratch(module.n(), output_cols, output_size);

        {
            let dst_ref = dst.to_backend_ref();
            let (base2k, digit_k, dnum) = (key.base2k(), key.digit_k(), key.dnum());
            let operand_size = gadget_input_size(base2k, digit_k, dnum, dst_ref.k(), dst_ref.size());
            let zero_prefix = gadget_known_zero_prefix(base2k, digit_k, dnum, shift).min(operand_size);
            let (mut a_dft, mut product_scratch) =
                scratch_1
                    .borrow()
                    .take_vec_znx_dft_scratch(module.n(), mask_cols, operand_size);
            // The limbs of the prefix are left as the scratch holds them: the product does not read them.
            if digits_aligned(base2k, digit_k) {
                for col in 0..mask_cols {
                    let mut suffix = a_dft.with_limb_range_mut(zero_prefix, operand_size);
                    module.vec_znx_dft_apply(1, zero_prefix, &mut suffix, col, dst_ref.data(), col + 1);
                }
            } else {
                let (mut digits, mut layout_scratch) = product_scratch.borrow().take_vec_znx_scratch(module.n(), 1, operand_size);
                let dst_live =
                    precision_limbs::<BE>(dst_ref.data(), base2k, dst_ref.k()).expect("the raised ciphertext has a precision");
                for col in 0..mask_cols {
                    module.vec_znx_gadget_decompose(
                        &mut digits,
                        0,
                        &dst_live,
                        col + 1,
                        base2k.as_usize(),
                        digit_k.as_usize(),
                        &mut layout_scratch.borrow(),
                    );
                    let mut suffix = a_dft.with_limb_range_mut(zero_prefix, operand_size);
                    module.vec_znx_dft_apply(1, zero_prefix, &mut suffix, col, &digits.to_backend_ref(), 0);
                }
            }
            BE::product_known_zero_prefix(
                &mut res_dft,
                &a_dft.to_backend_ref(),
                dsize,
                zero_prefix,
                product_limbs,
                key.data(),
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

unsafe impl CKKSEncapsulatedModUpImpl for NTT4x30Portable {
    fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
        module: &Module<Self>,
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
        encapsulated_mod_up_tmp_bytes(module, dst_infos, src_infos, dense_to_sparse_infos, sparse_to_dense_infos)
    }

    fn ckks_encapsulated_mod_up<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, Self>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        encapsulated_mod_up(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)
    }
}
