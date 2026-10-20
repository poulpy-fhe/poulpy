use super::NTT4x30Avx;
#[cfg(feature = "enable-rayon")]
use super::NTT4x30AvxRayon;
use poulpy_ckks::Scale;

use std::mem::size_of;

use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, GLWEPlaintextMeta, SetCKKSInfos,
    api::CKKSPow2Ops,
    oep::CKKSEncapsulatedModUpImpl,
    reference::bootstrapping::{ckks_encapsulated_mod_up_reference, ckks_encapsulated_mod_up_tmp_bytes_reference},
};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, digits_aligned,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
    oep::GGLWEProductDigitsStridedImpl,
    reference::keyswitching::glwe::{
        gadget_input_size, gadget_known_zero_prefix, gglwe_product_limbs, gglwe_product_output_size, precision_limbs,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxGadgetDecompose, VecZnxIdftApply,
    },
    execution::SerialTaskExecutor,
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VecZnxToBackendRef, VmpPMatBackendRef,
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

impl ModUpBackend for NTT4x30Avx {
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
        let bytes = <Self as GGLWEProductDigitsStridedImpl>::gglwe_product_digits_strided_tmp_bytes(
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
        super::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx_known_zero_prefix::<_, SerialTaskExecutor>(
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
impl ModUpBackend for NTT4x30AvxRayon {
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
        super::ntt4x30::vmp_apply_digits_strided_known_zero_prefix(
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
    BE: ModUpBackend + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSPow2Ops<BE>
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
    let k_large = dst.k().as_usize();
    let k_small = src.k().as_usize();
    // A strided key view is gathered by the reference body only.
    if dst.base2k() != sparse_to_dense.base2k()
        || sparse_to_dense.stride() != 1
        || sparse_to_dense.dsize().as_usize() < 2
        || k_large < k_small + scale_up
    {
        return ckks_encapsulated_mod_up_reference(module, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch);
    }

    module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    let shift = k_large - k_small - scale_up;
    module.glwe_copy(dst, src, scratch);
    module.glwe_rsh(shift, dst, scratch);
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
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
        // The prefix the raise leaves at zero is not transformed and the product does not read it.
        let (base2k, digit_k, dnum) = (key.base2k(), key.digit_k(), key.dnum());
        let operand_size = gadget_input_size(base2k, digit_k, dnum, dst_ref.k(), dst_ref.size());
        let zero_prefix = gadget_known_zero_prefix(base2k, digit_k, dnum, shift).min(operand_size);
        let (mut a_dft, mut product_scratch) = scratch_1
            .borrow()
            .take_vec_znx_dft_scratch(module.n(), mask_cols, operand_size);
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
    let (mut res_big, mut normalize_scratch) = scratch_1.take_vec_znx_big_scratch(module.n(), output_cols, output_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..output_cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut normalize_scratch);
    }
    module.vec_znx_big_add_small_assign(&mut res_big, 0, dst_ref.data(), 0);
    drop(dst_ref);

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

macro_rules! impl_encapsulated_mod_up {
    ($be:ty) => {
        unsafe impl CKKSEncapsulatedModUpImpl for $be {
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
                ckks_encapsulated_mod_up_tmp_bytes_reference(
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

impl_encapsulated_mod_up!(NTT4x30Avx);
#[cfg(feature = "enable-rayon")]
impl_encapsulated_mod_up!(NTT4x30AvxRayon);
