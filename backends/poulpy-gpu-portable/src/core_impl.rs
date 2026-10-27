//! Registration of the `poulpy-core` extension points.
//!
//! Every family forwards to its reference body, which composes HAL methods,
//! except the two products that have a fused kernel and sampling, which is
//! staged through the host.

use poulpy_core::{
    Distribution, Noise,
    oep::{GGLWEProductDigitsStridedImpl, SamplingImpl},
};
use poulpy_cpu_portable::{NTT4x30Portable, ScalarZnxFill, kernels::noise::add_noise_portable};
use poulpy_hal::{
    AlignedBuf,
    layouts::{
        ConjugateInvariant, DataView, DataViewMut, Module, Ring, ScalarZnx, ScalarZnxBackendMut, ScratchArena, VecZnx,
        VecZnxBackendMut, VecZnxBig, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef, ZnxView,
    },
    source::Source,
};

use crate::{
    NTT4x30Gpu,
    hal_impl::{big_column_mut, column_mut, gglwe_product_digits_strided},
    runtime::{GpuBuf, GpuRuntime, wipe_host},
    znx::{Column, Term, lincomb},
};

unsafe impl<RT: GpuRuntime, R: Ring> GGLWEProductDigitsStridedImpl for NTT4x30Gpu<RT, R> {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        _a_cols: usize,
        _a_size: usize,
        _dsize: usize,
        _pmat_rows: usize,
        _pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        0
    }

    fn gglwe_product_digits_strided(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        gglwe_product_digits_strided(res, a, dsize, product_limbs, pmat);
    }

    ::poulpy_core::impl_gglwe_product_step_hooks_reference!();
}

/// Adds the noise the portable sampler draws on a zeroed host column to
/// `res`. The sample only depends on the seed, the degree, the law, `k` and
/// `base2k`, so drawing it apart from the destination changes nothing; its
/// leading zero limbs are neither transferred nor added. The device copy of
/// the sample is zeroed before it is released.
fn add_host_noise<RT: GpuRuntime, T: bytemuck::Pod + PartialEq + Default>(res: &Column, n: usize, limbs: &[T]) {
    let zero = T::default();
    let Some(first) = limbs.chunks_exact(n).position(|limb| limb.iter().any(|x| *x != zero)) else {
        return;
    };
    let size = limbs.len() / n - first;
    let words = (size_of::<T>() / size_of::<u64>()) as u32;
    let mut device = GpuBuf::<RT>::from_host(bytemuck::cast_slice(&limbs[first * n..]));
    let noise = Column {
        binding: device.view().binding().expect("noise holds storage"),
        base: 0,
        stride: n as u32 * words,
        words,
        n,
        size,
    };
    let target = Column {
        binding: res.binding.clone(),
        base: u32::try_from(first)
            .ok()
            .and_then(|first| first.checked_mul(res.stride))
            .and_then(|skipped| res.base.checked_add(skipped))
            .expect("limb offset beyond the 32-bit word range"),
        stride: res.stride,
        words: res.words,
        n,
        size,
    };
    lincomb::<RT>(&target, size, Term::plus(&target), Term::plus(&noise));
    device.wipe();
}

/// The portable host samplers: the samples are drawn on the host with the
/// streams of `NTT4x30Portable` and reach the device in one transfer.
unsafe impl<RT: GpuRuntime, R: Ring> SamplingImpl for NTT4x30Gpu<RT, R> {
    fn scalar_znx_fill_distribution(
        _module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    ) {
        let n = res.n();
        assert!(res_col < res.cols(), "scalar_znx_fill_distribution: column out of range");
        let mut host = ScalarZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0u8; n * size_of::<i64>()]), n, 1);
        let mut source = Source::new(seed);
        match dist {
            Distribution::TernaryFixed(hw) => host.fill_ternary_hw(0, hw, &mut source),
            Distribution::TernaryProb(prob) => host.fill_ternary_prob(0, prob, &mut source),
            Distribution::BinaryFixed(hw) => host.fill_binary_hw(0, hw, &mut source),
            Distribution::BinaryProb(prob) => host.fill_binary_prob(0, prob, &mut source),
            Distribution::BinaryBlock(block_size) => host.fill_binary_block(0, block_size, &mut source),
            Distribution::ZERO => {}
            Distribution::NONE | Distribution::ENCAPSULATED(_) => {
                panic!("scalar_znx_fill_distribution: {dist:?} is not a sampleable distribution")
            }
        }
        // Host storage is padded; the device column has the exact length.
        let bytes = n * size_of::<i64>();
        res.data_mut()
            .region_mut(res_col * bytes, bytes)
            .write_prefix(&host.data()[..bytes]);
        wipe_host(host.data_mut());
    }

    fn vec_znx_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        let n = res.n();
        assert!(n.is_power_of_two() && n <= module.n(), "noise degree outside module");
        // The checks of the portable sampler, before anything is drawn.
        noise.validate();
        assert!((1..=63).contains(&base2k), "noise radix must be in 1..=63");
        assert!(k > 0, "noise precision must be positive");
        let size = k.div_ceil(base2k);
        assert!(size <= res.size(), "noise precision exceeds destination allocation");
        assert!(res_col < res.cols(), "noise column exceeds destination allocation");
        let mut host = VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0u8; n * size * size_of::<i64>()]), n, 1, size);
        add_noise_portable(base2k, k, &mut host, 0, noise, &mut Source::new(seed));
        if let Some(res) = column_mut(res, res_col) {
            add_host_noise::<RT, i64>(&res, n, host.raw());
        }
        wipe_host(host.data_mut());
    }

    fn vec_znx_big_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        let n = res.n();
        assert!(n.is_power_of_two() && n <= module.n(), "noise degree outside module");
        // The checks of the portable sampler, before anything is drawn.
        noise.validate();
        assert!((1..=63).contains(&base2k), "noise radix must be in 1..=63");
        assert!(k > 0, "noise precision must be positive");
        let size = k.div_ceil(base2k);
        assert!(size <= res.size(), "noise precision exceeds destination allocation");
        assert!(res_col < res.cols(), "noise column exceeds destination allocation");
        let mut host = VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(
            AlignedBuf::from(vec![0u8; n * size * size_of::<i128>()]),
            n,
            1,
            size,
        );
        add_noise_portable(base2k, k, &mut host, 0, noise, &mut Source::new(seed));
        if let Some(res) = big_column_mut(res, res_col) {
            add_host_noise::<RT, i128>(&res, n, host.raw());
        }
        wipe_host(host.data_mut());
    }
}

/// The families registered through the upstream macros, which take a
/// concrete backend type: one invocation per runtime and ring.
macro_rules! impl_core {
    ($be:ty) => {
        ::poulpy_core::impl_automorphism_reference_full!($be);
        ::poulpy_core::impl_decompression_reference_full!($be);
        ::poulpy_core::impl_decryption_reference_full!($be);
        ::poulpy_core::impl_ggsw_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_gglwe_keyswitch_derived_full!($be);
        ::poulpy_core::impl_ggsw_keyswitch_derived_full!($be);
        ::poulpy_core::impl_lwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_encryption_reference_full!($be);
        ::poulpy_core::impl_glwe_external_product_reference_full!($be);
        ::poulpy_core::impl_gglwe_external_product_derived_full!($be);
        ::poulpy_core::impl_ggsw_external_product_derived_full!($be);
        ::poulpy_core::impl_linear_transformation_reference_full!($be);
        ::poulpy_core::impl_operations_reference_full!($be);
        ::poulpy_core::impl_polynomial_evaluation_derived_full!($be);
        ::poulpy_core::impl_preparation_reference_full!($be);
        ::poulpy_core::impl_secret_reference_full!($be);
        ::poulpy_core::impl_glwe_trace_derived_full!($be);
    };
}

/// The families of the standard ring only.
macro_rules! impl_core_standard {
    ($be:ty) => {
        ::poulpy_core::impl_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_pack_derived_full!($be);
        ::poulpy_core::impl_glwe_rotate_reference_full!($be);
        ::poulpy_core::impl_ggsw_rotate_derived_full!($be);
        ::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!($be);
        ::poulpy_core::impl_glwe_ci_conversion_reference_full!($be);
    };
}

/// `GLWETensoringImpl` as `poulpy_core::impl_glwe_tensoring_reference_full!` registers
/// it, every method forwarding to its reference body, except
/// `glwe_mul_columns`, which is the fused three-column product.
macro_rules! impl_tensoring {
    ($be:ty) => {
        unsafe impl ::poulpy_core::oep::GLWETensoringImpl for $be {
            fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, a_size: usize, b_size: usize,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_tmp_bytes_reference(
                    module, res, a, a_size, b_size,
                )
            }

            fn glwe_tensor_apply_prepared_right<R, A, BP>(
                module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut R, a: &A, b: &BP, b_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_reference(
                    module, cnv_offset, res, a, b, b_size, scratch,
                )
            }

            fn glwe_tensor_apply_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                b: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_tmp_bytes_reference(
                    module, res, a,
                )
            }

            fn glwe_tensor_apply<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_reference(
                    module, cnv_offset, res, a, b, scratch,
                )
            }

            fn glwe_tensor_square_apply<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_reference(
                    module, cnv_offset, res, a, scratch,
                )
            }

            fn glwe_tensor_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_reference(
                    module, res, a, tsk, scratch,
                )
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_tmp_bytes_reference(
                    module, res, a, tsk,
                )
            }

            fn glwe_mul_relinearize_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_relinearize<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &B,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_relinearize_assign<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize_assign<R, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize<R, A, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &BP,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, b_size, b_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &AP,
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, tsk, scratch,
                )
            }

            // The three products come from one fused launch instead of the reference's five.
            fn glwe_mul_columns(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                cross: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::CnvPVecLBackendRef<'_, $be>,
                b: &::poulpy_hal::layouts::CnvPVecRBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                let _ = scratch;
                crate::hal_impl::glwe_mul_columns(module, cnv_offset_hi, diag, cross, a, b)
            }

            fn glwe_square_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                left: &mut ::poulpy_hal::layouts::CnvPVecLBackendMut<'_, $be>,
                right: &mut ::poulpy_hal::layouts::CnvPVecRBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_prepare_reference(
                    module, left, right, a, scratch,
                )
            }

            fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            fn glwe_mul_relinearize_sum<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_sum<R, AP, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &[&AP],
                b: &[&BP],
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, a_size, a_k, b_size, b_k, tsk, scratch,
                )
            }

            fn glwe_tensor_apply_prepared_tmp_bytes<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                base2k: ::poulpy_core::layouts::Base2K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_tmp_bytes_reference(
                    module, res, a_size, b_size, base2k,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared<R, AP, BP>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &AP,
                a_size: usize,
                b: &BP,
                b_size: usize,
                base2k: ::poulpy_core::layouts::Base2K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_reference(
                    module, cnv_offset, res, a, a_size, b, b_size, base2k, scratch,
                )
            }
        }
    };
}

#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
mod wgpu {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Wgpu);
    impl_tensoring!(crate::NTT4x30Wgpu);
    impl_core_standard!(crate::NTT4x30Wgpu);
    impl_core!(crate::NTT4x30Wgpu<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Wgpu<ConjugateInvariant>);
}

#[cfg(feature = "enable-cuda")]
mod cuda {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Cuda);
    impl_tensoring!(crate::NTT4x30Cuda);
    impl_core_standard!(crate::NTT4x30Cuda);
    impl_core!(crate::NTT4x30Cuda<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Cuda<ConjugateInvariant>);
}

#[cfg(feature = "enable-hip")]
mod hip {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Hip);
    impl_tensoring!(crate::NTT4x30Hip);
    impl_core_standard!(crate::NTT4x30Hip);
    impl_core!(crate::NTT4x30Hip<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Hip<ConjugateInvariant>);
}
