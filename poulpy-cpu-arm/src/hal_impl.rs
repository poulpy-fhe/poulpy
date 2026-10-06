#[allow(unused_imports)]
use std::mem::size_of;

use crate::{FFT64Neon, NTT4x30Neon};
use poulpy_cpu_portable::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ConvolutionDefault, NTT4x30ModuleDefault, NTT4x30SvpDefault, NTT4x30VecZnxBigDefault,
    NTT4x30VecZnxDftDefault, NTT4x30VmpDefault,
};
use poulpy_cpu_portable::kernels::{
    fft64::{
        module::{FFT64Plan, FFT64PlanNew},
        ring_arith::Fft64RingArith,
    },
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        vec_znx_dft::{NttPlan, NttPlanNew},
    },
    znx::ZnxAutomorphism,
};
use poulpy_hal::layouts::Ring;
#[allow(unused_imports)]
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, MatZnxBackendRef, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    assert!(BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()));
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!((bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()));
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64Neon {
    poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64Neon {
    poulpy_cpu_portable::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for FFT64Neon<R>
where
    Self: ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!();
    poulpy_cpu_portable::hal_impl_vec_znx_normalize!();
}

unsafe impl<R: Ring> HalModuleImpl for FFT64Neon<R>
where
    FFT64Plan<f64, R>: FFT64PlanNew,
{
    poulpy_cpu_portable::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl<R: Ring> HalVmpImpl for FFT64Neon<R>
where
    Self: Fft64RingArith + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl<R: Ring> HalConvolutionImpl for FFT64Neon<R>
where
    Self: Fft64RingArith + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl<R: Ring> HalVecZnxBigImpl for FFT64Neon<R>
where
    Self: ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl<R: Ring> HalSvpImpl for FFT64Neon<R>
where
    Self: Fft64RingArith + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl<R: Ring> HalVecZnxDftImpl for FFT64Neon<R>
where
    Self: Fft64RingArith + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30Neon {
    poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30Neon {
    poulpy_cpu_portable::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!();
    poulpy_cpu_portable::hal_impl_vec_znx_normalize!();
}

unsafe impl<R: Ring> HalModuleImpl for NTT4x30Neon<R>
where
    NttPlan<Primes30, R>: NttPlanNew,
{
    poulpy_cpu_portable::hal_impl_module!(NTT4x30ModuleDefault);
}

#[cfg(target_arch = "aarch64")]
unsafe impl<R: Ring> HalVmpImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        crate::ntt4x30::vmp::vmp_prepare_tmp_bytes_neon(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_tmp_bytes_neon(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_neon_pm(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_neon::<_, poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
    }

    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_add_neon::<_, poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        crate::ntt4x30::vmp::vmp_extract_selected_rows_neon_pm(res, a, first_row, row_step)
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as NTT4x30VmpDefault>::vmp_zero_default(module, res)
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl<R: Ring> HalVmpImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vmp!(NTT4x30VmpDefault);
}

unsafe impl<R: Ring> HalConvolutionImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_convolution!(NTT4x30ConvolutionDefault);
}

unsafe impl<R: Ring> HalVecZnxBigImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl<R: Ring> HalSvpImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_svp!(NTT4x30SvpDefault);
}

unsafe impl<R: Ring> HalVecZnxDftImpl for NTT4x30Neon<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
