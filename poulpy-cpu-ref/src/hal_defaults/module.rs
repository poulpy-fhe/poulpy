//! Backend defaults for [`Module`] construction.

use std::ptr::NonNull;

use crate::reference::{fft64::module::FFT64HandleFactory, ntt4x30::vec_znx_dft::NttHandleFactory};
use poulpy_hal::layouts::{Backend, Module};

#[doc(hidden)]
pub trait FFT64ModuleDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn module_new_default(n: u64) -> Module<Self>
    where
        Self::Handle: FFT64HandleFactory,
    {
        <Self::Handle as FFT64HandleFactory>::assert_fft64_runtime_support();
        let handle = <Self::Handle as FFT64HandleFactory>::create_fft64_handle(n as usize);
        let ptr: NonNull<Self::Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64ModuleDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30ModuleDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn module_new_default(n: u64) -> Module<Self>
    where
        Self::Handle: NttHandleFactory,
    {
        <Self::Handle as NttHandleFactory>::assert_ntt_runtime_support();
        let handle = <Self::Handle as NttHandleFactory>::create_ntt_handle(n as usize);
        let ptr: NonNull<Self::Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30ModuleDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
