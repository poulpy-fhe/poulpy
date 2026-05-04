//! Backend defaults for [`Module`] construction.

use std::ptr::NonNull;

use crate::reference::{fft64::module::FFT64HandleFactory, ntt120::vec_znx_dft::NttHandleFactory};
use poulpy_hal::layouts::{Backend, Module};

#[doc(hidden)]
pub trait FFT64ModuleDefaults<BE: Backend>: Backend {
    fn module_new_default(n: u64) -> Module<BE>
    where
        BE::Handle: FFT64HandleFactory,
    {
        <BE::Handle as FFT64HandleFactory>::assert_fft64_runtime_support();
        let handle = <BE::Handle as FFT64HandleFactory>::create_fft64_handle(n as usize);
        let ptr: NonNull<BE::Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

impl<BE: Backend> FFT64ModuleDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120ModuleDefaults<BE: Backend>: Backend {
    fn module_new_default(n: u64) -> Module<BE>
    where
        BE::Handle: NttHandleFactory,
    {
        <BE::Handle as NttHandleFactory>::assert_ntt_runtime_support();
        let handle = <BE::Handle as NttHandleFactory>::create_ntt_handle(n as usize);
        let ptr: NonNull<BE::Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

impl<BE: Backend> NTT120ModuleDefaults<BE> for BE {}
