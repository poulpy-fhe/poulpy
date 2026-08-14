//! Backend extension points for vector-matrix product (VMP) operations.
//!
//! Each flavor trait carries the kernels only: prepare, zero, and the `tmat` /
//! `pmat` `_to_dft` and `_to_dft_accumulate` variants. The derived variants are
//! emitted by [`hal_impl_vmp!`](crate::hal_impl_vmp) inside each backend's
//! `HalVmpImpl` block, so they dispatch back to that backend's kernels.
//!
//! `VmpTMat` and `VmpPMat` currently hold the same bytes on every CPU backend,
//! so the paired `tmat` and `pmat` methods do the same work. They stay separate
//! methods over separate types: a backend that gains a cheaper hot-prep form
//! repoints its `tmat` methods alone, and no caller changes.

use poulpy_hal::{
    api::HostBufMut,
    layouts::{Backend, ScratchArena},
};

#[inline]
pub(super) fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
{
    assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!(
        (bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "scratch region is not aligned to align_of::<T>() = {}",
        std::mem::align_of::<T>()
    );
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

mod pmat;
mod tmat;

pub use pmat::{FFT64VmpPMatDefault, NTT4x30VmpPMatDefault};
pub use tmat::{FFT64VmpTMatDefault, NTT4x30VmpTMatDefault};
