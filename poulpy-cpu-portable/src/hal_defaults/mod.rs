pub mod convolution;
pub mod module;
pub mod svp_ppol;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp_pmat;

pub use convolution::{FFT64ConvolutionDefault, NTT4x30ConvolutionDefault};
pub use module::{FFT64ModuleDefault, NTT4x30ModuleDefault};
pub use svp_ppol::{FFT64SvpDefault, NTT4x30SvpDefault};
pub use vec_znx::{BigWordHadamardProduct, HalVecZnxDefault};
pub use vec_znx_big::{FFT64VecZnxBigDefault, NTT4x30VecZnxBigDefault};
pub use vec_znx_dft::{FFT64VecZnxDftDefault, NTT4x30VecZnxDftDefault};
pub use vmp_pmat::{FFT64VmpDefault, NTT4x30VmpDefault};

use poulpy_hal::{
    api::HostBufMut,
    layouts::{Backend, ScratchArena},
};

/// Takes `len` values of `T` from the front of a host scratch arena.
///
/// The region retains its existing scratch bytes. `T` must implement
/// [`bytemuck::Pod`] so every byte pattern is valid and writing a `T` cannot
/// introduce uninitialized padding into the scratch buffer.
///
/// Types with invalid byte patterns, such as `bool`, `char`, and enums, cannot
/// be taken from arbitrary scratch bytes:
///
/// ```compile_fail,E0277
/// # use poulpy_cpu_portable::{FFT64Portable, hal_defaults::take_host_typed};
/// # use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};
/// let mut scratch = ScratchOwned::<FFT64Portable>::alloc(64);
/// let _ = take_host_typed::<FFT64Portable, bool>(scratch.arena(), 1);
/// ```
///
/// ```compile_fail,E0277
/// # use poulpy_cpu_portable::{FFT64Portable, hal_defaults::take_host_typed};
/// # use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};
/// let mut scratch = ScratchOwned::<FFT64Portable>::alloc(64);
/// let _ = take_host_typed::<FFT64Portable, char>(scratch.arena(), 1);
/// ```
///
/// ```compile_fail,E0277
/// # use poulpy_cpu_portable::{FFT64Portable, hal_defaults::take_host_typed};
/// # use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};
/// #[derive(Clone, Copy)]
/// #[repr(u8)]
/// enum Flag { Off, On }
/// let mut scratch = ScratchOwned::<FFT64Portable>::alloc(64);
/// let _ = take_host_typed::<FFT64Portable, Flag>(scratch.arena(), 1);
/// ```
#[inline]
pub fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: bytemuck::Pod,
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
    let bytes: &'a mut [u8] = &mut buf.into_bytes()[..byte_len];
    assert!(
        (bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "scratch region is not aligned to align_of::<T>() = {}",
        std::mem::align_of::<T>()
    );
    // SAFETY: `bytes` exclusively borrows exactly `len * size_of::<T>()` valid
    // bytes, and its non-null pointer has the checked alignment. `Pod` permits
    // every existing byte pattern and forbids uninitialized padding on writes.
    // The raw cast also preserves `len` for zero-sized types such as `()`.
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}
