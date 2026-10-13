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
#[inline]
pub fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
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
