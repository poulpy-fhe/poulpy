pub mod module;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;

pub use module::{FFT64ModuleDefault, NTT4x30ModuleDefault};
pub use vec_znx::HalVecZnxDefault;
pub use vec_znx_big::{FFT64VecZnxBigDefault, NTT4x30VecZnxBigDefault};
pub use vec_znx_dft::{FFT64VecZnxDftDefault, NTT4x30VecZnxDftDefault};

use poulpy_hal::{
    api::HostBufMut,
    layouts::{Backend, ScratchArena},
};

fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: bytemuck::Pod,
{
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = bytemuck::cast_slice_mut(bytes);
    (slice, arena)
}
