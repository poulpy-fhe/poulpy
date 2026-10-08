//! Device buffers and their borrowed views.
//!
//! An owned buffer is one CubeCL allocation. A view is a byte range of an
//! allocation; kernels bind the whole allocation and address the range through
//! an element offset, so two views of the same allocation can be read and
//! written by one kernel.

use std::marker::PhantomData;

use cubecl::{bytes::Bytes, prelude::*, server::Handle};
use poulpy_hal::layouts::{CopyFromHost, CopyToHost};

use super::{Binding, GpuRuntime, cubes, launched};

/// Device allocations are a whole number of 64-bit words.
const WORD: usize = 8;

#[cube(launch_unchecked)]
fn fill_zero(out: &mut [u32], base: u32, len: u32) {
    if ABSOLUTE_POS < len as usize {
        out[base as usize + ABSOLUTE_POS] = 0u32;
    }
}

#[derive(Clone)]
struct Allocation {
    /// Identity of the allocation, for alias checks.
    id: u64,
    handle: Handle,
    /// Bytes reserved on the device, a multiple of [`WORD`].
    reserved: usize,
}

fn allocate<RT: GpuRuntime>(len: usize) -> Option<Allocation> {
    if len == 0 {
        return None;
    }
    let reserved = len.div_ceil(WORD) * WORD;
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    Some(Allocation {
        id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        handle: RT::client().empty(reserved),
        reserved,
    })
}

fn read<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, dst: &mut [u8]) {
    let Some(allocation) = allocation else {
        assert!(dst.is_empty(), "read of {} bytes from an empty device buffer", dst.len());
        return;
    };
    if dst.is_empty() {
        return;
    }
    let end = offset + dst.len();
    assert!(end <= allocation.reserved, "device read out of bounds");
    let handle = allocation
        .handle
        .clone()
        .offset_start(offset as u64)
        .offset_end((allocation.reserved - end) as u64);
    let bytes = RT::client()
        .read_one(handle)
        .unwrap_or_else(|error| panic!("device read failed: {error:?}"));
    dst.copy_from_slice(&bytes[..dst.len()]);
}

fn write<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, src: &[u8]) {
    let Some(allocation) = allocation else {
        assert!(src.is_empty(), "write of {} bytes to an empty device buffer", src.len());
        return;
    };
    if src.is_empty() {
        return;
    }
    let end = offset + src.len();
    assert!(end <= allocation.reserved, "device write out of bounds");
    let handle = allocation
        .handle
        .clone()
        .offset_start(offset as u64)
        .offset_end((allocation.reserved - end) as u64);
    RT::client().write(&handle, Bytes::from_bytes_vec(src.to_vec()));
}

fn zero<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, len: usize) {
    let Some(allocation) = allocation else { return };
    if len == 0 {
        return;
    }
    assert!(
        offset.is_multiple_of(4) && len.is_multiple_of(4),
        "device zeroing is 32-bit aligned"
    );
    let client = RT::client();
    let words = len / 4;
    unsafe {
        fill_zero::launch_unchecked(
            client,
            cubes(client, words, 256),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(allocation.handle.clone(), allocation.reserved / 4),
            (offset / 4) as u32,
            words as u32,
        );
    }
    launched(len);
}

fn same_range(a: &Option<Allocation>, a_offset: usize, b: &Option<Allocation>, b_offset: usize) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.id == b.id && a_offset == b_offset,
        _ => false,
    }
}

fn equal<RT: GpuRuntime>(
    a: &Option<Allocation>,
    a_offset: usize,
    a_len: usize,
    b: &Option<Allocation>,
    b_offset: usize,
    b_len: usize,
) -> bool {
    if a_len != b_len {
        return false;
    }
    if same_range(a, a_offset, b, b_offset) {
        return true;
    }
    let (mut x, mut y) = (vec![0u8; a_len], vec![0u8; b_len]);
    read::<RT>(a, a_offset, &mut x);
    read::<RT>(b, b_offset, &mut y);
    x == y
}

/// Owned device storage.
pub struct GpuBuf<RT: GpuRuntime> {
    allocation: Option<Allocation>,
    len: usize,
    _runtime: PhantomData<fn() -> RT>,
}

/// Shared view of a byte range of a [`GpuBuf`].
pub struct GpuRef<'a, RT: GpuRuntime> {
    allocation: Option<Allocation>,
    offset: usize,
    len: usize,
    _borrow: PhantomData<&'a ()>,
    _runtime: PhantomData<fn() -> RT>,
}

/// Exclusive view of a byte range of a [`GpuBuf`].
pub struct GpuMut<'a, RT: GpuRuntime> {
    allocation: Option<Allocation>,
    offset: usize,
    len: usize,
    _borrow: PhantomData<&'a mut ()>,
    _runtime: PhantomData<fn() -> RT>,
}

impl<RT: GpuRuntime> GpuBuf<RT> {
    /// A buffer of `len` bytes with unspecified contents.
    pub(crate) fn uninit(len: usize) -> Self {
        Self {
            allocation: allocate::<RT>(len),
            len,
            _runtime: PhantomData,
        }
    }

    /// A zero-initialized buffer of `len` bytes.
    pub fn zeroed(len: usize) -> Self {
        let allocation = allocate::<RT>(len);
        if let Some(allocation) = &allocation {
            zero::<RT>(&Some(allocation.clone()), 0, allocation.reserved);
        }
        Self {
            allocation,
            len,
            _runtime: PhantomData,
        }
    }

    /// A buffer holding `bytes`.
    pub fn from_host(bytes: &[u8]) -> Self {
        let mut buf = Self {
            allocation: allocate::<RT>(bytes.len()),
            len: bytes.len(),
            _runtime: PhantomData,
        };
        if let Some(allocation) = &buf.allocation {
            if allocation.reserved == bytes.len() {
                write::<RT>(&buf.allocation, 0, bytes);
            } else {
                let mut padded = vec![0u8; allocation.reserved];
                padded[..bytes.len()].copy_from_slice(bytes);
                write::<RT>(&buf.allocation, 0, &padded);
            }
        }
        buf.len = bytes.len();
        buf
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_host(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.len];
        read::<RT>(&self.allocation, 0, &mut bytes);
        bytes
    }

    /// Writes `src` at the start of the buffer and zeroes the rest.
    pub fn write_prefix(&mut self, src: &[u8]) {
        assert!(src.len() <= self.len, "host slice longer than the device buffer");
        let Some(allocation) = &self.allocation else { return };
        let mut padded = vec![0u8; allocation.reserved];
        padded[..src.len()].copy_from_slice(src);
        write::<RT>(&self.allocation, 0, &padded);
    }

    pub fn view(&self) -> GpuRef<'_, RT> {
        self.region(0, self.len)
    }

    pub fn view_mut(&mut self) -> GpuMut<'_, RT> {
        let len = self.len;
        self.region_mut(0, len)
    }

    pub fn region(&self, offset: usize, len: usize) -> GpuRef<'_, RT> {
        assert!(offset + len <= self.len, "device region out of bounds");
        GpuRef {
            allocation: self.allocation.clone(),
            offset,
            len,
            _borrow: PhantomData,
            _runtime: PhantomData,
        }
    }

    pub fn region_mut(&mut self, offset: usize, len: usize) -> GpuMut<'_, RT> {
        assert!(offset + len <= self.len, "device region out of bounds");
        GpuMut {
            allocation: self.allocation.clone(),
            offset,
            len,
            _borrow: PhantomData,
            _runtime: PhantomData,
        }
    }
}

macro_rules! view_common {
    ($view:ident) => {
        impl<RT: GpuRuntime> $view<'_, RT> {
            pub fn len(&self) -> usize {
                self.len
            }

            pub fn is_empty(&self) -> bool {
                self.len == 0
            }

            pub fn to_host(&self) -> Vec<u8> {
                let mut bytes = vec![0u8; self.len];
                read::<RT>(&self.allocation, self.offset, &mut bytes);
                bytes
            }

            pub fn read_prefix(&self, dst: &mut [u8]) {
                assert!(dst.len() <= self.len, "host slice longer than the device view");
                read::<RT>(&self.allocation, self.offset, dst);
            }

            pub fn region(&self, offset: usize, len: usize) -> GpuRef<'_, RT> {
                assert!(offset + len <= self.len, "device region out of bounds");
                GpuRef {
                    allocation: self.allocation.clone(),
                    offset: self.offset + offset,
                    len,
                    _borrow: PhantomData,
                    _runtime: PhantomData,
                }
            }

            /// The launch binding of this view, or `None` for an empty one.
            pub(crate) fn binding(&self) -> Option<Binding> {
                self.allocation.as_ref().map(|allocation| Binding {
                    id: allocation.id,
                    handle: allocation.handle.clone(),
                    reserved: allocation.reserved,
                    offset: self.offset,
                    len: self.len,
                })
            }
        }

        impl<RT: GpuRuntime> Default for $view<'_, RT> {
            fn default() -> Self {
                Self {
                    allocation: None,
                    offset: 0,
                    len: 0,
                    _borrow: PhantomData,
                    _runtime: PhantomData,
                }
            }
        }

        impl<RT: GpuRuntime> CopyToHost for $view<'_, RT> {
            fn len_bytes(&self) -> usize {
                self.len
            }

            fn copy_to_host(&self, dst: &mut [u8]) {
                assert_eq!(dst.len(), self.len);
                read::<RT>(&self.allocation, self.offset, dst);
            }
        }
    };
}

view_common!(GpuRef);
view_common!(GpuMut);

impl<RT: GpuRuntime> GpuMut<'_, RT> {
    pub fn region_mut(&mut self, offset: usize, len: usize) -> GpuMut<'_, RT> {
        assert!(offset + len <= self.len, "device region out of bounds");
        GpuMut {
            allocation: self.allocation.clone(),
            offset: self.offset + offset,
            len,
            _borrow: PhantomData,
            _runtime: PhantomData,
        }
    }

    /// Writes `src` at the start of the view and zeroes the rest.
    pub fn write_prefix(&mut self, src: &[u8]) {
        assert!(src.len() <= self.len, "host slice longer than the device view");
        if src.len() == self.len {
            write::<RT>(&self.allocation, self.offset, src);
        } else {
            let mut padded = vec![0u8; self.len];
            padded[..src.len()].copy_from_slice(src);
            write::<RT>(&self.allocation, self.offset, &padded);
        }
    }
}

impl<RT: GpuRuntime> CopyFromHost for GpuMut<'_, RT> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn copy_from_host(&mut self, src: &[u8]) {
        assert_eq!(src.len(), self.len);
        write::<RT>(&self.allocation, self.offset, src);
    }
}

impl<RT: GpuRuntime> Default for GpuBuf<RT> {
    fn default() -> Self {
        Self {
            allocation: None,
            len: 0,
            _runtime: PhantomData,
        }
    }
}

impl<RT: GpuRuntime> CopyToHost for GpuBuf<RT> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn copy_to_host(&self, dst: &mut [u8]) {
        assert_eq!(dst.len(), self.len);
        read::<RT>(&self.allocation, 0, dst);
    }
}

impl<RT: GpuRuntime> CopyFromHost for GpuBuf<RT> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn copy_from_host(&mut self, src: &[u8]) {
        assert_eq!(src.len(), self.len);
        self.write_prefix(src);
    }
}

impl<RT: GpuRuntime> PartialEq for GpuBuf<RT> {
    fn eq(&self, other: &Self) -> bool {
        equal::<RT>(&self.allocation, 0, self.len, &other.allocation, 0, other.len)
    }
}

impl<RT: GpuRuntime> PartialEq for GpuRef<'_, RT> {
    fn eq(&self, other: &Self) -> bool {
        equal::<RT>(
            &self.allocation,
            self.offset,
            self.len,
            &other.allocation,
            other.offset,
            other.len,
        )
    }
}

impl<RT: GpuRuntime> PartialEq for GpuMut<'_, RT> {
    fn eq(&self, other: &Self) -> bool {
        equal::<RT>(
            &self.allocation,
            self.offset,
            self.len,
            &other.allocation,
            other.offset,
            other.len,
        )
    }
}

impl<RT: GpuRuntime> Eq for GpuBuf<RT> {}
impl<RT: GpuRuntime> Eq for GpuRef<'_, RT> {}
impl<RT: GpuRuntime> Eq for GpuMut<'_, RT> {}
