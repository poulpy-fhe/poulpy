//! Device buffers and their borrowed views.
//!
//! An owned buffer is one CubeCL allocation. A view is a byte range of an
//! allocation; kernels bind the whole allocation and address the range through
//! an element offset, so two views of the same allocation can be read and
//! written by one kernel.

use std::{
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use cubecl::{bytes::Bytes, prelude::*, server::Handle};
use poulpy_hal::layouts::{CopyFromHost, CopyToHost};

use super::{Binding, GpuRuntime, HostEvent, Label, access, host_event, launch, on_stream};

/// Owned allocations are padded like host ones (`poulpy_hal::layouts::padded_bytes`):
/// a host-staged transfer moves an owned buffer byte for byte, so both sides
/// must span the same length.
const PADDING: usize = poulpy_hal::DEFAULTALIGN;

/// Allocations from this size on are exact-fit; smaller ones are pooled.
const EXACT_FIT_FROM: usize = 1 << 20;

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
    /// Bytes reserved on the device, a multiple of [`PADDING`].
    reserved: usize,
    /// Whether the [`GpuBuf`] that owns the allocation is alive. A recorded
    /// graph keeps the handle, hence the device memory, past that point:
    /// the flag is how it learns that nobody can name the buffer any more.
    owned: Arc<AtomicBool>,
}

/// The buffers a body being recorded dropped, kept for it to allocate again.
///
/// A capture window allocates nothing: the recorded run of a body is served
/// from what its warm-up run released. Rather than depend on what CubeCL's
/// pools keep of it, the buffers dropped on the recording thread are parked
/// here, and an allocation of the same size takes one back. Both runs
/// allocate and drop in the same order, so the recorded run finds every
/// buffer it asks for.
pub(crate) mod parked {
    use std::cell::RefCell;

    use cubecl::server::Handle;

    thread_local! {
        /// `Some` while this thread records: the parked allocations, by reserved size.
        static PARKED: RefCell<Option<Vec<(usize, Handle)>>> = const { RefCell::new(None) };
    }

    /// Parks the buffers this thread drops until the guard is dropped, which releases them.
    pub(crate) struct Parking(());

    pub(crate) fn start() -> Parking {
        PARKED.with_borrow_mut(|parked| *parked = Some(Vec::new()));
        Parking(())
    }

    impl Drop for Parking {
        fn drop(&mut self) {
            PARKED.with_borrow_mut(|parked| *parked = None);
        }
    }

    pub(super) fn active() -> bool {
        PARKED.with_borrow(Option::is_some)
    }

    pub(super) fn take(reserved: usize) -> Option<Handle> {
        PARKED.with_borrow_mut(|parked| {
            let parked = parked.as_mut()?;
            let position = parked.iter().position(|(size, _)| *size == reserved)?;
            Some(parked.remove(position).1)
        })
    }

    /// Parks `handle` when this thread records; `false` otherwise.
    pub(super) fn park(reserved: usize, handle: &Handle) -> bool {
        PARKED.with_borrow_mut(|parked| match parked {
            Some(parked) => {
                parked.push((reserved, handle.clone()));
                true
            }
            None => false,
        })
    }
}

fn allocate<RT: GpuRuntime>(len: usize) -> Option<Allocation> {
    if len == 0 {
        return None;
    }
    let reserved = len.next_multiple_of(PADDING);
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    host_event(HostEvent::Allocation, id);
    let _access = access::shared();
    let client = RT::client();
    let handle = match parked::take(reserved) {
        Some(handle) => handle,
        None => {
            // CubeCL's default pools carve allocations from pages sized to
            // the largest one served, and keep them: with a few hundred
            // megabytes per key, a key set reserved half as much again as it
            // holds. The persistent pool is exact-fit, and a freed slice
            // serves the next request of its size. It does not grow once a
            // capture is prepared: a recorded body allocates from the
            // default pools, and reuses what it drops (see `parked`).
            let handle = if reserved >= EXACT_FIT_FROM && !parked::active() {
                on_stream(|| client.memory_persistent_allocation((), |()| client.empty(reserved)))
            } else {
                client.empty(reserved)
            };
            // An allocation the device refused only fails the launches that
            // bind it, and silently when nothing reads their results back.
            if let Err(error) = client.check([&handle]) {
                panic!("device allocation of {reserved} bytes failed: {error:?}");
            }
            handle
        }
    };
    Some(Allocation {
        id,
        handle,
        reserved,
        owned: Arc::new(AtomicBool::new(true)),
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
    host_event(HostEvent::Read, allocation.id);
    let _access = access::shared();
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
    host_event(HostEvent::Write, allocation.id);
    let _access = access::shared();
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
    let words = len / 4;
    let binding = Binding {
        id: allocation.id,
        handle: allocation.handle.clone(),
        reserved: allocation.reserved,
        offset: 0,
        len: allocation.reserved,
        owned: allocation.owned.clone(),
    };
    launch!(RT, Label::BufferZero, len, fill_zero, (words, RT::GROUP, RT::GROUP); [binding => u32]; [
            u32::try_from(offset / 4).expect("device view starts beyond the 32-bit element range"),
            u32::try_from(words).expect("device view longer than the 32-bit element range"),
        ]; []);
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
    /// A buffer of at least `len` bytes with unspecified contents.
    pub(crate) fn uninit(len: usize) -> Self {
        let allocation = allocate::<RT>(len);
        Self {
            len: allocation.as_ref().map_or(0, |allocation| allocation.reserved),
            allocation,
            _runtime: PhantomData,
        }
    }

    /// A zero-initialized buffer of at least `len` bytes.
    pub fn zeroed(len: usize) -> Self {
        let allocation = allocate::<RT>(len);
        if let Some(allocation) = &allocation {
            zero::<RT>(&Some(allocation.clone()), 0, allocation.reserved);
        }
        Self {
            len: allocation.as_ref().map_or(0, |allocation| allocation.reserved),
            allocation,
            _runtime: PhantomData,
        }
    }

    /// A buffer holding `bytes`, zero-padded.
    pub fn from_host(bytes: &[u8]) -> Self {
        let allocation = allocate::<RT>(bytes.len());
        let buf = Self {
            len: allocation.as_ref().map_or(0, |allocation| allocation.reserved),
            allocation,
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
                    owned: allocation.owned.clone(),
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

impl<RT: GpuRuntime> Drop for GpuBuf<RT> {
    fn drop(&mut self) {
        if let Some(allocation) = &self.allocation {
            allocation.owned.store(false, Ordering::Release);
            parked::park(allocation.reserved, &allocation.handle);
        }
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
