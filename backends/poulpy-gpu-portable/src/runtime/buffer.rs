//! Device buffers and their borrowed views.
//!
//! An owned buffer is one CubeCL allocation. A view is a byte range of an
//! allocation; kernels bind the whole allocation and address the range through
//! an element offset, so two views of the same allocation can be read and
//! written by one kernel.

use std::{
    any::TypeId,
    collections::BTreeMap,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
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
    /// Whether the allocation still owes its zero fill: a zeroed buffer is
    /// filled at its first use, unless that use overwrites all of it.
    fresh: Arc<AtomicBool>,
}

/// Zero-fills `allocation` if it still owes it.
fn settle<RT: GpuRuntime>(allocation: &Allocation) {
    if allocation.fresh.swap(false, Ordering::AcqRel) {
        zero::<RT>(allocation, 0, allocation.reserved);
    }
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
        /// Whether this thread is inside a capture window, where the stream records and answers nothing.
        static WINDOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Marks this thread inside a capture window until the guard is dropped.
    pub(crate) struct Window(());

    pub(crate) fn window() -> Window {
        WINDOW.set(true);
        Window(())
    }

    impl Drop for Window {
        fn drop(&mut self) {
            WINDOW.set(false);
        }
    }

    pub(super) fn in_window() -> bool {
        WINDOW.get()
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

/// The owned allocations alive, by id, with their runtime. CubeCL reports a
/// launch failure on the buffers the launch writes, not at the barrier, so
/// [`synchronize`] asks about each of them.
static LIVE: Mutex<BTreeMap<u64, (TypeId, Handle)>> = Mutex::new(BTreeMap::new());

fn live() -> std::sync::MutexGuard<'static, BTreeMap<u64, (TypeId, Handle)>> {
    LIVE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The launch failures CubeCL reported on buffers as they were released, by
/// runtime: [`synchronize`] raises them.
static RELEASED: Mutex<Vec<(TypeId, String)>> = Mutex::new(Vec::new());

fn released() -> std::sync::MutexGuard<'static, Vec<(TypeId, String)>> {
    RELEASED.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Waits for the work queued on the runtime and panics on a failure it
/// reports: a device fault, a launch failure an allocation alive at the call
/// carries, or one a buffer carried when it was released since the last call.
pub(crate) fn synchronize<RT: GpuRuntime>() {
    let handles: Vec<Handle> = live()
        .values()
        .filter(|(runtime, _)| *runtime == TypeId::of::<RT>())
        .map(|(_, handle)| handle.clone())
        .collect();
    let _access = access::shared();
    pollster::block_on(RT::client().sync_buffers(handles.iter()))
        .unwrap_or_else(|error| panic!("device synchronization failed: {error:?}"));
    raise_released::<RT>("device synchronization failed");
}

/// Panics with the first launch failure a buffer of the runtime carried when
/// it was released, if any, clearing them.
fn raise_released<RT: GpuRuntime>(context: &str) {
    let failure = {
        let mut released = released();
        let failure = released.iter().find(|(runtime, _)| *runtime == TypeId::of::<RT>()).cloned();
        released.retain(|(runtime, _)| *runtime != TypeId::of::<RT>());
        failure
    };
    if let Some((_, error)) = failure {
        panic!("{context}: a released buffer carried {error}");
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
                panic!("device allocation failed: {error:?}");
            }
            handle
        }
    };
    live().insert(id, (TypeId::of::<RT>(), handle.clone()));
    Some(Allocation {
        id,
        handle,
        reserved,
        owned: Arc::new(AtomicBool::new(true)),
        fresh: Arc::new(AtomicBool::new(false)),
    })
}

fn read<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, dst: &mut [u8]) {
    let Some(allocation) = allocation else {
        assert!(dst.is_empty(), "read from an empty device buffer");
        return;
    };
    if dst.is_empty() {
        return;
    }
    let end = offset + dst.len();
    assert!(end <= allocation.reserved, "device read out of bounds");
    settle::<RT>(allocation);
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
    // The read waited for the queued work: what released buffers carried is known.
    raise_released::<RT>("device read failed");
    dst.copy_from_slice(&bytes[..dst.len()]);
}

fn write<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, src: &[u8]) {
    write_vec::<RT>(allocation, offset, src.to_vec());
}

/// [`write`] of a host copy the caller already owns: CubeCL takes it as is.
fn write_vec<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, src: Vec<u8>) {
    let Some(allocation) = allocation else {
        assert!(src.is_empty(), "write to an empty device buffer");
        return;
    };
    if src.is_empty() {
        return;
    }
    let end = offset + src.len();
    assert!(end <= allocation.reserved, "device write out of bounds");
    if src.len() == allocation.reserved {
        allocation.fresh.store(false, Ordering::Release);
    } else {
        settle::<RT>(allocation);
    }
    let handle = allocation
        .handle
        .clone()
        .offset_start(offset as u64)
        .offset_end((allocation.reserved - end) as u64);
    host_event(HostEvent::Write, allocation.id);
    let _access = access::shared();
    RT::client().write(&handle, Bytes::from_bytes_vec(src));
}

/// Writes `src` at byte `offset` of `allocation` and zeroes the following
/// `len - src.len()` bytes on the device, without a padded host copy when
/// the zeroed range is 32-bit aligned.
fn write_prefix_at<RT: GpuRuntime>(allocation: &Option<Allocation>, offset: usize, len: usize, src: &[u8]) {
    assert!(src.len() <= len, "host slice longer than the device buffer");
    let end = offset + src.len();
    if end.is_multiple_of(4) && (offset + len).is_multiple_of(4) {
        write::<RT>(allocation, offset, src);
        if let Some(allocation) = allocation {
            zero::<RT>(allocation, end, len - src.len());
        }
    } else {
        let mut padded = vec![0u8; len];
        padded[..src.len()].copy_from_slice(src);
        write_vec::<RT>(allocation, offset, padded);
    }
}

fn allocation_of(binding: &Binding) -> Option<Allocation> {
    Some(Allocation {
        id: binding.id,
        handle: binding.handle.clone(),
        reserved: binding.reserved,
        owned: binding.owned.clone(),
        // A binding is only made of a settled allocation.
        fresh: Arc::new(AtomicBool::new(false)),
    })
}

/// Zeroes a host copy of secret-derived words before it is freed, with
/// volatile stores the compiler cannot drop as dead. The staging copy CubeCL
/// owns for a transfer, uploaded or read back, is out of its reach.
pub(crate) fn wipe_host(bytes: &mut [u8]) {
    for byte in bytes.iter_mut() {
        // SAFETY: `byte` is a valid, exclusive reference.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// Reads `dst.len()` bytes at byte `offset` of the allocation `binding` views.
pub(crate) fn read_at<RT: GpuRuntime>(binding: &Binding, offset: usize, dst: &mut [u8]) {
    read::<RT>(&allocation_of(binding), offset, dst);
}

/// Writes `src` at byte `offset` of the allocation `binding` views.
pub(crate) fn write_at<RT: GpuRuntime>(binding: &Binding, offset: usize, src: &[u8]) {
    write::<RT>(&allocation_of(binding), offset, src);
}

fn zero<RT: GpuRuntime>(allocation: &Allocation, offset: usize, len: usize) {
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

    /// A zero-initialized buffer of at least `len` bytes. The fill is
    /// deferred to the first use of the buffer.
    pub(crate) fn zeroed(len: usize) -> Self {
        let allocation = allocate::<RT>(len);
        if let Some(allocation) = &allocation {
            allocation.fresh.store(true, Ordering::Release);
        }
        Self {
            len: allocation.as_ref().map_or(0, |allocation| allocation.reserved),
            allocation,
            _runtime: PhantomData,
        }
    }

    /// A buffer holding `bytes`, zero-padded.
    pub(crate) fn from_host(bytes: &[u8]) -> Self {
        let allocation = allocate::<RT>(bytes.len());
        let buf = Self {
            len: allocation.as_ref().map_or(0, |allocation| allocation.reserved),
            allocation,
            _runtime: PhantomData,
        };
        write_prefix_at::<RT>(&buf.allocation, 0, buf.len, bytes);
        buf
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Zeroes the whole allocation on the device.
    pub(crate) fn wipe(&mut self) {
        if let Some(allocation) = &self.allocation {
            settle::<RT>(allocation);
            zero::<RT>(allocation, 0, self.len);
        }
    }

    pub(crate) fn to_host(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.len];
        read::<RT>(&self.allocation, 0, &mut bytes);
        bytes
    }

    /// Writes `src` at the start of the buffer and zeroes the rest.
    pub(crate) fn write_prefix(&mut self, src: &[u8]) {
        write_prefix_at::<RT>(&self.allocation, 0, self.len, src);
    }

    pub(crate) fn view(&self) -> GpuRef<'_, RT> {
        self.region(0, self.len)
    }

    pub(crate) fn view_mut(&mut self) -> GpuMut<'_, RT> {
        let len = self.len;
        self.region_mut(0, len)
    }

    pub(crate) fn region(&self, offset: usize, len: usize) -> GpuRef<'_, RT> {
        assert!(offset + len <= self.len, "device region out of bounds");
        GpuRef {
            allocation: self.allocation.clone(),
            offset,
            len,
            _borrow: PhantomData,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn region_mut(&mut self, offset: usize, len: usize) -> GpuMut<'_, RT> {
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
            pub(crate) fn len(&self) -> usize {
                self.len
            }

            pub(crate) fn region(&self, offset: usize, len: usize) -> GpuRef<'_, RT> {
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
                self.allocation.as_ref().map(|allocation| {
                    settle::<RT>(allocation);
                    self.bound(allocation)
                })
            }

            fn bound(&self, allocation: &Allocation) -> Binding {
                Binding {
                    id: allocation.id,
                    handle: allocation.handle.clone(),
                    reserved: allocation.reserved,
                    offset: self.offset,
                    len: self.len,
                    owned: allocation.owned.clone(),
                }
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
                assert!(
                    dst.len() == self.len,
                    "host slice length differs from the device buffer"
                );
                read::<RT>(&self.allocation, self.offset, dst);
            }
        }
    };
}

view_common!(GpuRef);
view_common!(GpuMut);

// Host staging: the CKKS encoding and the tests.
#[cfg_attr(not(feature = "enable-ckks"), allow(dead_code))]
impl<RT: GpuRuntime> GpuRef<'_, RT> {
    pub(crate) fn to_host(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.len];
        read::<RT>(&self.allocation, self.offset, &mut bytes);
        bytes
    }

    pub(crate) fn read_prefix(&self, dst: &mut [u8]) {
        assert!(dst.len() <= self.len, "host slice longer than the device view");
        read::<RT>(&self.allocation, self.offset, dst);
    }
}

impl<RT: GpuRuntime> GpuMut<'_, RT> {
    /// [`Self::binding`] for a launch that writes the first `written` bytes
    /// of the view: when that is a whole buffer that still owes its zero
    /// fill, the fill is skipped.
    pub(crate) fn binding_overwritten(&self, written: usize) -> Option<Binding> {
        self.allocation.as_ref().map(|allocation| {
            if self.offset == 0 && written == allocation.reserved {
                allocation.fresh.store(false, Ordering::Release);
            } else {
                settle::<RT>(allocation);
            }
            self.bound(allocation)
        })
    }

    pub(crate) fn region_mut(&mut self, offset: usize, len: usize) -> GpuMut<'_, RT> {
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
    pub(crate) fn write_prefix(&mut self, src: &[u8]) {
        write_prefix_at::<RT>(&self.allocation, self.offset, self.len, src);
    }
}

impl<RT: GpuRuntime> CopyFromHost for GpuMut<'_, RT> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn copy_from_host(&mut self, src: &[u8]) {
        assert!(src.len() == self.len, "host slice length differs from the device buffer");
        write::<RT>(&self.allocation, self.offset, src);
    }
}

impl<RT: GpuRuntime> Drop for GpuBuf<RT> {
    fn drop(&mut self) {
        if let Some(allocation) = &self.allocation {
            allocation.owned.store(false, Ordering::Release);
            live().remove(&allocation.id);
            parked::park(allocation.reserved, &allocation.handle);
            // The capture window records launches and answers no query; the
            // warm-up run before it executes, and its buffers are checked.
            if parked::in_window() {
                return;
            }
            // The launch failures recorded on the buffer, kept for the next wait or read.
            let _access = access::shared();
            if let Err(error) = RT::client().check([&allocation.handle]) {
                released().push((TypeId::of::<RT>(), format!("{error:?}")));
            }
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
        assert!(dst.len() == self.len, "host slice length differs from the device buffer");
        read::<RT>(&self.allocation, 0, dst);
    }
}

impl<RT: GpuRuntime> CopyFromHost for GpuBuf<RT> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn copy_from_host(&mut self, src: &[u8]) {
        assert!(src.len() == self.len, "host slice length differs from the device buffer");
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
