//! Owned storage that frees with the layout it was allocated with.

use bytemuck::Zeroable;
use std::{
    alloc::Layout,
    fmt,
    hash::{Hash, Hasher},
    mem::{align_of, size_of},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// Owned, aligned, zero-initialized storage of `len` elements of `T`.
///
/// Built by [`alloc_aligned`](crate::alloc_aligned) and
/// [`alloc_aligned_custom`](crate::alloc_aligned_custom), or by
/// [`AlignedVec::zeroed`]; the layout used for the allocation is kept and
/// handed back on drop. Dereferences to `[T]`. `T` is `Copy`, so elements are
/// never dropped one by one, and [`Zeroable`], so the zero fill of a fresh
/// buffer and of the padding past a copied slice is a valid `T`; a type with
/// a niche at zero (`NonZeroU64`, a reference) does not qualify.
pub struct AlignedVec<T: Copy + Zeroable> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

/// Owned aligned byte storage, the host-resident owned buffer of every layout.
pub type AlignedBuf = AlignedVec<u8>;

// SAFETY: the buffer owns its allocation exclusively; sharing or sending it is
// sharing or sending a `[T]`.
unsafe impl<T: Copy + Zeroable + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Zeroable + Sync> Sync for AlignedVec<T> {}

impl<T: Copy + Zeroable> AlignedVec<T> {
    /// Zero-initialized storage of `len` elements whose first element sits on
    /// an `align`-byte boundary.
    ///
    /// # Panics
    ///
    /// If `T` is zero-sized, `align` is not a power of two, `align` is below
    /// the alignment of `T`, or `len * size_of::<T>()` is not a multiple of
    /// `align`.
    /// - If `len * size_of::<T>()` overflows `usize`.
    pub fn zeroed(len: usize, align: usize) -> Self {
        assert!(size_of::<T>() > 0, "AlignedVec: zero-sized types are not supported");
        assert!(align.is_power_of_two(), "Alignment must be a power of two but is {align}");
        assert!(
            align >= align_of::<T>(),
            "align={align} is below the alignment of the element type, {}",
            align_of::<T>()
        );
        let size: usize = len
            .checked_mul(size_of::<T>())
            .expect("AlignedVec: the element count times the element size overflows usize");
        assert!(
            size.is_multiple_of(align),
            "AlignedVec: the byte size must be a multiple of the alignment"
        );
        let layout: Layout = Layout::from_size_align(size, align).expect("AlignedVec: the byte size exceeds isize::MAX");
        if size == 0 {
            return Self {
                ptr: dangling_aligned(align),
                len: 0,
                layout,
            };
        }
        // SAFETY: `layout` has a non-zero size.
        let raw: *mut u8 = unsafe { std::alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        // Advise before the zero-fill so the faults materialise huge pages
        // directly rather than relying on khugepaged promotion.
        crate::advise_hugepage(ptr.as_ptr(), size);
        // SAFETY: `ptr` is a fresh allocation of `size` bytes.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, size) };
        Self {
            ptr: ptr.cast(),
            len,
            layout,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer holds no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment in bytes of the first element.
    pub fn align(&self) -> usize {
        self.layout.align()
    }

    /// Number of elements the allocation holds, `len` or more.
    pub fn capacity(&self) -> usize {
        self.layout.size() / size_of::<T>()
    }

    /// The elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` points at `len` initialized `T` that this buffer owns.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, through a unique borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Shortens the buffer to `len` elements; the allocation keeps its layout.
    /// A `len` at or past the current length leaves it unchanged.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }
}

/// A non-null, unallocated pointer on an `align`-byte boundary, the empty
/// buffer's address. Readers assert the alignment of their word type even on a
/// zero-length buffer, which `NonNull::dangling` does not give for `u8`.
fn dangling_aligned<T>(align: usize) -> NonNull<T> {
    NonNull::new(std::ptr::without_provenance_mut::<T>(align)).expect("align is non-zero")
}

impl<T: Copy + Zeroable> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: `ptr` was returned by `alloc(self.layout)` in `zeroed`.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), self.layout) };
        }
    }
}

impl<T: Copy + Zeroable> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::zeroed(0, crate::DEFAULTALIGN.max(align_of::<T>()))
    }
}

/// The clone allocates for its own length, padded to the alignment, whatever
/// the original's allocation holds.
impl<T: Copy + Zeroable> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let align = self.layout.align();
        let size = (self.len * size_of::<T>()).next_multiple_of(align);
        let mut out: Self = Self::zeroed(size / size_of::<T>(), align);
        out.truncate(self.len);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl<T: Copy + Zeroable> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Zeroable> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Zeroable> AsRef<[T]> for AlignedVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Zeroable> AsMut<[T]> for AlignedVec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Zeroable + PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Zeroable + Eq> Eq for AlignedVec<T> {}

impl<T: Copy + Zeroable + Hash> Hash for AlignedVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Copy + Zeroable + fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

/// Copies the slice into storage padded to a multiple of
/// [`DEFAULTALIGN`](crate::DEFAULTALIGN) bytes; the tail past the copied
/// elements is zero.
/// The padded length is computed in bytes, so a `T` whose size does not
/// divide 64 may not fit its own length; every word type these buffers
/// carry has a size that divides 64.
impl<T: Copy + Zeroable> From<&[T]> for AlignedVec<T> {
    fn from(src: &[T]) -> Self {
        let mut out: Self = crate::alloc_aligned::<T>(src.len());
        out.as_mut_slice()[..src.len()].copy_from_slice(src);
        out
    }
}

/// Copies the vector into padded aligned storage, as the slice conversion does.
impl<T: Copy + Zeroable> From<Vec<T>> for AlignedVec<T> {
    fn from(src: Vec<T>) -> Self {
        Self::from(src.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULTALIGN, alloc_aligned, is_aligned};

    #[test]
    fn zeroed_is_aligned_and_zero() {
        let bytes = AlignedVec::<u8>::zeroed(128, 64);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes.align(), 64);
        assert!(is_aligned(bytes.as_ptr()));
        assert!(bytes.iter().all(|&b| b == 0));

        let words = AlignedVec::<u64>::zeroed(16, 64);
        assert_eq!(words.len(), 16);
        assert!(is_aligned(words.as_ptr()));
        assert!(words.iter().all(|&w| w == 0));
    }

    #[test]
    fn alloc_aligned_pads_to_the_default_alignment() {
        assert_eq!(alloc_aligned::<u8>(100).len(), 128);
        assert_eq!(alloc_aligned::<u64>(3).len(), 8);
        assert_eq!(alloc_aligned::<f64>(0).len(), 0);
        assert_eq!(alloc_aligned::<u8>(64).align(), DEFAULTALIGN);
        // An empty buffer still reports an aligned address: its readers assert
        // the alignment of their word type whatever the length.
        assert!(is_aligned(alloc_aligned::<u8>(0).as_ptr()));
        assert!(is_aligned(AlignedBuf::default().as_ptr()));
    }

    #[test]
    fn drop_frees_with_the_allocating_layout() {
        // Every size and alignment below is a distinct layout Miri checks on drop.
        for (len, align) in [(64usize, 64usize), (128, 64), (4096, 64), (32, 32), (256, 128)] {
            let buf = AlignedVec::<u8>::zeroed(len, align);
            assert_eq!(buf.len(), len);
            drop(buf);
        }
        drop(AlignedVec::<u64>::zeroed(8, 64));
        drop(AlignedVec::<u8>::zeroed(0, 64));
        drop(AlignedVec::<u8>::default());
    }

    #[test]
    fn clone_copies_and_keeps_alignment() {
        let mut a = alloc_aligned::<u64>(8);
        for (i, w) in a.iter_mut().enumerate() {
            *w = i as u64 * 3 + 1;
        }
        let b = a.clone();
        assert_eq!(a, b);
        assert!(is_aligned(b.as_ptr()));
        assert_eq!(b.align(), a.align());
        a[0] = 99;
        assert_ne!(a, b);
    }

    #[test]
    fn from_slice_pads_and_zeroes_the_tail() {
        let src: Vec<u8> = (0..100).map(|i| (i * 7 + 3) as u8).collect();
        let buf = AlignedBuf::from(src.as_slice());
        assert_eq!(buf.len(), 128);
        assert!(is_aligned(buf.as_ptr()));
        assert_eq!(&buf[..100], &src[..]);
        assert!(buf[100..].iter().all(|&b| b == 0));

        let from_vec = AlignedBuf::from(src.clone());
        assert_eq!(from_vec, buf);

        let empty = AlignedBuf::from(Vec::<u8>::new());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn truncate_shrinks_the_length_only() {
        let mut buf = AlignedVec::<u8>::zeroed(128, 64);
        buf.truncate(200);
        assert_eq!(buf.len(), 128);
        buf.truncate(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.as_slice().len(), 100);
        let copy = buf.clone();
        assert_eq!(copy.len(), 100);
        assert_eq!(copy, buf);
        // Dropping a truncated buffer frees the original layout (Miri checks it).
    }

    #[test]
    fn eq_hash_and_debug_follow_the_slice() {
        use std::{collections::hash_map::DefaultHasher, hash::Hasher};
        let a = AlignedBuf::from(vec![1u8, 2, 3]);
        let b = AlignedBuf::from(vec![1u8, 2, 3]);
        let c = AlignedBuf::from(vec![1u8, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let hash = |v: &AlignedBuf| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
        assert_eq!(format!("{:?}", a), format!("{:?}", a.as_slice()));
    }

    /// The clone of a truncated buffer allocates for the truncated length.
    #[test]
    fn clone_allocates_for_the_truncated_length() {
        let mut v = AlignedVec::<u64>::zeroed(64, 64);
        v.as_mut_slice()[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        v.truncate(8);
        let c = v.clone();
        assert_eq!(c.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!((c.len(), c.capacity(), c.align()), (8, 8, 64));
        assert_eq!((v.len(), v.capacity()), (8, 64));
    }

    #[test]
    #[should_panic(expected = "must be a multiple of the alignment")]
    fn zeroed_rejects_an_unpadded_size() {
        let _ = AlignedVec::<u8>::zeroed(100, 64);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn zeroed_rejects_a_non_power_of_two_alignment() {
        let _ = AlignedVec::<u8>::zeroed(96, 48);
    }

    #[test]
    #[should_panic(expected = "is below the alignment of the element type")]
    fn zeroed_rejects_an_alignment_below_the_element() {
        let _ = AlignedVec::<u64>::zeroed(8, 4);
    }

    #[test]
    fn default_respects_an_element_alignment_above_the_default() {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(align(128))]
        struct Wide([u8; 128]);
        // SAFETY: an all-zero byte array is a valid `Wide`.
        unsafe impl Zeroable for Wide {}
        let empty = AlignedVec::<Wide>::default();
        assert!(empty.is_empty());
        assert_eq!(empty.align(), 128);
        assert!((empty.as_ptr() as usize).is_multiple_of(128));
        let copy = empty.clone();
        assert_eq!(copy.len(), 0);
        assert_eq!(copy.align(), 128);
    }
}
