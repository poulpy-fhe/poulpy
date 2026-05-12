use std::{marker::PhantomData, ptr::NonNull};

use crate::layouts::Backend;

/// Owned scratch buffer for temporary workspace during polynomial operations.
///
/// Operations such as normalization, DFT, and vector-matrix products require
/// temporary scratch memory. `ScratchOwned` holds a backend-owned buffer that
/// can be borrowed as a [`ScratchArena`].
///
/// The required size for each operation is obtained via the corresponding
/// `*_tmp_bytes` method on the API trait (e.g.
/// [`VecZnxNormalizeTmpBytes`](crate::api::VecZnxNormalizeTmpBytes)).
#[repr(C)]
pub struct ScratchOwned<B: Backend> {
    pub data: B::OwnedBuf,
    pub _phantom: PhantomData<B>,
}

/// Backend-native scratch arena borrowed from a [`ScratchOwned`].
///
/// This arena keeps backend ownership explicit and carves typed temporaries
/// using the backend's native borrowed buffer view (`B::BufMut<'a>`).
pub struct ScratchArena<'a, B: Backend> {
    data: NonNull<B::OwnedBuf>,
    start: usize,
    end: usize,
    _phantom: PhantomData<&'a mut B::OwnedBuf>,
}

impl<B: Backend> ScratchOwned<B> {
    /// Borrows this owned scratch buffer as a backend-native arena.
    pub fn arena(&mut self) -> ScratchArena<'_, B> {
        ScratchArena {
            data: NonNull::from(&mut self.data),
            start: 0,
            end: B::len_bytes(&self.data),
            _phantom: PhantomData,
        }
    }
}

impl<'a, B: Backend> ScratchArena<'a, B> {
    /// Reborrows this arena with a shorter lifetime.
    pub fn borrow<'b>(&'b mut self) -> ScratchArena<'b, B> {
        ScratchArena {
            data: self.data,
            start: self.start,
            end: self.end,
            _phantom: PhantomData,
        }
    }

    /// Runs `f` with a shorter-lived reborrow of this arena.
    ///
    /// This is useful for nested workspace use where the borrowed arena
    /// must not leak into the outer function's scratch lifetime.
    pub fn scope<R>(&mut self, f: impl for<'b> FnOnce(ScratchArena<'b, B>) -> R) -> R {
        f(self.borrow())
    }

    /// Applies `f` to this arena through a temporary mutable borrow and returns the advanced arena.
    ///
    /// This is useful while migrating callers that still thread scratch by value around newer
    /// `&mut ScratchArena` APIs.
    pub fn apply_mut(mut self, f: impl FnOnce(&mut ScratchArena<'a, B>)) -> Self {
        f(&mut self);
        self
    }

    /// Runs `f` on a shorter-lived owned reborrow and commits the returned remainder.
    ///
    /// This is useful while migrating APIs from arena-by-value to `&mut ScratchArena`:
    /// existing helpers can keep their `(result, remainder)` style internally, while the
    /// outer mutable arena advances to the returned remainder.
    pub fn consume<R>(&mut self, f: impl for<'b> FnOnce(ScratchArena<'b, B>) -> (R, ScratchArena<'b, B>)) -> R {
        let arena = ScratchArena {
            data: self.data,
            start: self.start,
            end: self.end,
            _phantom: PhantomData,
        };
        let (res, rem) = f(arena);
        self.start = rem.start;
        self.end = rem.end;
        res
    }
    /// Returns the number of aligned bytes that can still be carved out.
    pub fn available(&self) -> usize {
        self.end.saturating_sub(align_up::<B>(self.start))
    }

    /// Splits off `len` aligned bytes from the front of this arena.
    pub fn split_at(self, len: usize) -> (Self, Self) {
        let start: usize = align_up::<B>(self.start);
        let mid: usize = start.checked_add(len).expect("scratch arena split overflow");
        assert!(
            mid <= self.end,
            "Attempted to take {len} from scratch arena with {} aligned bytes left",
            self.available()
        );
        (
            Self {
                data: self.data,
                start,
                end: mid,
                _phantom: PhantomData,
            },
            Self {
                data: self.data,
                start: mid,
                end: self.end,
                _phantom: PhantomData,
            },
        )
    }

    /// Splits this arena into `n` disjoint aligned chunks of `len` bytes each.
    pub fn split(self, n: usize, len: usize) -> (Vec<Self>, Self) {
        assert!(self.available() >= n * len);
        let mut arenas: Vec<Self> = Vec::with_capacity(n);
        let mut arena: Self = self;
        for _ in 0..n {
            let (taken, rem) = arena.split_at(len);
            arena = rem;
            arenas.push(taken);
        }
        (arenas, arena)
    }

    /// Takes a backend-native mutable region of `len` bytes.
    pub fn take_region(self, len: usize) -> (B::BufMut<'a>, Self) {
        let start: usize = align_up::<B>(self.start);
        let end: usize = start.checked_add(len).expect("scratch arena take overflow");
        assert!(
            end <= self.end,
            "Attempted to take {len} from scratch arena with {} aligned bytes left",
            self.available()
        );

        let data: &mut B::OwnedBuf = unsafe {
            // Safety: `self.data` originates from `ScratchOwned::arena`, which ties
            // the raw pointer to the lifetime `'a`. Each new arena produced from this
            // value advances or splits the byte range, so callers can only obtain
            // disjoint mutable regions from the same backing buffer.
            self.data.as_ptr().as_mut().expect("scratch arena owner pointer is null")
        };
        let region: B::BufMut<'a> = B::region_mut(data, start, len);
        (
            region,
            Self {
                data: self.data,
                start: end,
                end: self.end,
                _phantom: PhantomData,
            },
        )
    }
}

#[inline]
fn align_up<B: Backend>(offset: usize) -> usize {
    offset.next_multiple_of(B::SCRATCH_ALIGN)
}
