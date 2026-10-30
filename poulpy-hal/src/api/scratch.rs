//! Scratch memory allocation, borrowing, and arena-style sub-allocation.
//!
//! Provides traits for creating scratch buffers, borrowing them as
//! backend-native [`ScratchArena`] values, and carving typed layout
//! objects (e.g., [`VecZnx`], [`VecZnxDft`], [`VmpPMat`]) out of them.

use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLViewMut, CnvPVecR, CnvPVecRViewMut, MatZnx, MatZnxViewMut, PrepareHint, ScalarZnx,
    ScalarZnxViewMut, ScratchArena, SvpPPol, SvpPPolViewMut, VecZnx, VecZnxBig, VecZnxBigViewMut, VecZnxDft, VecZnxDftViewMut,
    VecZnxViewMut, VmpPMat, VmpPMatViewMut,
};

/// Allocates a [`ScratchOwned`](crate::layouts::ScratchOwned).
///
/// ```text
/// op         ScratchOwned::<BE>::alloc(size)
/// class      support
/// mutation   none
/// domain     size: bytes, at least the largest `*_tmp_bytes` the caller will hand out
/// ensures    returns an owned buffer of at least `size` bytes with the backend's alignment; its contents are unspecified
/// test       none
/// ```
pub trait ScratchOwnedAlloc<B: Backend> {
    /// Returns an owned buffer of at least `size` bytes at the backend's alignment.
    fn alloc(size: usize) -> Self;
}

/// Borrows an owned scratch buffer as a backend-native arena.
///
/// ```text
/// op         borrow()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns a ScratchArena over the whole owned buffer; the arena's contents are unspecified on entry and after every call that takes it
/// test       none
/// ```
pub trait ScratchOwnedBorrow<B: Backend> {
    /// Returns a [`ScratchArena`] over the whole owned buffer.
    fn borrow(&mut self) -> ScratchArena<'_, B>;
}

/// Queries how many bytes remain in a [`ScratchArena`].
///
/// ```text
/// op         available()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the number of bytes still carvable from the arena
/// test       none
/// ```
pub trait ScratchAvailable {
    /// Returns the number of bytes still carvable from the arena.
    fn available(&self) -> usize;
}

/// A borrowed scratch region the host can address as a byte slice.
///
/// ```text
/// op         into_bytes()
/// class      support
/// mutation   none
/// domain     a borrowed scratch region of a backend whose memory the host can address
/// ensures    returns the region as a host byte slice; a backend whose scratch lives on a device does not implement this
/// test       none
/// ```
pub trait HostBufMut<'a>: Sized {
    /// Returns the region as a host byte slice.
    fn into_bytes(self) -> &'a mut [u8];
}

impl<'a> HostBufMut<'a> for &'a mut [u8] {
    #[inline]
    fn into_bytes(self) -> &'a mut [u8] {
        self
    }
}

/// Arena allocation of [`CnvPVecL`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_cnv_pvec_left_scratch(n, cols, size, hint)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeCnvPVecL<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`CnvPVecL`] of `cols` columns and `size` limbs under `hint`, beside the remaining arena.
    fn take_cnv_pvec_left_scratch(self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> (CnvPVecLViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`CnvPVecR`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_cnv_pvec_right_scratch(n, cols, size, hint)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeCnvPVecR<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`CnvPVecR`] of `cols` columns and `size` limbs under `hint`, beside the remaining arena.
    fn take_cnv_pvec_right_scratch(self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> (CnvPVecRViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`ScalarZnx`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_scalar_znx_scratch(n, cols)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeScalarZnx<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`ScalarZnx`] of `cols` columns, beside the remaining arena.
    fn take_scalar_znx_scratch(self, n: usize, cols: usize) -> (ScalarZnxViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`SvpPPol`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_svp_ppol_scratch(n, cols, hint)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeSvpPPol<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`SvpPPol`] of `cols` columns under `hint`, beside the remaining arena.
    fn take_svp_ppol_scratch(self, n: usize, cols: usize, hint: PrepareHint) -> (SvpPPolViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`VecZnx`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_vec_znx_scratch(n, cols, size) / take_vec_znx_slice_scratch(len, n, cols, size)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeVecZnx<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`VecZnx`] of `cols` columns and `size` limbs, beside the remaining arena.
    fn take_vec_znx_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxViewMut<'a, B>, Self)
    where
        B: 'a;

    /// Returns `len` consecutive degree-`n` [`VecZnx`] of `cols` columns and `size` limbs, beside the remaining arena.
    fn take_vec_znx_slice_scratch(self, len: usize, n: usize, cols: usize, size: usize) -> (Vec<VecZnxViewMut<'a, B>>, Self)
    where
        B: 'a,
    {
        let mut scratch: Self = self;
        let mut slice: Vec<VecZnxViewMut<'a, B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, rem) = scratch.take_vec_znx_scratch(n, cols, size);
            scratch = rem;
            slice.push(znx);
        }
        (slice, scratch)
    }
}

/// Arena allocation of [`VecZnxBig`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_vec_znx_big_scratch(n, cols, size)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeVecZnxBig<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`VecZnxBig`] of `cols` columns and `size` limbs, beside the remaining arena.
    fn take_vec_znx_big_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxBigViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`VecZnxDft`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_vec_znx_dft_scratch(n, cols, size) / take_vec_znx_dft_slice_scratch(n, len, cols, size)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeVecZnxDft<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`VecZnxDft`] of `cols` columns and `size` limbs, beside the remaining arena.
    fn take_vec_znx_dft_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxDftViewMut<'a, B>, Self)
    where
        B: 'a;

    /// Returns `len` consecutive degree-`n` [`VecZnxDft`] of `cols` columns and `size` limbs, beside the remaining arena.
    fn take_vec_znx_dft_slice_scratch(
        self,
        n: usize,
        len: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDftViewMut<'a, B>>, Self)
    where
        B: 'a,
    {
        let mut scratch: Self = self;
        let mut slice: Vec<VecZnxDftViewMut<'a, B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, rem) = scratch.take_vec_znx_dft_scratch(n, cols, size);
            scratch = rem;
            slice.push(znx);
        }
        (slice, scratch)
    }
}

/// Arena allocation of [`VmpPMat`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_vmp_pmat_scratch(n, rows, cols_in, cols_out, size, hint)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeVmpPMat<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`VmpPMat`] of the given dimensions under `hint`, beside the remaining arena.
    fn take_vmp_pmat_scratch(
        self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        hint: PrepareHint,
    ) -> (VmpPMatViewMut<'a, B>, Self)
    where
        B: 'a;
}

/// Arena allocation of [`MatZnx`] layouts out of a [`ScratchArena`].
///
/// ```text
/// op         take_mat_znx_scratch(n, rows, cols_in, cols_out, size)
/// class      support
/// mutation   none
/// domain     n: the degree of the carved layout, a power of two at most the module's degree; a take fed to a kernel obeys that kernel's degree floor; the arena holds at least the matching *_bytes_of
/// ensures    consumes the arena and returns the carved layout, tagged with the requested dimensions, beside the remaining arena; the carved bytes are unspecified, so a caller that reads before writing zeroes first
/// test       none
/// ```
pub trait ScratchArenaTakeMatZnx<'a, B: Backend>: Sized {
    /// Returns a degree-`n` [`MatZnx`] of the given dimensions, beside the remaining arena.
    fn take_mat_znx_scratch(
        self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnxViewMut<'a, B>, Self)
    where
        B: 'a;
}

impl<'a, B: Backend> ScratchArenaTakeCnvPVecL<'a, B> for ScratchArena<'a, B> {
    fn take_cnv_pvec_left_scratch(self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> (CnvPVecLViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::cnv_pvec_left_bytes_of(n, cols, size, hint));
        (
            CnvPVecLViewMut::from_inner(CnvPVecL::from_data(data, n, cols, size, hint)),
            arena,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeCnvPVecR<'a, B> for ScratchArena<'a, B> {
    fn take_cnv_pvec_right_scratch(self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> (CnvPVecRViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::cnv_pvec_right_bytes_of(n, cols, size, hint));
        (
            CnvPVecRViewMut::from_inner(CnvPVecR::from_data(data, n, cols, size, hint)),
            arena,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeScalarZnx<'a, B> for ScratchArena<'a, B> {
    fn take_scalar_znx_scratch(self, n: usize, cols: usize) -> (ScalarZnxViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::scalar_znx_bytes_of(n, cols));
        (ScalarZnxViewMut::from_inner(ScalarZnx::from_data(data, n, cols)), arena)
    }
}

impl<'a, B: Backend> ScratchArenaTakeSvpPPol<'a, B> for ScratchArena<'a, B> {
    fn take_svp_ppol_scratch(self, n: usize, cols: usize, hint: PrepareHint) -> (SvpPPolViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::svp_ppol_bytes_of(n, cols, hint));
        (SvpPPolViewMut::from_inner(SvpPPol::from_data(data, n, cols, hint)), arena)
    }
}

impl<'a, B: Backend> ScratchArenaTakeVecZnx<'a, B> for ScratchArena<'a, B> {
    fn take_vec_znx_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::vec_znx_bytes_of(n, cols, size));
        (VecZnxViewMut::from_inner(VecZnx::from_data(data, n, cols, size)), arena)
    }
}

impl<'a, B: Backend> ScratchArenaTakeVecZnxBig<'a, B> for ScratchArena<'a, B> {
    fn take_vec_znx_big_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxBigViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::vec_znx_big_bytes_of(n, cols, size));
        (VecZnxBigViewMut::from_inner(VecZnxBig::from_data(data, n, cols, size)), arena)
    }
}

impl<'a, B: Backend> ScratchArenaTakeVecZnxDft<'a, B> for ScratchArena<'a, B> {
    fn take_vec_znx_dft_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxDftViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::vec_znx_dft_bytes_of(n, cols, size));
        (VecZnxDftViewMut::from_inner(VecZnxDft::from_data(data, n, cols, size)), arena)
    }
}

impl<'a, B: Backend> ScratchArenaTakeVmpPMat<'a, B> for ScratchArena<'a, B> {
    fn take_vmp_pmat_scratch(
        self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        hint: PrepareHint,
    ) -> (VmpPMatViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::vmp_pmat_bytes_of(n, rows, cols_in, cols_out, size, hint));
        (
            VmpPMatViewMut::from_inner(VmpPMat::from_data(data, n, rows, cols_in, cols_out, size, hint)),
            arena,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeMatZnx<'a, B> for ScratchArena<'a, B> {
    fn take_mat_znx_scratch(
        self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnxViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(B::mat_znx_bytes_of(n, rows, cols_in, cols_out, size));
        (
            MatZnxViewMut::from_inner(MatZnx::from_data(data, n, rows, cols_in, cols_out, size)),
            arena,
        )
    }
}
