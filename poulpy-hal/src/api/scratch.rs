//! Scratch memory allocation, borrowing, and arena-style sub-allocation.
//!
//! Provides traits for creating scratch buffers, borrowing them as
//! backend-native [`ScratchArena`] values, and carving typed layout
//! objects (e.g., [`VecZnx`], [`VecZnxDft`], [`VmpPMat`]) out of them.

use crate::{
    api::{CnvPVecBytesOf, ModuleN, SvpPPolBytesOf, VecZnxBigBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{
        Backend, CnvPVecL, CnvPVecLViewMut, CnvPVecR, CnvPVecRViewMut, MatZnx, MatZnxViewMut, ScalarZnx, ScalarZnxViewMut,
        ScratchArena, SvpPPol, SvpPPolViewMut, VecZnx, VecZnxBig, VecZnxBigViewMut, VecZnxDft, VecZnxDftViewMut, VecZnxViewMut,
        VmpPMat, VmpPMatViewMut,
    },
};

/// Allocates a new [crate::layouts::ScratchOwned] of `size` aligned bytes.
pub trait ScratchOwnedAlloc<B: Backend> {
    fn alloc(size: usize) -> Self;
}

/// Borrows an owned scratch buffer as a backend-native arena.
pub trait ScratchOwnedBorrow<B: Backend> {
    fn borrow(&mut self) -> ScratchArena<'_, B>;
}

/// Returns how many bytes left can be taken from the scratch.
pub trait ScratchAvailable {
    fn available(&self) -> usize;
}

/// Host-visible borrowed scratch region for a backend.
///
/// Device backends should not implement this unless their borrowed mutable
/// scratch region is directly accessible as a host byte slice.
pub trait HostBufMut<'a>: Sized {
    fn into_bytes(self) -> &'a mut [u8];
}

impl<'a> HostBufMut<'a> for &'a mut [u8] {
    #[inline]
    fn into_bytes(self) -> &'a mut [u8] {
        self
    }
}

/// Backend-native arena allocation of typed HAL layouts.
///
/// This is the additive, backend-owned scratch path introduced for
/// incremental device-backend integration. It consumes a [`ScratchArena`]
/// by value and returns the carved layout together with the remaining arena.
pub trait ScratchArenaTakeBasic<'a, B: Backend>: Sized {
    /// Takes a [`CnvPVecL`] from the scratch arena.
    fn take_cnv_pvec_left_scratch<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecLViewMut<'a, B>, Self)
    where
        B: 'a,
        M: ModuleN + CnvPVecBytesOf;

    /// Takes a [`CnvPVecR`] from the scratch arena.
    fn take_cnv_pvec_right_scratch<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecRViewMut<'a, B>, Self)
    where
        B: 'a,
        M: ModuleN + CnvPVecBytesOf;

    /// Takes a [`ScalarZnx`] from the scratch arena.
    fn take_scalar_znx_scratch(self, n: usize, cols: usize) -> (ScalarZnxViewMut<'a, B>, Self)
    where
        B: 'a;

    /// Takes a [`SvpPPol`] from the scratch arena.
    fn take_svp_ppol_scratch<M>(self, module: &M, cols: usize) -> (SvpPPolViewMut<'a, B>, Self)
    where
        B: 'a,
        M: SvpPPolBytesOf + ModuleN;

    /// Takes a [`VecZnx`] from the scratch arena.
    fn take_vec_znx_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxViewMut<'a, B>, Self)
    where
        B: 'a;

    /// Takes a [`VecZnxBig`] from the scratch arena.
    fn take_vec_znx_big_scratch<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxBigViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VecZnxBigBytesOf + ModuleN;

    /// Takes a [`VecZnxDft`] from the scratch arena.
    fn take_vec_znx_dft_scratch<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxDftViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VecZnxDftBytesOf + ModuleN;

    /// Takes `len` consecutive [`VecZnxDft`] objects from the scratch arena.
    fn take_vec_znx_dft_slice_scratch<M>(
        self,
        module: &M,
        len: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDftViewMut<'a, B>>, Self)
    where
        B: 'a,
        M: VecZnxDftBytesOf + ModuleN,
    {
        let mut scratch: Self = self;
        let mut slice: Vec<VecZnxDftViewMut<'a, B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, rem) = scratch.take_vec_znx_dft_scratch(module, cols, size);
            scratch = rem;
            slice.push(znx);
        }
        (slice, scratch)
    }

    /// Takes `len` consecutive [`VecZnx`] objects from the scratch arena.
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

    /// Takes a [`VmpPMat`] from the scratch arena.
    fn take_vmp_pmat_scratch<M>(
        self,
        module: &M,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMatViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VmpPMatBytesOf + ModuleN;

    /// Takes a [`MatZnx`] from the scratch arena.
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

impl<'a, B: Backend> ScratchArenaTakeBasic<'a, B> for ScratchArena<'a, B> {
    fn take_cnv_pvec_left_scratch<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecLViewMut<'a, B>, Self)
    where
        B: 'a,
        M: ModuleN + CnvPVecBytesOf,
    {
        let (data, arena) = self.take_region(module.bytes_of_cnv_pvec_left(cols, size));
        (
            CnvPVecLViewMut::from_inner(CnvPVecL::from_data(data, module.n(), cols, size)),
            arena,
        )
    }

    fn take_cnv_pvec_right_scratch<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecRViewMut<'a, B>, Self)
    where
        B: 'a,
        M: ModuleN + CnvPVecBytesOf,
    {
        let (data, arena) = self.take_region(module.bytes_of_cnv_pvec_right(cols, size));
        (
            CnvPVecRViewMut::from_inner(CnvPVecR::from_data(data, module.n(), cols, size)),
            arena,
        )
    }

    fn take_scalar_znx_scratch(self, n: usize, cols: usize) -> (ScalarZnxViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(ScalarZnx::bytes_of(n, cols));
        (ScalarZnxViewMut::from_inner(ScalarZnx::from_data(data, n, cols)), arena)
    }

    fn take_svp_ppol_scratch<M>(self, module: &M, cols: usize) -> (SvpPPolViewMut<'a, B>, Self)
    where
        B: 'a,
        M: SvpPPolBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_svp_ppol(cols));
        (SvpPPolViewMut::from_inner(SvpPPol::from_data(data, module.n(), cols)), arena)
    }

    fn take_vec_znx_scratch(self, n: usize, cols: usize, size: usize) -> (VecZnxViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, arena) = self.take_region(VecZnx::bytes_of(n, cols, size));
        (VecZnxViewMut::from_inner(VecZnx::from_data(data, n, cols, size)), arena)
    }

    fn take_vec_znx_big_scratch<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxBigViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VecZnxBigBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vec_znx_big(cols, size));
        (
            VecZnxBigViewMut::from_inner(VecZnxBig::from_data(data, module.n(), cols, size)),
            arena,
        )
    }

    fn take_vec_znx_dft_scratch<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxDftViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VecZnxDftBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vec_znx_dft(cols, size));
        (
            VecZnxDftViewMut::from_inner(VecZnxDft::from_data(data, module.n(), cols, size)),
            arena,
        )
    }

    fn take_vmp_pmat_scratch<M>(
        self,
        module: &M,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMatViewMut<'a, B>, Self)
    where
        B: 'a,
        M: VmpPMatBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vmp_pmat(rows, cols_in, cols_out, size));
        (
            VmpPMatViewMut::from_inner(VmpPMat::from_data(data, module.n(), rows, cols_in, cols_out, size)),
            arena,
        )
    }

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
        let (data, arena) = self.take_region(MatZnx::bytes_of(n, rows, cols_in, cols_out, size));
        (
            MatZnxViewMut::from_inner(MatZnx::from_data(data, n, rows, cols_in, cols_out, size)),
            arena,
        )
    }
}
