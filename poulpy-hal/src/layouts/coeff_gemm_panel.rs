use std::marker::PhantomData;

use crate::layouts::{Backend, Data, DataView, DataViewMut, HostDataMut, HostDataRef};

/// Shape of a prepared `U` panel: a piece-major `i16`/`i32` decomposition of a
/// `CoeffMatrix`. `w` is the SIMD piece width (16 or 32), `up` the number of
/// `U`-pieces (1 or 2). Element size is 2 bytes for `w = 16`, else 4.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct CoeffGemmPanelShape {
    rows_out: usize,
    rows_in: usize,
    u_size: usize,
    w: u32,
    up: usize,
}

impl CoeffGemmPanelShape {
    pub const fn new(rows_out: usize, rows_in: usize, u_size: usize, w: u32, up: usize) -> Self {
        Self {
            rows_out,
            rows_in,
            u_size,
            w,
            up,
        }
    }
    pub const fn rows_out(self) -> usize {
        self.rows_out
    }
    pub const fn rows_in(self) -> usize {
        self.rows_in
    }
    pub const fn u_size(self) -> usize {
        self.u_size
    }
    pub const fn w(self) -> u32 {
        self.w
    }
    pub const fn up(self) -> usize {
        self.up
    }
    /// Element byte size implied by `w` (`i16` for `w = 16`, else `i32`).
    pub const fn elt_bytes(self) -> usize {
        if self.w <= 16 { 2 } else { 4 }
    }
    /// Number of prepared elements.
    pub const fn elems(self) -> usize {
        self.u_size * self.rows_out * self.rows_in * self.up
    }
}

/// Backend-owned prepared `U` panel for the GEMM `vec_znx_matmul_prepared` path.
/// Holds the piece-major balanced decomposition of `U` (computed once) so it can
/// be reused across many right-hand-side batches without re-preparing `U`.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct CoeffGemmPanel<D: Data, B: Backend> {
    data: D,
    shape: CoeffGemmPanelShape,
    _phantom: PhantomData<B>,
}

impl<D: Data, B: Backend> DataView for CoeffGemmPanel<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}
impl<D: Data, B: Backend> DataViewMut for CoeffGemmPanel<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, B: Backend> CoeffGemmPanel<D, B> {
    pub fn shape(&self) -> CoeffGemmPanelShape {
        self.shape
    }
    pub fn rows_out(&self) -> usize {
        self.shape.rows_out()
    }
    pub fn rows_in(&self) -> usize {
        self.shape.rows_in()
    }
    pub fn u_size(&self) -> usize {
        self.shape.u_size()
    }
    pub fn w(&self) -> u32 {
        self.shape.w()
    }
    pub fn up(&self) -> usize {
        self.shape.up()
    }

    pub fn bytes_of(rows_out: usize, rows_in: usize, u_size: usize, w: u32, up: usize) -> usize {
        let elt = if w <= 16 { 2 } else { 4 };
        u_size * rows_out * rows_in * up * elt
    }

    pub fn from_data(data: D, rows_out: usize, rows_in: usize, u_size: usize, w: u32, up: usize) -> Self {
        Self {
            data,
            shape: CoeffGemmPanelShape::new(rows_out, rows_in, u_size, w, up),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, B: Backend> CoeffGemmPanel<D, B> {
    /// Prepared elements reinterpreted as `i16` (valid iff `w <= 16`).
    pub fn raw_i16(&self) -> &[i16] {
        debug_assert!(self.shape.w() <= 16, "CoeffGemmPanel::raw_i16 on a w>16 panel");
        unsafe { std::slice::from_raw_parts(self.data.as_ref().as_ptr() as *const i16, self.shape.elems()) }
    }
    /// Prepared elements reinterpreted as `i32` (valid iff `w > 16`).
    pub fn raw_i32(&self) -> &[i32] {
        debug_assert!(self.shape.w() > 16, "CoeffGemmPanel::raw_i32 on a w<=16 panel");
        unsafe { std::slice::from_raw_parts(self.data.as_ref().as_ptr() as *const i32, self.shape.elems()) }
    }
}

impl<D: HostDataMut, B: Backend> CoeffGemmPanel<D, B> {
    pub fn raw_mut_i16(&mut self) -> &mut [i16] {
        let elems = self.shape.elems();
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut().as_mut_ptr() as *mut i16, elems) }
    }
    pub fn raw_mut_i32(&mut self) -> &mut [i32] {
        let elems = self.shape.elems();
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut().as_mut_ptr() as *mut i32, elems) }
    }
}

impl<B: Backend> CoeffGemmPanel<<B as Backend>::OwnedBuf, B> {
    pub fn alloc(rows_out: usize, rows_in: usize, u_size: usize, w: u32, up: usize) -> Self {
        let data = B::alloc_zeroed_bytes(Self::bytes_of(rows_out, rows_in, u_size, w, up));
        Self {
            data,
            shape: CoeffGemmPanelShape::new(rows_out, rows_in, u_size, w, up),
            _phantom: PhantomData,
        }
    }
}

pub type CoeffGemmPanelOwned<B> = CoeffGemmPanel<<B as Backend>::OwnedBuf, B>;
pub type CoeffGemmPanelBackendRef<'a, B> = CoeffGemmPanel<<B as Backend>::BufRef<'a>, B>;
pub type CoeffGemmPanelBackendMut<'a, B> = CoeffGemmPanel<<B as Backend>::BufMut<'a>, B>;

pub trait CoeffGemmPanelToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> CoeffGemmPanelBackendRef<'_, B>;
}
impl<B: Backend> CoeffGemmPanelToBackendRef<B> for CoeffGemmPanel<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> CoeffGemmPanelBackendRef<'_, B> {
        CoeffGemmPanel {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}
impl<'b, B: Backend + 'b> CoeffGemmPanel<B::BufMut<'b>, B> {
    pub fn to_backend_ref(&self) -> CoeffGemmPanelBackendRef<'_, B> {
        CoeffGemmPanel {
            data: B::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

pub trait CoeffGemmPanelToBackendMut<B: Backend>: CoeffGemmPanelToBackendRef<B> {
    fn to_backend_mut(&mut self) -> CoeffGemmPanelBackendMut<'_, B>;
}
impl<B: Backend> CoeffGemmPanelToBackendMut<B> for CoeffGemmPanel<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> CoeffGemmPanelBackendMut<'_, B> {
        CoeffGemmPanel {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}
