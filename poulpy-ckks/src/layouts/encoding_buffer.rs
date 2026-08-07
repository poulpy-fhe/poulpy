//! Backend-resident scalar workspace for CKKS encoding transforms.
//!
//! The buffer carries only a scalar count and opaque backend storage. It is
//! deliberately not a Rust slice: a device backend can keep the values in
//! device memory and run its FFT, slot permutation, and plaintext codec there.

use std::marker::PhantomData;

use anyhow::{Result, ensure};
use bytemuck::Pod;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef};

/// Shape information shared by owned and borrowed encoding buffers.
pub trait CKKSEncodingBufferInfos {
    /// Number of scalar values in the buffer.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A one-dimensional array of CKKS encoding scalars in backend storage.
///
/// Slot values use planar layout `[re_0, ..., re_{m-1}, im_0, ..., im_{m-1}]`.
/// The same storage holds the `2m` real polynomial coefficients after the
/// in-place slot-to-coefficient transform.
#[repr(C)]
pub struct CKKSEncodingBuffer<D: Data, F> {
    /// Opaque host- or device-resident bytes.
    pub data: D,
    len: usize,
    _scalar: PhantomData<F>,
}

impl<D: Data, F> CKKSEncodingBuffer<D, F> {
    pub fn from_data(data: D, len: usize) -> Self {
        Self {
            data,
            len,
            _scalar: PhantomData,
        }
    }

    pub fn bytes_of(len: usize) -> usize {
        len.checked_mul(std::mem::size_of::<F>())
            .expect("CKKS encoding buffer size overflow")
    }
}

impl<D: Data, F> CKKSEncodingBufferInfos for CKKSEncodingBuffer<D, F> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<D: HostDataRef, F: Pod> CKKSEncodingBuffer<D, F> {
    /// Host view used by host backend implementations.
    pub fn as_slice(&self) -> &[F] {
        let values = bytemuck::cast_slice(self.data.as_ref());
        assert_eq!(values.len(), self.len);
        values
    }
}

impl<D: HostDataMut, F: Pod> CKKSEncodingBuffer<D, F> {
    /// Mutable host view used by host backend implementations.
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        let values = bytemuck::cast_slice_mut(self.data.as_mut());
        assert_eq!(values.len(), self.len);
        values
    }
}

impl<D: Data, F: Pod> CKKSEncodingBuffer<D, F> {
    /// Uploads a host scalar array into a backend-owned encoding buffer.
    pub fn from_host<BE: Backend<OwnedBuf = D>>(values: &[F]) -> Self {
        Self::from_data(BE::from_host_bytes(bytemuck::cast_slice(values)), values.len())
    }

    /// Downloads a backend-owned encoding buffer into host scalars.
    pub fn to_host<BE: Backend<OwnedBuf = D>>(&self) -> Vec<F> {
        let mut values = vec![F::zeroed(); self.len];
        BE::copy_to_host(&self.data, bytemuck::cast_slice_mut(&mut values));
        values
    }

    /// Replaces the values without reallocating the backend buffer.
    pub fn copy_from_host<BE: Backend<OwnedBuf = D>>(&mut self, values: &[F]) {
        assert_eq!(values.len(), self.len);
        BE::copy_from_host(&mut self.data, bytemuck::cast_slice(values));
    }
}

pub type CKKSEncodingBufferBackendRef<'a, BE, F> = CKKSEncodingBuffer<<BE as Backend>::BufRef<'a>, F>;
pub type CKKSEncodingBufferBackendMut<'a, BE, F> = CKKSEncodingBuffer<<BE as Backend>::BufMut<'a>, F>;

/// Mutable encoding-buffer view carved from a backend scratch arena.
pub struct CKKSEncodingBufferViewMut<'a, BE: Backend + 'a, F> {
    inner: CKKSEncodingBufferBackendMut<'a, BE, F>,
}

impl<'a, BE: Backend + 'a, F> CKKSEncodingBufferViewMut<'a, BE, F> {
    pub(crate) fn from_inner(inner: CKKSEncodingBufferBackendMut<'a, BE, F>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> CKKSEncodingBufferBackendMut<'a, BE, F> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a, F> CKKSEncodingBufferInfos for CKKSEncodingBufferViewMut<'a, BE, F> {
    fn len(&self) -> usize {
        self.inner.len
    }
}

/// Converts owned or borrowed encoding storage to a shared backend-native view.
pub trait CKKSEncodingBufferToBackendRef<BE: Backend, F>: CKKSEncodingBufferInfos {
    fn to_backend_ref(&self) -> CKKSEncodingBufferBackendRef<'_, BE, F>;
}

/// Converts owned or scratch-borrowed encoding storage to a mutable backend-native view.
pub trait CKKSEncodingBufferToBackendMut<BE: Backend, F>: CKKSEncodingBufferToBackendRef<BE, F> {
    fn to_backend_mut(&mut self) -> CKKSEncodingBufferBackendMut<'_, BE, F>;
}

impl<BE: Backend, F> CKKSEncodingBufferToBackendRef<BE, F> for CKKSEncodingBuffer<BE::OwnedBuf, F> {
    fn to_backend_ref(&self) -> CKKSEncodingBufferBackendRef<'_, BE, F> {
        CKKSEncodingBuffer::from_data(BE::view(&self.data), self.len)
    }
}

impl<BE: Backend, F> CKKSEncodingBufferToBackendMut<BE, F> for CKKSEncodingBuffer<BE::OwnedBuf, F> {
    fn to_backend_mut(&mut self) -> CKKSEncodingBufferBackendMut<'_, BE, F> {
        CKKSEncodingBuffer::from_data(BE::view_mut(&mut self.data), self.len)
    }
}

impl<'a, BE: Backend + 'a, F> CKKSEncodingBufferToBackendRef<BE, F> for CKKSEncodingBufferViewMut<'a, BE, F> {
    fn to_backend_ref(&self) -> CKKSEncodingBufferBackendRef<'_, BE, F> {
        CKKSEncodingBuffer::from_data(BE::view_ref_mut(&self.inner.data), self.inner.len)
    }
}

impl<'a, BE: Backend + 'a, F> CKKSEncodingBufferToBackendMut<BE, F> for CKKSEncodingBufferViewMut<'a, BE, F> {
    fn to_backend_mut(&mut self) -> CKKSEncodingBufferBackendMut<'_, BE, F> {
        CKKSEncodingBuffer::from_data(BE::view_mut_ref(&mut self.inner.data), self.inner.len)
    }
}

/// Explicit host-to-backend transfer into an owned or arena-borrowed encoding
/// buffer.
pub(crate) fn copy_host_into_encoding_buffer<BE, F, C>(dst: &mut C, values: &[F]) -> Result<()>
where
    BE: Backend,
    F: Pod,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    ensure!(dst.len() == values.len(), "encoding buffer length does not match host input");
    let mut dst = dst.to_backend_mut();
    BE::copy_host_to_view(&mut dst.data, bytemuck::cast_slice(values));
    Ok(())
}

/// Explicit backend-to-host transfer from an owned or arena-borrowed encoding
/// buffer.
pub(crate) fn copy_encoding_buffer_into_host<BE, F, C>(src: &C, values: &mut [F]) -> Result<()>
where
    BE: Backend,
    F: Pod,
    C: CKKSEncodingBufferToBackendRef<BE, F>,
{
    ensure!(src.len() == values.len(), "encoding buffer length does not match host output");
    let src = src.to_backend_ref();
    BE::copy_view_to_host(&src.data, bytemuck::cast_slice_mut(values));
    Ok(())
}

/// Explicitly uploads separate real and imaginary host slices into the planar
/// backend layout `[re | im]`.
pub(crate) fn copy_reim_host_into_encoding_buffer<BE, F, C>(dst: &mut C, re: &[F], im: &[F]) -> Result<()>
where
    BE: Backend,
    F: Pod,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    ensure!(re.len() == im.len(), "real and imaginary slot counts differ");
    let len = re
        .len()
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("CKKS slot count overflows usize"))?;
    ensure!(dst.len() == len, "encoding buffer must contain twice the slot count");
    let half_bytes = CKKSEncodingBuffer::<BE::BufMut<'_>, F>::bytes_of(re.len());
    let mut dst = dst.to_backend_mut();
    {
        let mut re_dst = BE::region_mut_ref(&mut dst.data, 0, half_bytes);
        BE::copy_host_to_view(&mut re_dst, bytemuck::cast_slice(re));
    }
    {
        let mut im_dst = BE::region_mut_ref(&mut dst.data, half_bytes, half_bytes);
        BE::copy_host_to_view(&mut im_dst, bytemuck::cast_slice(im));
    }
    Ok(())
}

/// Explicitly downloads planar backend slots `[re | im]` into separate host
/// slices.
pub(crate) fn copy_encoding_buffer_into_reim_host<BE, F, C>(src: &C, re: &mut [F], im: &mut [F]) -> Result<()>
where
    BE: Backend,
    F: Pod,
    C: CKKSEncodingBufferToBackendRef<BE, F>,
{
    ensure!(re.len() == im.len(), "real and imaginary slot counts differ");
    let len = re
        .len()
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("CKKS slot count overflows usize"))?;
    ensure!(src.len() == len, "encoding buffer must contain twice the slot count");
    let half_bytes = CKKSEncodingBuffer::<BE::BufRef<'_>, F>::bytes_of(re.len());
    let src = src.to_backend_ref();
    let re_src = BE::region_ref(&src.data, 0, half_bytes);
    BE::copy_view_to_host(&re_src, bytemuck::cast_slice_mut(re));
    let im_src = BE::region_ref(&src.data, half_bytes, half_bytes);
    BE::copy_view_to_host(&im_src, bytemuck::cast_slice_mut(im));
    Ok(())
}
