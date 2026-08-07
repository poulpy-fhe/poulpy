//! Public CKKS encoding operations.
//!
//! The four primitive operations use a backend-resident scalar buffer. Host
//! slices are convenience ingress/egress only and are staged through the
//! caller's standard backend arena. This lets a device backend compose its
//! native FFT and plaintext codec without a device-to-host round-trip.

use crate::{CKKSResult as Result, ckks_ensure};
use bytemuck::Pod;
use num_traits::FloatConst;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef,
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef, CKKSScalar, ScratchArenaTakeCKKS,
        copy_encoding_buffer_into_host, copy_encoding_buffer_into_reim_host, copy_host_into_encoding_buffer,
        copy_reim_host_into_encoding_buffer,
    },
};

/// Scalar types accepted by CKKS encoding operations.
///
/// `Pod` is required because host/device transfers preserve the scalar's byte
/// representation. Backends still opt into each precision independently by
/// implementing [`CKKSEncodingImpl`](crate::oep::CKKSEncodingImpl).
pub trait CKKSEncodingScalar: CKKSScalar + FloatConst + Pod + Send + Sync + 'static {}

impl<T> CKKSEncodingScalar for T where T: CKKSScalar + FloatConst + Pod + Send + Sync + 'static {}

/// Backend-resident CKKS encoding operations at scalar precision `F`.
///
/// Every scalar operand is a backend buffer. Host ingress and egress live in
/// [`CKKSEncodingHostOps`], so device code can use this trait without exposing
/// Rust slices or introducing an implicit transfer.
///
/// The scalar is a trait parameter so the methods stay free of backend bounds:
/// the delegating impl on `Module<BE>` requires the
/// [`CKKSEncodingImpl<BE, F>`](crate::oep::CKKSEncodingImpl) seam at the impl
/// level, and a backend overrides that seam independently of any bounds the
/// reference implementation carries.
pub trait CKKSEncodingOps<BE: Backend, F: CKKSEncodingScalar> {
    /// Destructively encodes a planar backend slot buffer `[re | im]` into a
    /// plaintext by composing the slot→coefficient transform and coefficient
    /// mapping.
    fn ckks_encode_slots_assign_into<P, C>(&self, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        self.ckks_slots_to_coeffs_assign(slots)?;
        self.ckks_encode_coeffs_into(pt, slots)
    }

    /// Decodes a plaintext into a planar backend slot buffer `[re | im]` by
    /// composing coefficient mapping and the coefficient→slot transform.
    fn ckks_decode_slots_into<P, C>(&self, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        self.ckks_decode_coeffs_into(pt, slots)?;
        self.ckks_coeffs_to_slots_assign(slots)
    }

    /// Quantizes backend-resident polynomial coefficients without an IFFT.
    fn ckks_encode_coeffs_into<P, C>(&self, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendRef<BE, F>;

    /// Dequantizes a plaintext into backend-resident polynomial coefficients.
    fn ckks_decode_coeffs_into<P, C>(&self, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// In-place planar complex slots → polynomial coefficients.
    fn ckks_slots_to_coeffs_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// In-place polynomial coefficients → planar complex slots.
    fn ckks_coeffs_to_slots_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>;
}

/// Host-slice adapters over [`CKKSEncodingOps`].
///
/// These methods make the host/backend transfer boundary explicit. They stage
/// through the caller's standard backend arena and delegate all transforms and
/// plaintext mapping to the backend-resident operations above.
pub trait CKKSEncodingHostOps<BE: Backend, F: CKKSEncodingScalar>: CKKSEncodingOps<BE, F> {
    /// Standard-arena bytes required by a host-slice encode/decode adapter.
    fn ckks_reim_tmp_bytes(&self, slots: usize) -> usize {
        let len = slots.checked_mul(2).expect("CKKS slot count overflow");
        CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(len)
    }

    /// Encodes host complex slots through one backend-native arena buffer.
    fn ckks_encode_reim_into<P>(&self, pt: &mut P, re: &[F], im: &[F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
    {
        ckks_ensure!(re.len() == im.len(), "real and imaginary slot counts differ");
        ckks_ensure!(
            !re.is_empty() && re.len().is_power_of_two(),
            "slot count must be a non-zero power of two, got {}",
            re.len()
        );
        let len = re
            .len()
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("CKKS slot count overflows usize"))?;
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(len);
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch.scope(|arena| {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(len);
            copy_reim_host_into_encoding_buffer::<BE, F, _>(&mut values, re, im)?;
            self.ckks_encode_slots_assign_into(pt, &mut values)
        })
    }

    /// Decodes a plaintext to host complex slots through one backend-native
    /// arena buffer.
    fn ckks_decode_reim_into<P>(&self, pt: &P, re: &mut [F], im: &mut [F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    {
        ckks_ensure!(re.len() == im.len(), "real and imaginary slot counts differ");
        ckks_ensure!(
            !re.is_empty() && re.len().is_power_of_two(),
            "slot count must be a non-zero power of two, got {}",
            re.len()
        );
        let len = re
            .len()
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("CKKS slot count overflows usize"))?;
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(len);
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS decoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch
            .scope(|arena| {
                let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(len);
                self.ckks_decode_slots_into(pt, &mut values)?;
                copy_encoding_buffer_into_reim_host::<BE, F, _>(&values, re, im)
            })
            .map_err(Into::into)
    }

    /// Quantizes host polynomial coefficients without applying an IFFT.
    fn ckks_encode_coeffs_host_into<P>(&self, pt: &mut P, coeffs: &[F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
    {
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(coeffs.len());
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS coefficient encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch.scope(|arena| {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(coeffs.len());
            copy_host_into_encoding_buffer::<BE, F, _>(&mut values, coeffs)?;
            self.ckks_encode_coeffs_into(pt, &values)
        })
    }

    /// Dequantizes polynomial coefficients to a host slice without an FFT.
    fn ckks_decode_coeffs_host_into<P>(&self, pt: &P, coeffs: &mut [F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    {
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(coeffs.len());
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS coefficient decoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch
            .scope(|arena| {
                let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(coeffs.len());
                self.ckks_decode_coeffs_into(pt, &mut values)?;
                copy_encoding_buffer_into_host::<BE, F, _>(&values, coeffs)
            })
            .map_err(Into::into)
    }
}

impl<BE, F, T> CKKSEncodingHostOps<BE, F> for T
where
    BE: Backend,
    F: CKKSEncodingScalar,
    T: CKKSEncodingOps<BE, F>,
{
}
