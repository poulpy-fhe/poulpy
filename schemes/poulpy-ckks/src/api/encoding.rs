//! Public CKKS encoding operations.
//!
//! The four primitive operations use a backend-resident scalar buffer. Host
//! slices are convenience ingress/egress only and are staged through the
//! caller's standard backend arena. This lets a device backend compose its
//! native FFT and plaintext codec without a device-to-host round-trip.

use crate::{CKKSResult as Result, ckks_ensure};
use anyhow::Context;
use bytemuck::Pod;
use num_traits::FloatConst;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Ring, ScratchArena};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, GLWEPlaintextInfos, SetCKKSInfos, SlotLayout,
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef, CKKSScalar, ScratchArenaTakeCKKS,
        copy_encoding_buffer_into_host, copy_encoding_buffer_into_reim_host, copy_host_into_encoding_buffer,
        copy_reim_host_into_encoding_buffer, download_from_encoding_buffer, upload_into_encoding_buffer,
    },
    shaping::ShapingRing,
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
/// A conjugate invariant module packs up to `N` real slots. Encoding discards
/// imaginary inputs; decoding returns zero imaginary parts. Slot transforms
/// use `2m` scalars for `m` slots, with the independent invariant coefficients
/// in the first `m` entries. Every encoding fills its plaintext: `n`
/// coefficients for a plaintext of degree `n`, or all its
/// [`ckks_slots_of_degree`](crate::CKKSModuleInfos::ckks_slots_of_degree) slots.
///
/// Every scalar operand is a backend buffer. Host ingress and egress live in
/// [`CKKSEncodingHostOps`], so device code can use this trait without exposing
/// Rust slices or introducing an implicit transfer.
///
/// The scalar is a trait parameter so the methods stay free of backend bounds:
/// the delegating impl on `Module<BE>` requires the
/// [`CKKSEncodingImpl<F>`](crate::oep::CKKSEncodingImpl) seam at the impl
/// level, and a backend overrides that seam independently of any bounds the
/// reference implementation carries.
pub trait CKKSEncodingOps<BE: Backend, F: CKKSEncodingScalar> {
    /// Destructively encodes a planar backend slot buffer `[re | im]` into a
    /// plaintext by composing the slot→coefficient transform and coefficient
    /// mapping.
    fn ckks_encode_slots_assign_into<P, C>(&self, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// Decodes a plaintext into a planar backend slot buffer `[re | im]` by
    /// composing coefficient mapping and the coefficient→slot transform.
    fn ckks_decode_slots_into<P, C>(&self, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// Quantizes backend-resident polynomial coefficients without an IFFT.
    fn ckks_encode_coeffs_into<P, C>(&self, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendRef<BE, F>;

    /// Dequantizes a plaintext into backend-resident polynomial coefficients.
    fn ckks_decode_coeffs_into<P, C>(&self, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// In-place planar slots → polynomial coefficients.
    fn ckks_slots_to_coeffs_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>;

    /// In-place polynomial coefficients → planar slots.
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
    ///
    /// For a [`SlotLayout::Replicated`] plaintext, `re` and `im` hold the batch
    /// of `2^log_k` values, written to the data slots of `2^log_k · R` slots.
    fn ckks_encode_reim_into<P>(&self, pt: &mut P, re: &[F], im: &[F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    {
        if let SlotLayout::Replicated(layout) = pt.slot_layout() {
            ckks_ensure!(
                re.len() == 1 << layout.log_k() && im.len() == re.len(),
                "a replicated layout encodes {} values, got {} and {}",
                1usize << layout.log_k(),
                re.len(),
                im.len()
            );
            let plaintext_slots = slots_of_degree::<BE>(pt.n().as_usize());
            ckks_ensure!(
                (1usize << layout.log_k()) << layout.log_r() <= plaintext_slots,
                "{} values of period {} need more than the {plaintext_slots} slots of the plaintext",
                1usize << layout.log_k(),
                1usize << layout.log_r(),
            );
            let (period, slots) = (1usize << layout.log_r(), re.len() << layout.log_r());
            let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(2 * slots);
            ckks_ensure!(
                scratch.available() >= required,
                "CKKS encoding needs {required} scratch bytes, but only {} are available",
                scratch.available()
            );
            return scratch.scope(|arena| -> Result<()> {
                let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(2 * slots);
                upload_into_encoding_buffer::<BE, F, _>(&mut values, |t| {
                    let (part, s) = if t < slots { (re, t) } else { (im, t - slots) };
                    if s.is_multiple_of(period) {
                        part[s / period]
                    } else {
                        F::zero()
                    }
                });
                self.ckks_encode_slots_assign_into(pt, &mut values)
            });
        }
        self.ckks_encode_reim_slots_into(pt, re, im, scratch)
    }

    /// Encodes host complex slots through one backend-native arena buffer,
    /// every slot as given whatever the layout.
    fn ckks_encode_reim_slots_into<P>(&self, pt: &mut P, re: &[F], im: &[F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
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
        scratch.scope(|arena| -> Result<()> {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(len);
            copy_reim_host_into_encoding_buffer::<BE, F, _>(&mut values, re, im)?;
            self.ckks_encode_slots_assign_into(pt, &mut values)
        })
    }

    /// Decodes a plaintext to host complex slots through one backend-native
    /// arena buffer.
    ///
    /// For a [`SlotLayout::Replicated`] plaintext, `re` and `im` receive the
    /// batch of `2^log_k` values: reading `2^log_k · R` slots averages the
    /// replicas, and the data slots are returned.
    fn ckks_decode_reim_into<P>(&self, pt: &P, re: &mut [F], im: &mut [F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    {
        if let SlotLayout::Replicated(layout) = pt.slot_layout() {
            ckks_ensure!(
                re.len() == 1 << layout.log_k() && im.len() == re.len(),
                "a replicated layout decodes {} values, got {} and {}",
                1usize << layout.log_k(),
                re.len(),
                im.len()
            );
            let plaintext_slots = slots_of_degree::<BE>(pt.n().as_usize());
            ckks_ensure!(
                (1usize << layout.log_k()) << layout.log_r() <= plaintext_slots,
                "{} values of period {} need more than the {plaintext_slots} slots of the plaintext",
                1usize << layout.log_k(),
                1usize << layout.log_r(),
            );
            let (period, slots) = (1usize << layout.log_r(), re.len() << layout.log_r());
            let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(2 * slots);
            ckks_ensure!(
                scratch.available() >= required,
                "CKKS decoding needs {required} scratch bytes, but only {} are available",
                scratch.available()
            );
            return scratch.scope(|arena| -> Result<()> {
                let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(2 * slots);
                self.ckks_decode_slots_into(pt, &mut values)?;
                download_from_encoding_buffer::<BE, F, _>(&values, |t, v| {
                    let (part, s) = if t < slots { (&mut *re, t) } else { (&mut *im, t - slots) };
                    if s.is_multiple_of(period) {
                        part[s / period] = v;
                    }
                });
                Ok(())
            });
        }
        self.ckks_decode_reim_slots_into(pt, re, im, scratch)
    }

    /// Decodes a plaintext to host complex slots through one backend-native
    /// arena buffer, every slot whatever the layout.
    fn ckks_decode_reim_slots_into<P>(&self, pt: &P, re: &mut [F], im: &mut [F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
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
        scratch.scope(|arena| -> Result<()> {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(len);
            self.ckks_decode_slots_into(pt, &mut values)?;
            copy_encoding_buffer_into_reim_host::<BE, F, _>(&values, re, im).map_err(Into::into)
        })
    }

    /// Encodes the masked constant `(re + i·im)·π_S` of a
    /// [`SlotLayout::Replicated`] plaintext: `re + i·im` on every data slot and
    /// zero on the guards. `π_S`, one on the data class, lies in `R[Y]`,
    /// `Y = X^(N/(2R))`, with coefficients `cos(π·j/(2R))/R`, and `i·π_S = Y^R·π_S`,
    /// so the constant has `2R` coefficients, quantized with shaped rounding.
    /// `π_S` is invariant, so the conjugate-invariant ring, where `Y = X^(N/R)`,
    /// holds `re*π_S` by its first `R` coefficients and drops `im`, as its slot
    /// encoder does.
    fn ckks_encode_masked_constant_into<P>(&self, pt: &mut P, re: F, im: F, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    {
        let SlotLayout::Replicated(layout) = pt.slot_layout() else {
            return Err(anyhow::anyhow!("a masked constant needs a replicated slot layout").into());
        };
        let r = 1usize << layout.log_r();
        let order = layout.log_r() as u32 + 2;
        let r_inv = F::from_usize(r)
            .context("slot period is not representable by the encoding scalar")?
            .recip();
        let pi_s = |j: usize| F::ckks_root_of_unity(j as u64, order).0 * r_inv;
        let n = pt.n().as_usize();
        let period = match ShapingRing::of::<BE::Ring>() {
            ShapingRing::Standard => 2 * r,
            ShapingRing::ConjugateInvariant => r,
        };
        ckks_ensure!(
            n.is_multiple_of(period),
            "a masked constant of period {r} needs a degree multiple of {period}, got {n}"
        );
        let stride = n / period;
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(n);
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS coefficient encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch.scope(|arena| -> Result<()> {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(n);
            upload_into_encoding_buffer::<BE, F, _>(&mut values, |t| {
                if !t.is_multiple_of(stride) {
                    return F::zero();
                }
                let j = t / stride;
                if period == r {
                    return re * pi_s(j);
                }
                let shifted = if j >= r { pi_s(j - r) } else { -pi_s(j + r) };
                re * pi_s(j) + im * shifted
            });
            self.ckks_encode_coeffs_into(pt, &values)
        })
    }

    /// Quantizes host polynomial coefficients without applying an IFFT.
    fn ckks_encode_coeffs_host_into<P>(&self, pt: &mut P, coeffs: &[F], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    {
        let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(coeffs.len());
        ckks_ensure!(
            scratch.available() >= required,
            "CKKS coefficient encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        scratch.scope(|arena| -> Result<()> {
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
        scratch.scope(|arena| -> Result<()> {
            let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(coeffs.len());
            self.ckks_decode_coeffs_into(pt, &mut values)?;
            copy_encoding_buffer_into_host::<BE, F, _>(&values, coeffs).map_err(Into::into)
        })
    }
}

impl<BE, F, T> CKKSEncodingHostOps<BE, F> for T
where
    BE: Backend,
    F: CKKSEncodingScalar,
    T: CKKSEncodingOps<BE, F>,
{
}

/// Slots of a plaintext of degree `n` on the ring of `BE`: `n/2` complex
/// slots on the standard ring and `n` real ones on the conjugate-invariant ring.
fn slots_of_degree<BE: Backend>(n: usize) -> usize {
    n * BE::Ring::CYCLOTOMIC_ORDER_FACTOR as usize / 4
}
