//! Canonical CKKS slot ordering and coefficient quantization.
//!
//! These host reference routines define the scheme math independently of backend
//! plan caches and FFT bindings. Backends own their resident implementations and
//! may reuse these routines when host storage is available. PaCo and SHIP
//! embeddings are defined by the explicitly named host helpers re-exported here.

use anyhow::{Context, Result, ensure};
use num_traits::NumCast;
use poulpy_core::layouts::{GLWEInfos, IntPolyInfos, LWEInfos};
use poulpy_hal::{
    GALOISGENERATOR,
    api::NegacyclicFFT,
    layouts::{Backend, HostDataMut, HostDataRef},
};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef,
    api::CKKSEncodingScalar,
    layouts::{CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef},
};

pub use crate::encoding::{paco_coeff_encodings_host, ship_coeff_encodings_host};

/// Canonical CKKS slot permutation for one transform dimension.
///
/// A backend may cache this scheme geometry alongside its own transform plans.
/// The host reference methods compose this permutation with a caller-selected
/// HAL transform; resident encoding implementations reproduce the same map.
pub struct EncodingPermutation {
    slots: usize,
    slot_scatter_swaps: Vec<(usize, usize)>,
}

impl EncodingPermutation {
    pub fn new(slots: usize) -> Result<Self> {
        ensure!(
            slots > 0 && slots.is_power_of_two(),
            "slot count must be a non-zero power of two"
        );
        let two_n = slots.checked_mul(4).context("CKKS slot geometry overflows usize")?;
        let log_n = (2 * slots).trailing_zeros();
        let mut slot_map = Vec::with_capacity(slots);
        let mut exponent = 1usize;
        for _ in 0..slots {
            slot_map.push(((exponent - 1) / 2).reverse_bits() >> (usize::BITS - log_n));
            exponent = (exponent * GALOISGENERATOR as usize) & (two_n - 1);
        }

        let mut seen = vec![false; slots];
        let mut slot_scatter_swaps = Vec::new();
        for start in 0..slots {
            if seen[start] {
                continue;
            }
            let mut current = start;
            seen[current] = true;
            loop {
                let next = slot_map[current];
                if next == start {
                    break;
                }
                slot_scatter_swaps.push((start, next));
                current = next;
                assert!(!seen[current], "CKKS slot map is not a permutation");
                seen[current] = true;
            }
        }
        Ok(Self {
            slots,
            slot_scatter_swaps,
        })
    }

    pub fn slots_to_coeffs_assign<F, T>(&self, fft: &T, values: &mut [F]) -> Result<()>
    where
        F: CKKSEncodingScalar + NumCast,
        T: NegacyclicFFT<F>,
    {
        ensure!(values.len() == 2 * self.slots);
        ensure!(fft.m() == self.slots);
        for &(a, b) in &self.slot_scatter_swaps {
            values.swap(a, b);
            values.swap(self.slots + a, self.slots + b);
        }
        fft.ifft(values);
        let inv_slots = F::from(self.slots)
            .context("slot count is not representable by the encoding scalar")?
            .recip();
        values.iter_mut().for_each(|value| *value = *value * inv_slots);
        Ok(())
    }

    pub fn coeffs_to_slots_assign<F, T>(&self, fft: &T, values: &mut [F]) -> Result<()>
    where
        F: CKKSEncodingScalar,
        T: NegacyclicFFT<F>,
    {
        ensure!(values.len() == 2 * self.slots);
        ensure!(fft.m() == self.slots);
        fft.fft(values);
        for &(a, b) in self.slot_scatter_swaps.iter().rev() {
            values.swap(a, b);
            values.swap(self.slots + a, self.slots + b);
        }
        Ok(())
    }
}

fn coefficient_gap<P>(pt: &P, coeff_count: usize) -> Result<usize>
where
    P: GLWEInfos,
{
    let n = pt.n().as_usize();
    ensure!(pt.rank().as_usize() == 0, "CKKS plaintext encoding expects rank zero");
    ensure!(coeff_count > 0, "coefficient count must be non-zero");
    ensure!(
        coeff_count <= n && n.is_multiple_of(coeff_count),
        "coefficient count must divide plaintext degree"
    );
    let gap = n / coeff_count;
    ensure!(gap.is_power_of_two(), "coefficient gap must be a power of two");
    Ok(gap)
}

/// Host reference quantization into canonical integer coefficients at the plaintext scale.
///
/// Coefficients are rounded to the nearest integer, with halfway cases away from
/// zero. Sparse inputs occupy every `degree / coeff_count` coefficient; the
/// remaining coefficients are zero. Invalid shape or non-representable scalar
/// inputs are rejected before writing the plaintext. The input and metadata are
/// preserved. Host-access bounds are specific to this reference helper, not the
/// resident encoding extension point.
pub fn encode_coeffs_into_host<BE, F, P>(pt: &mut P, coeffs: &CKKSEncodingBufferBackendRef<'_, BE, F>) -> Result<()>
where
    BE: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar + NumCast,
    P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let coeffs = coeffs.as_slice();
    let gap = coefficient_gap(pt, coeffs.len())?;
    let log_delta = pt.log_delta();
    let log_budget = pt.log_budget();
    let scale = F::from_usize(log_delta)
        .context("CKKS plaintext scale exponent is not representable by the codec scalar")?
        .exp2();
    let narrow = log_delta + log_budget <= 63;
    let limit = F::from_usize(if narrow { 63 } else { 127 })
        .context("CKKS coefficient range is not representable by the codec scalar")?
        .exp2();
    let quantize = |value: F| -> Result<F> {
        let rounded = (value * scale).round();
        // Scalar conversion implementations may saturate instead of returning
        // None. Check the signed half-open range explicitly, which also rejects
        // NaN and infinities before any destination coefficient is written.
        ensure!(
            rounded >= -limit && rounded < limit,
            "CKKS coefficient is outside the codec integer range"
        );
        Ok(rounded)
    };
    let base2k = pt.base2k().as_usize();
    let k = pt.encoded_k().as_usize();
    let mut backend = pt.to_backend_mut();

    if narrow {
        let data: Vec<i64> = coeffs
            .iter()
            .map(|&x| {
                quantize(x)?
                    .to_i64()
                    .context("CKKS coefficient is not representable as an i64 at the plaintext scale")
            })
            .collect::<Result<_>>()?;
        backend.data_mut().encode_vec_i64_strided(base2k, 0, k, gap, &data);
    } else {
        let data: Vec<i128> = coeffs
            .iter()
            .map(|&x| {
                quantize(x)?
                    .to_i128()
                    .context("CKKS coefficient is not representable as an i128 at the plaintext scale")
            })
            .collect::<Result<_>>()?;
        backend.data_mut().encode_vec_i128_strided(base2k, 0, k, gap, &data);
    }
    Ok(())
}

/// Host reference dequantization of canonical integer coefficients.
///
/// The caller selects the coefficient count and therefore the sparse stride.
/// The plaintext and its metadata are preserved.
pub fn decode_coeffs_into_host<BE, F, P>(pt: &P, coeffs: &mut CKKSEncodingBufferBackendMut<'_, BE, F>) -> Result<()>
where
    BE: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let coeffs = coeffs.as_mut_slice();
    let gap = coefficient_gap(pt, coeffs.len())?;
    let log_delta = pt.log_delta();
    let log_budget = pt.log_budget();
    ensure!(
        log_delta + log_budget <= 127,
        "CKKS host decoding supports at most 127 torus bits"
    );
    let scale =
        (-F::from_usize(log_delta).context("CKKS plaintext scale exponent is not representable by the codec scalar")?).exp2();
    let base2k = pt.base2k().as_usize();
    let k = pt.encoded_k().as_usize();
    let backend = pt.to_backend_ref();

    if log_delta + log_budget <= 63 {
        let mut data = vec![0i64; coeffs.len()];
        backend.data().decode_vec_i64_strided(base2k, 0, k, gap, &mut data);
        for (coefficient, &value) in coeffs.iter_mut().zip(&data) {
            *coefficient =
                F::from_i64(value).context("decoded i64 coefficient is not representable by the codec scalar")? * scale;
        }
    } else {
        let mut data = vec![0i128; coeffs.len()];
        backend.data().decode_vec_i128_strided(base2k, 0, k, gap, &mut data);
        for (coefficient, &value) in coeffs.iter_mut().zip(&data) {
            *coefficient =
                F::from_i128(value).context("decoded i128 coefficient is not representable by the codec scalar")? * scale;
        }
    }
    Ok(())
}
