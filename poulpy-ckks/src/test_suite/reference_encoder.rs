use std::fmt::Debug;

use anyhow::Result;
use num_traits::{Float, FloatConst, NumCast};
use poulpy_hal::{
    GALOISGENERATOR,
    api::{NegacyclicFFT, NegacyclicFFTNew},
};

use crate::{layouts::CKKSScalar, layouts::plaintext::CKKSPlaintextVecHostCodec};

/// Test-suite reference encoder/decoder for CKKS real and imaginary vectors.
///
/// The encoder maps `m` complex slots onto an RNX plaintext of size `2m`
/// through the canonical FFT/IFFT packing used by the rest of the crate.
///
/// `T` is the negacyclic FFT implementation (e.g. `FFT64ReimTable<f64>`
/// from `poulpy-cpu-ref`).
pub struct ReferenceEncoder<T> {
    table: T,
    plan: ReferenceEncodingPlan,
}

/// Private slot permutation used by the test-suite reference encoder.
///
/// Production encoding plans belong to each backend and are reached through
/// `CKKSEncodingImpl`; this reference plan exists only to provide an independent
/// host-side oracle for backend conformance tests.
struct ReferenceEncodingPlan {
    m: usize,
    slot_scatter_swaps: Vec<(usize, usize)>,
}

impl ReferenceEncodingPlan {
    fn new(m: usize) -> Result<Self> {
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        let slot_map = Self::build_slot_map(m);
        let slot_scatter_swaps = Self::build_scatter_swaps(&slot_map);
        Ok(Self { m, slot_scatter_swaps })
    }

    fn build_slot_map(m: usize) -> Vec<usize> {
        let two_n = 4 * m;
        let log_n = (2 * m).trailing_zeros();
        let mut slot_map = Vec::with_capacity(m);
        let mut exp = 1usize;
        for _ in 0..m {
            slot_map.push(((exp - 1) / 2).reverse_bits() >> (usize::BITS - log_n));
            exp = (exp * GALOISGENERATOR as usize) & (two_n - 1);
        }
        slot_map
    }

    /// Precomputes a swap program for `dst[slot_map[k]] = src[k]`.
    fn build_scatter_swaps(slot_map: &[usize]) -> Vec<(usize, usize)> {
        let mut seen = vec![false; slot_map.len()];
        let mut swaps = Vec::new();
        for start in 0..slot_map.len() {
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
                swaps.push((start, next));
                current = next;
                assert!(!seen[current], "CKKS slot map is not a permutation");
                seen[current] = true;
            }
        }
        swaps
    }

    fn scatter_slots_assign<F>(&self, values: &mut [F]) {
        let m = self.m;
        for &(a, b) in &self.slot_scatter_swaps {
            values.swap(a, b);
            values.swap(m + a, m + b);
        }
    }

    fn gather_slots_assign<F>(&self, values: &mut [F]) {
        let m = self.m;
        for &(a, b) in self.slot_scatter_swaps.iter().rev() {
            values.swap(a, b);
            values.swap(m + a, m + b);
        }
    }

    fn slots_to_coeffs_assign<F, T>(&self, table: &T, values: &mut [F]) -> Result<()>
    where
        F: Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
    {
        anyhow::ensure!(values.len() == 2 * self.m);
        anyhow::ensure!(table.m() == self.m);
        self.scatter_slots_assign(values);
        table.ifft(values);
        let inv_m = <F as NumCast>::from(self.m).unwrap().recip();
        values.iter_mut().for_each(|x| *x = *x * inv_m);
        Ok(())
    }

    fn coeffs_to_slots_assign<F, T>(&self, table: &T, values: &mut [F]) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        T: NegacyclicFFT<F>,
    {
        anyhow::ensure!(values.len() == 2 * self.m);
        anyhow::ensure!(table.m() == self.m);
        table.fft(values);
        self.gather_slots_assign(values);
        Ok(())
    }

    fn pack_reim_coeffs<F, T>(&self, table: &T, coeffs: &mut [F], re: &[F], im: &[F]) -> Result<()>
    where
        F: Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
    {
        anyhow::ensure!(coeffs.len() == 2 * self.m);
        anyhow::ensure!(table.m() == self.m);
        anyhow::ensure!(re.len() == self.m);
        anyhow::ensure!(im.len() == self.m);
        coeffs[..self.m].copy_from_slice(re);
        coeffs[self.m..].copy_from_slice(im);
        self.slots_to_coeffs_assign(table, coeffs)
    }

    fn unpack_reim_coeffs<F, T>(&self, table: &T, coeffs: &[F], re: &mut [F], im: &mut [F]) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        T: NegacyclicFFT<F>,
    {
        anyhow::ensure!(coeffs.len() == 2 * self.m);
        anyhow::ensure!(table.m() == self.m);
        anyhow::ensure!(re.len() == self.m);
        anyhow::ensure!(im.len() == self.m);
        let m = self.m;
        let mut reim_tmp = vec![F::zero(); coeffs.len()];
        reim_tmp.copy_from_slice(coeffs);
        self.coeffs_to_slots_assign(table, &mut reim_tmp)?;
        re.copy_from_slice(&reim_tmp[..m]);
        im.copy_from_slice(&reim_tmp[m..]);
        Ok(())
    }
}

impl<T> ReferenceEncoder<T> {
    /// Creates an encoder for `m` complex CKKS slots.
    ///
    /// Inputs:
    /// - `m`: number of complex slots
    ///
    /// Output:
    /// - an encoder configured for plaintext polynomials of size `2m`
    ///
    /// Errors:
    /// - returns an error if `m == 0` or if `m` is not a power of two
    pub fn new<F>(m: usize) -> Result<Self>
    where
        F: Float + FloatConst + Debug,
        T: NegacyclicFFTNew<F>,
    {
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        Ok(Self {
            table: <T as NegacyclicFFTNew<F>>::new(m),
            plan: ReferenceEncodingPlan::new(m)?,
        })
    }

    /// Creates an encoder from an already-constructed FFT table.
    ///
    /// Use this when the concrete FFT type isn't known at compile time
    /// (e.g. `T = Box<dyn NegacyclicFFT<F>>`).
    pub fn from_table(table: T, m: usize) -> Result<Self> {
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        Ok(Self {
            table,
            plan: ReferenceEncodingPlan::new(m)?,
        })
    }

    /// Cleartext encode direction on plain float slices: writes the ring
    /// element `[Re | Im]` whose complex coefficient pairing
    /// `c_j = coeffs[j] + i·coeffs[j+m]` has slot values `(re, im)` — the
    /// slot-order scatter plus an IFFT with the `1/m` normalization, no
    /// plaintext and no quantization. [`Self::encode_reim`] is this followed
    /// by the plaintext coefficient codec.
    pub fn pack_reim_coeffs<F>(&self, coeffs: &mut [F], re: &[F], im: &[F]) -> Result<()>
    where
        F: Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
    {
        self.plan.pack_reim_coeffs(&self.table, coeffs, re, im)
    }

    /// Cleartext decode direction on plain float slices: reads `coeffs` as the
    /// ring element `[Re | Im]` (complex coefficient pairing
    /// `c_j = coeffs[j] + i·coeffs[j+m]`) and writes its `m` complex slot
    /// values — an FFT plus the encoder's slot-order gather, no plaintext and
    /// no quantization. [`Self::decode_reim`] is the plaintext coefficient
    /// codec followed by this.
    pub fn unpack_reim_coeffs<F>(&self, coeffs: &[F], re: &mut [F], im: &mut [F]) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        T: NegacyclicFFT<F>,
    {
        self.plan.unpack_reim_coeffs(&self.table, coeffs, re, im)
    }

    /// Encodes complex slot values into a host-backed ZNX plaintext buffer.
    pub fn encode_reim<F, P>(&self, pt: &mut P, re: &[F], im: &[F]) -> Result<()>
    where
        F: CKKSScalar + Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
        P: CKKSPlaintextVecHostCodec<F>,
    {
        let m = self.table.m();
        let coeff_count = 2 * m;
        let n = pt.n().as_usize();
        anyhow::ensure!(
            coeff_count <= n && n.is_multiple_of(coeff_count),
            "encoded coefficient count {coeff_count} must divide plaintext degree {n}"
        );
        let gap = n / coeff_count;
        anyhow::ensure!(gap.is_power_of_two());
        let mut coeffs = vec![F::zero(); coeff_count];
        self.pack_reim_coeffs(&mut coeffs, re, im)?;
        pt.encode_host_floats(&coeffs)
    }

    /// Decodes a host-backed ZNX plaintext buffer into complex slot values.
    pub fn decode_reim<F, P>(&self, pt: &P, re: &mut [F], im: &mut [F]) -> Result<()>
    where
        F: CKKSScalar + Float + FloatConst + Debug,
        T: NegacyclicFFT<F>,
        P: CKKSPlaintextVecHostCodec<F>,
    {
        let m = self.table.m();
        let coeff_count = 2 * m;
        let n = pt.n().as_usize();
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);
        anyhow::ensure!(
            coeff_count <= n && n.is_multiple_of(coeff_count),
            "decoded coefficient count {coeff_count} must divide plaintext degree {n}"
        );
        let gap = n / coeff_count;
        anyhow::ensure!(gap.is_power_of_two());
        let mut coeffs = vec![F::zero(); coeff_count];
        pt.decode_host_floats(&mut coeffs)?;
        self.unpack_reim_coeffs(&coeffs, re, im)
    }
}
