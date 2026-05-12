use std::fmt::Debug;

use anyhow::Result;
use poulpy_hal::{
    GALOISGENERATOR,
    api::{NegacyclicFFT, NegacyclicFFTNew},
};
use rand_distr::num_traits::{Float, FloatConst, NumCast};

use crate::{layouts::CKKSScalar, layouts::plaintext::CKKSPlaintextVecHostCodec};

/// Slot encoder/decoder for CKKS real and imaginary vectors.
///
/// The encoder maps `m` complex slots onto an RNX plaintext of size `2m`
/// through the canonical FFT/IFFT packing used by the rest of the crate.
///
/// `T` is the negacyclic FFT implementation (e.g. `FFT64ReimTable<f64>`
/// from `poulpy-cpu-ref`).
pub struct Encoder<T> {
    table: T,
    slot_map: Vec<usize>,
}

impl<T> Encoder<T> {
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
        let slot_map = Self::build_slot_map(m);
        Ok(Self {
            table: <T as NegacyclicFFTNew<F>>::new(m),
            slot_map,
        })
    }

    /// Creates an encoder from an already-constructed FFT table.
    ///
    /// Use this when the concrete FFT type isn't known at compile time
    /// (e.g. `T = Box<dyn NegacyclicFFT<F>>`).
    pub fn from_table(table: T, m: usize) -> Result<Self> {
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        let slot_map = Self::build_slot_map(m);
        Ok(Self { table, slot_map })
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

    fn pack_reim_coeffs<F>(&self, coeffs: &mut [F], re: &[F], im: &[F]) -> Result<()>
    where
        F: Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
    {
        let n = coeffs.len();
        let m = n / 2;
        anyhow::ensure!(self.table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);
        coeffs.fill(F::zero());
        for k in 0..m {
            let idx = self.slot_map[k];
            coeffs[idx] = re[k];
            coeffs[m + idx] = im[k];
        }
        self.table.ifft(coeffs);
        let inv_m = <F as NumCast>::from(m).unwrap().recip();
        coeffs.iter_mut().for_each(|x| *x = *x * inv_m);
        Ok(())
    }

    fn unpack_reim_coeffs<F>(&self, coeffs: &[F], re: &mut [F], im: &mut [F]) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        T: NegacyclicFFT<F>,
    {
        let n = coeffs.len();
        let m = n / 2;
        anyhow::ensure!(self.table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);
        let mut reim_tmp = vec![F::zero(); n];
        reim_tmp.copy_from_slice(coeffs);
        self.table.fft(&mut reim_tmp);
        for k in 0..m {
            let idx = self.slot_map[k];
            re[k] = reim_tmp[idx];
            im[k] = reim_tmp[m + idx];
        }
        Ok(())
    }

    /// Encodes complex slot values into a host-backed ZNX plaintext buffer.
    pub fn encode_reim<F, P>(&self, pt: &mut P, re: &[F], im: &[F]) -> Result<()>
    where
        F: CKKSScalar + Float + FloatConst + Debug + NumCast,
        T: NegacyclicFFT<F>,
        P: CKKSPlaintextVecHostCodec<F>,
    {
        let n = pt.n().as_usize();
        let mut coeffs = vec![F::zero(); n];
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
        let n = pt.n().as_usize();
        anyhow::ensure!(re.len() == n / 2);
        anyhow::ensure!(im.len() == n / 2);
        let mut coeffs = vec![F::zero(); n];
        pt.decode_host_floats(&mut coeffs)?;
        self.unpack_reim_coeffs(&coeffs, re, im)
    }
}
