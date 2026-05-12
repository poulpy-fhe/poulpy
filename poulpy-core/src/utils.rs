use crate::layouts::{GLWEPlaintext, LWEInfos, LWEPlaintext, TorusPrecision};
use dashu_float::{FBig, round::mode::HalfEven};
use poulpy_hal::layouts::{HostDataMut, HostDataRef, Stats};

impl<D: HostDataMut> GLWEPlaintext<D> {
    /// Encodes a slice of `i64` values into the plaintext's coefficient slots.
    ///
    /// Values are scaled by `2^k` and decomposed into the base-2k limb
    /// representation. The slice length must not exceed the ring degree *N*.
    pub fn encode_vec_i64(&mut self, data: &[i64], k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data.encode_vec_i64(base2k, 0, k.into(), data);
    }

    pub fn encode_vec_i128(&mut self, data: &[i128], k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data.encode_vec_i128(base2k, 0, k.into(), data);
    }

    /// Encodes a single `i64` value into coefficient slot `idx`.
    pub fn encode_coeff_i64(&mut self, data: i64, k: TorusPrecision, idx: usize) {
        let base2k: usize = self.base2k().into();
        self.data.encode_coeff_i64(base2k, 0, k.into(), idx, data);
    }
}

impl<D: HostDataRef> GLWEPlaintext<D> {
    /// Decodes the plaintext coefficients into a slice of `i64` values
    /// using `k` bits of torus precision.
    pub fn decode_vec_i64(&self, data: &mut [i64], k: TorusPrecision) {
        self.data.decode_vec_i64(self.base2k().into(), 0, k.into(), data);
    }

    pub fn decode_vec_i128(&self, data: &mut [i128], k: TorusPrecision) {
        self.data.decode_vec_i128(self.base2k().into(), 0, k.into(), data);
    }

    /// Decodes a single coefficient at slot `idx` as an `i64`.
    pub fn decode_coeff_i64(&self, k: TorusPrecision, idx: usize) -> i64 {
        self.data.decode_coeff_i64(self.base2k().into(), 0, k.into(), idx)
    }

    /// Decodes the plaintext coefficients into arbitrary-precision floats.
    pub fn decode_vec_float(&self, data: &mut [FBig<HalfEven>]) {
        self.data.decode_vec_float(self.base2k().into(), 0, data);
    }

    /// Returns per-coefficient statistics (min, max, mean, variance) of the plaintext.
    pub fn stats(&self) -> Stats {
        self.data.stats(self.base2k().into(), 0)
    }
}

impl<D: HostDataMut> LWEPlaintext<D> {
    /// Encodes a single `i64` value into the LWE plaintext scalar slot.
    pub fn encode_i64(&mut self, data: i64, k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data.encode_coeff_i64(base2k, 0, k.into(), 0, data);
    }

    /// Encodes a single `i128` value into the LWE plaintext scalar slot.
    pub fn encode_i128(&mut self, data: i128, k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data.encode_vec_i128(base2k, 0, k.into(), &[data]);
    }
}

impl<D: HostDataRef> LWEPlaintext<D> {
    /// Decodes the LWE plaintext scalar as an `i64` using `k` bits of torus precision.
    pub fn decode_i64(&self, k: TorusPrecision) -> i64 {
        self.data.decode_coeff_i64(self.base2k().into(), 0, k.into(), 0)
    }

    /// Decodes the LWE plaintext scalar as an arbitrary-precision float.
    pub fn decode_float(&self) -> FBig<HalfEven> {
        let mut out = [FBig::<HalfEven>::ZERO];
        self.data.decode_vec_float(self.base2k().into(), 0, &mut out);
        out[0].clone()
    }
}
