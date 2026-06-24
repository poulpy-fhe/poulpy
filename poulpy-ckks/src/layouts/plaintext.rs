use std::{
    fmt::{self},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Compact, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBSGSMeta, SetBase2k, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

use super::CKKSScalar;

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintext<D: Data = Vec<u8>> {
    /// Raw GLWE plaintext limb storage.
    pub(crate) inner: GLWEPlaintext<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
}

impl<D: Data> CKKSPlaintext<D> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    /// Rebuilds this backend-owned plaintext as a host-owned [`CKKSPlaintext<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> CKKSPlaintext<Vec<u8>>
    where
        BE: Backend<OwnedBuf = D>,
    {
        CKKSPlaintext::from_inner(self.inner.to_host_owned::<BE>(), self.meta)
    }

    /// Formats this backend-owned plaintext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D>,
    {
        self.to_host_owned::<BE>().to_string()
    }

    /// Replaces the semantic metadata after checking that the current storage
    /// can represent it.
    ///
    /// This is intended for callers that build plaintext buffers manually.
    /// Normal CKKS operations update metadata themselves.
    pub fn set_meta_checked(&mut self, meta: CKKSMeta) -> Result<()> {
        anyhow::ensure!(
            self.k().as_usize() <= self.max_k().as_usize() && meta.log_delta <= self.k().as_usize(),
            crate::CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta,
                base2k: self.base2k().as_usize(),
                requested_limbs: self.max_size(),
            }
        );
        self.meta = meta;
        Ok(())
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSPlaintext<D>
where
    GLWEPlaintext<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSPlaintext<D>
where
    GLWEPlaintext<D>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }
}

impl<D: Data> Deref for CKKSPlaintext<D> {
    type Target = GLWEPlaintext<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data> DerefMut for CKKSPlaintext<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data> LWEInfos for CKKSPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<D: Data> GLWEInfos for CKKSPlaintext<D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data> SetCKKSInfos for CKKSPlaintext<D> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        poulpy_core::layouts::SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data> Compact for CKKSPlaintext<D> {
    fn compact(&mut self) {
        //let limbs = self.k().div_ceil(self.base2k().as_usize()).max(1).min(self.size());
        //self.inner.data().set_size(limbs);
    }
}

impl<D: Data> SetBSGSMeta for CKKSPlaintext<D> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: HostDataMut> SetBase2k for CKKSPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.set_base2k(base2k);
    }
}

impl<D: HostDataRef> fmt::Display for CKKSPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D: Data> CKKSInfos for CKKSPlaintext<D> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta
    }

    fn log_budget(&self) -> usize {
        // Derived from the wrapped GLWE plaintext's torus width: `k - log_delta`.
        self.inner.k().as_usize().saturating_sub(self.meta.log_delta)
    }
}

impl<D: Data> BSGSMeta for CKKSPlaintext<D> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

pub trait CKKSPlaintextVecHostCodec<F: CKKSScalar>: CKKSInfos + LWEInfos {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()>;
    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()>;

    /// Sparse (gap-interleaved) coefficient encoding.
    ///
    /// `coeffs` has length `L = 2·slots` — a power of two that divides `n` — laid
    /// out as `re(slots) || im(slots)`. It is spread into the `n` polynomial
    /// coefficients with gap `g = n / L`: `re[i]` lands at coefficient `i·g` and
    /// `im[i]` at `n/2 + i·g`; all other coefficients are zero. This is how a
    /// sparsely-packed (`logSlots < logMaxSlots`) vector is encoded coefficient-wise
    /// for the homomorphic DFT. For full packing (`L == n`, `g == 1`) it reduces
    /// to the dense placement `re || im`.
    fn encode_host_floats_sparse(&mut self, coeffs: &[F]) -> Result<()>;

    /// Inverse of [`Self::encode_host_floats_sparse`]: reads the gap-interleaved
    /// coefficients of `self` back into `coeffs` (length `L = 2·slots`).
    fn decode_host_floats_sparse(&self, coeffs: &mut [F]) -> Result<()>;
}

/// Validates a sparse `coeffs` length (`L = 2·slots`, a power of two dividing `n`,
/// laid out `re(slots) || im(slots)`) and returns the coefficient gap `g = n / L`.
/// With this gap, `coeffs[j]` lands at coefficient `j·g`, so the real parts fall in
/// `[0, n/2)` and the imaginary parts in `[n/2, n)` (since `slots·g = n/2`).
fn sparse_gap<F>(coeffs: &[F], n: usize) -> Result<usize> {
    let l = coeffs.len();
    anyhow::ensure!(
        l >= 2 && l.is_power_of_two(),
        "sparse coeffs length must be a power of two ≥ 2, got {l}"
    );
    anyhow::ensure!(l <= n && n.is_multiple_of(l), "sparse coeffs length {l} must divide n {n}");
    Ok(n / l)
}

/// Quantizes `coeffs` (length `N/gap`) and encodes them gap-strided directly into
/// the plaintext via `encode_vec_i*_strided` — the quantized integer buffer is only
/// `coeffs.len()` long, so the sparse path never materializes a length-`N` buffer.
/// `gap == 1` is the dense path (`coeffs.len() == N`).
fn encode_host_floats_strided<F, D>(pt: &mut CKKSPlaintext<D>, coeffs: &[F], gap: usize) -> Result<()>
where
    F: CKKSScalar,
    D: HostDataMut + HostDataRef,
{
    let log_delta = pt.log_delta();
    let log_budget = pt.log_budget();
    // The encoding scale (`log_delta`) is decoupled from `F`'s mantissa width: a
    // scale wider than the float's precision is allowed, it simply yields values
    // accurate only to `F`'s mantissa. The integer-width checks below are the real
    // correctness guards (the quantized value must fit i64/i128).
    let scale = F::from_usize(log_delta).unwrap().exp2();
    let k = pt.max_k();
    if log_delta + log_budget <= 63 {
        let data: Vec<i64> = coeffs.iter().map(|&x| (x * scale).round().to_i64().unwrap()).collect();
        pt.encode_vec_i64_strided(gap, &data, k);
    } else {
        let data: Vec<i128> = coeffs.iter().map(|&x| (x * scale).round().to_i128().unwrap()).collect();
        pt.encode_vec_i128_strided(gap, &data, k);
    }
    Ok(())
}

/// Decodes the gap-strided coefficients of `pt` into `coeffs` (length `N/gap`),
/// inverse of [`encode_host_floats_strided`]. Only `coeffs.len()` coefficients are
/// reconstructed.
fn decode_host_floats_strided<F, D>(pt: &CKKSPlaintext<D>, coeffs: &mut [F], gap: usize) -> Result<()>
where
    F: CKKSScalar,
    D: HostDataRef,
{
    let log_delta = pt.log_delta();
    let log_budget = pt.log_budget();
    anyhow::ensure!(log_delta + log_budget <= 127);
    let scale = (-F::from_usize(log_delta).unwrap()).exp2();
    let k = pt.max_k();
    if log_delta + log_budget <= 63 {
        let mut data = vec![0i64; coeffs.len()];
        pt.decode_vec_i64_strided(gap, &mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::from_i64(i).unwrap() * scale;
        }
    } else {
        let mut data = vec![0i128; coeffs.len()];
        pt.decode_vec_i128_strided(gap, &mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::from_i128(i).unwrap() * scale;
        }
    }
    Ok(())
}

impl<F: CKKSScalar, D: HostDataMut + HostDataRef> CKKSPlaintextVecHostCodec<F> for CKKSPlaintext<D> {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()> {
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        encode_host_floats_strided(self, coeffs, 1)
    }

    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()> {
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        decode_host_floats_strided(self, coeffs, 1)
    }

    fn encode_host_floats_sparse(&mut self, coeffs: &[F]) -> Result<()> {
        let gap = sparse_gap(coeffs, self.n().as_usize())?;
        encode_host_floats_strided(self, coeffs, gap)
    }

    fn decode_host_floats_sparse(&self, coeffs: &mut [F]) -> Result<()> {
        let gap = sparse_gap(coeffs, self.n().as_usize())?;
        decode_host_floats_strided(self, coeffs, gap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::CKKSModuleAlloc;
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    #[test]
    fn plaintext_coeff_pack_allocates_requested_degree() {
        let module = Module::<HostBytesBackend>::new(16);
        let prec = CKKSMeta {
            log_sparsity: 0,
            log_delta: 40,
        };
        let base2k: Base2K = 52usize.into();

        let mut pt = module.ckks_pt_coeffs_alloc(3, base2k, (12usize + 40).into());
        pt.set_meta(prec);

        assert_eq!(pt.n().as_usize(), 3);
        assert_eq!(pt.base2k(), base2k);
        assert_eq!(pt.meta(), prec);
        assert!(pt.k() <= pt.max_k());
    }

    /// Sparse coefficient pack/unpack round-trips, and places values at the
    /// expected gap-spaced positions of the full coefficient buffer.
    #[test]
    fn sparse_coeff_pack_roundtrip_and_layout() {
        let n = 16usize;
        let module = Module::<HostBytesBackend>::new(n as u64);
        let prec = CKKSMeta {
            log_sparsity: 0,
            log_delta: 40,
        };
        let base2k: Base2K = 50usize.into();

        // L = 4 (slots = 2), so gap = n/L = 4.
        let small = vec![1.0_f64, 2.0, 3.0, 4.0]; // re=[1,2], im=[3,4]
        let mut pt = module.ckks_pt_vec_alloc(base2k, (10usize + 40).into());
        pt.set_meta(prec);
        pt.encode_host_floats_sparse(&small).unwrap();

        // Full coefficients: re at 0,4; im at n/2=8, 12; rest zero.
        let mut full = vec![0.0_f64; n];
        pt.decode_host_floats(&mut full).unwrap();
        let approx = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(approx(full[0], 1.0) && approx(full[4], 2.0), "re placement: {full:?}");
        assert!(approx(full[8], 3.0) && approx(full[12], 4.0), "im placement: {full:?}");
        for (i, &v) in full.iter().enumerate() {
            if ![0, 4, 8, 12].contains(&i) {
                assert!(approx(v, 0.0), "coeff {i} should be zero, got {v}");
            }
        }

        // Round-trip back to the small vector.
        let mut got = vec![0.0_f64; 4];
        pt.decode_host_floats_sparse(&mut got).unwrap();
        for (a, b) in small.iter().zip(&got) {
            assert!(approx(*a, *b), "sparse round-trip: {small:?} vs {got:?}");
        }
    }
}
