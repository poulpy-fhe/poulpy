use poulpy_core::layouts::IntPolyInfos;
use std::{
    fmt::{self},
    ops::{Deref, DerefMut},
};

use anyhow::{Context, Result};
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextReborrowBackendMut, GLWEPlaintextReborrowBackendRef,
    GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetBSGSMeta, SetBase2k, SetK, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ZnxWord};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

use super::CKKSScalar;

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintext<D: Data, W: ZnxWord> {
    /// Raw GLWE plaintext limb storage.
    pub(crate) inner: GLWEPlaintext<D, W>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
}

impl<D: Data, W: ZnxWord> CKKSPlaintext<D, W> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D, W>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    /// Rebuilds this backend-owned plaintext as a host-owned [`CKKSPlaintext<Vec<u8>, i64>`].
    pub fn to_host_owned<BE>(&self) -> CKKSPlaintext<Vec<u8>, W>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        CKKSPlaintext::from_inner(self.inner.to_host_owned::<BE>(), self.meta)
    }

    /// Formats this backend-owned plaintext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
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

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSPlaintext<D, BE::ZnxWord>
where
    GLWEPlaintext<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSPlaintext<D, BE::ZnxWord>
where
    GLWEPlaintext<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }
}

/// Backend-owned CKKS plaintext: the backend's buffer type and its coefficient word.
pub type CKKSPlaintextOwned<BE> = CKKSPlaintext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

poulpy_core::view_wrapper!(
    /// Scratch-backed mutable CKKS plaintext view.
    ///
    /// This nominal wrapper contains an ordinary
    /// `CKKSPlaintext<BE::BufMut<'a>, BE::ZnxWord>`; both its polynomial storage and CKKS
    /// metadata therefore have exactly the same representation as other CKKS
    /// plaintexts.
    CKKSPlaintextViewMut,
    CKKSPlaintext<BE::BufMut<'a>, BE::ZnxWord>
);
poulpy_core::impl_glwe_infos!(CKKSPlaintextViewMut);
crate::impl_ckks_infos!(inner_meta CKKSPlaintextViewMut);

impl<'a, BE: poulpy_hal::layouts::Backend + 'a> poulpy_core::layouts::IntPolyInfos for CKKSPlaintextViewMut<'a, BE> {
    fn encoded_k(&self) -> TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<'a, BE: Backend + 'a> SetBase2k for CKKSPlaintextViewMut<'a, BE> {
    fn set_base2k(&mut self, base2k: Base2K) {
        SetBase2k::set_base2k(&mut self.inner, base2k);
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for CKKSPlaintextViewMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        <GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord> as GLWEPlaintextReborrowBackendRef<BE>>::reborrow_backend_ref(
            &self.inner.inner,
        )
    }
}

impl<'a, BE: Backend + 'a> GLWEToBackendMut<BE> for CKKSPlaintextViewMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        <GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord> as GLWEPlaintextReborrowBackendMut<BE>>::reborrow_backend_mut(
            &mut self.inner.inner,
        )
    }
}

impl<D: Data, W: ZnxWord> Deref for CKKSPlaintext<D, W> {
    type Target = GLWEPlaintext<D, W>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, W: ZnxWord> DerefMut for CKKSPlaintext<D, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for CKKSPlaintext<D, W> {
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

impl<D: Data, W: ZnxWord> poulpy_core::layouts::IntPolyInfos for CKKSPlaintext<D, W> {
    fn encoded_k(&self) -> TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for CKKSPlaintext<D, W> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord> SetCKKSInfos for CKKSPlaintext<D, W> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }

    fn set_k(&mut self, k: TorusPrecision) {
        poulpy_core::layouts::SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord> SetK for CKKSPlaintext<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord> SetBSGSMeta for CKKSPlaintext<D, W> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: Data, W: ZnxWord> SetBase2k for CKKSPlaintext<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.set_base2k(base2k);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for CKKSPlaintext<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D: Data, W: ZnxWord> CKKSInfos for CKKSPlaintext<D, W> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<D: Data, W: ZnxWord> BSGSMeta for CKKSPlaintext<D, W> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

pub trait CKKSPlaintextVecHostCodec<F: CKKSScalar>: CKKSInfos {
    /// Quantizes a coefficient vector whose length divides the plaintext
    /// degree. A full-length vector is dense; a shorter vector is placed with
    /// the implied coefficient gap.
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()>;

    /// Inverse of [`Self::encode_host_floats`]. The output length selects the
    /// dense or gap-strided coefficient representation.
    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()>;
}

/// Validates a non-empty coefficient vector of length `L` dividing `n` and
/// returns the coefficient gap `g = n / L`. With this gap, `coeffs[j]` lands
/// at polynomial coefficient `j·g`.
fn coefficient_gap<F>(coeffs: &[F], n: usize) -> Result<usize> {
    let l = coeffs.len();
    anyhow::ensure!(l > 0, "coefficient count must be non-zero");
    anyhow::ensure!(l <= n && n.is_multiple_of(l), "coefficient count {l} must divide n {n}");
    Ok(n / l)
}

/// Quantizes `coeffs` (length `N/gap`) and encodes them gap-strided directly into
/// the plaintext via `encode_vec_i*_strided` — the quantized integer buffer is only
/// `coeffs.len()` long, so the sparse path never materializes a length-`N` buffer.
/// `gap == 1` is the dense path (`coeffs.len() == N`).
fn encode_host_floats_strided<F, D>(pt: &mut CKKSPlaintext<D, i64>, coeffs: &[F], gap: usize) -> Result<()>
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
    let scale = F::from_usize(log_delta)
        .context("CKKS plaintext scale exponent is not representable by the codec scalar")?
        .exp2();
    let k = pt.encoded_k();
    if log_delta + log_budget <= 63 {
        let data: Vec<i64> = coeffs
            .iter()
            .enumerate()
            .map(|(index, &x)| {
                (x * scale)
                    .round()
                    .to_i64()
                    .with_context(|| format!("CKKS coefficient {index} is not representable as an i64 at scale 2^{log_delta}"))
            })
            .collect::<Result<_>>()?;
        pt.encode_vec_i64_strided(gap, &data, k);
    } else {
        let data: Vec<i128> = coeffs
            .iter()
            .enumerate()
            .map(|(index, &x)| {
                (x * scale)
                    .round()
                    .to_i128()
                    .with_context(|| format!("CKKS coefficient {index} is not representable as an i128 at scale 2^{log_delta}"))
            })
            .collect::<Result<_>>()?;
        pt.encode_vec_i128_strided(gap, &data, k);
    }
    Ok(())
}

/// Decodes the gap-strided coefficients of `pt` into `coeffs` (length `N/gap`),
/// inverse of [`encode_host_floats_strided`]. Only `coeffs.len()` coefficients are
/// reconstructed.
fn decode_host_floats_strided<F, D>(pt: &CKKSPlaintext<D, i64>, coeffs: &mut [F], gap: usize) -> Result<()>
where
    F: CKKSScalar,
    D: HostDataRef,
{
    let log_delta = pt.log_delta();
    let log_budget = pt.log_budget();
    anyhow::ensure!(
        log_delta + log_budget <= 127,
        "CKKS host decoding supports at most 127 torus bits, got {}",
        log_delta + log_budget,
    );
    let scale =
        (-F::from_usize(log_delta).context("CKKS plaintext scale exponent is not representable by the codec scalar")?).exp2();
    let k = pt.encoded_k();
    if log_delta + log_budget <= 63 {
        let mut data = vec![0i64; coeffs.len()];
        pt.decode_vec_i64_strided(gap, &mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::from_i64(i).context("decoded i64 coefficient is not representable by the codec scalar")? * scale;
        }
    } else {
        let mut data = vec![0i128; coeffs.len()];
        pt.decode_vec_i128_strided(gap, &mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::from_i128(i).context("decoded i128 coefficient is not representable by the codec scalar")? * scale;
        }
    }
    Ok(())
}

impl<F: CKKSScalar, D: HostDataMut + HostDataRef> CKKSPlaintextVecHostCodec<F> for CKKSPlaintext<D, i64> {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()> {
        let gap = coefficient_gap(coeffs, self.n().as_usize())?;
        encode_host_floats_strided(self, coeffs, gap)
    }

    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()> {
        let gap = coefficient_gap(coeffs, self.n().as_usize())?;
        decode_host_floats_strided(self, coeffs, gap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlotsKind;
    use crate::layouts::CKKSModuleAlloc;
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    #[test]
    fn plaintext_coeff_pack_allocates_requested_degree() {
        let module = Module::<HostBytesBackend>::new(16);
        let prec = CKKSMeta {
            log_sparsity: 0,
            log_delta: 40,
            slots: SlotsKind::Complex,
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
            slots: SlotsKind::Complex,
        };
        let base2k: Base2K = 50usize.into();

        // L = 4 (slots = 2), so gap = n/L = 4.
        let small = vec![1.0_f64, 2.0, 3.0, 4.0]; // re=[1,2], im=[3,4]
        let mut pt = module.ckks_pt_vec_alloc(base2k, (10usize + 40).into());
        pt.set_meta(prec);
        pt.encode_host_floats(&small).unwrap();

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
        pt.decode_host_floats(&mut got).unwrap();
        for (a, b) in small.iter().zip(&got) {
            assert!(approx(*a, *b), "sparse round-trip: {small:?} vs {got:?}");
        }
    }
}
