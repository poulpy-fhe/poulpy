use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::AlignedBuf;
use std::{
    fmt::{self},
    ops::{Deref, DerefMut},
};

use anyhow::{Context, Result};
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextInfos, GLWEPlaintextMeta,
    GLWEPlaintextReborrowBackendMut, GLWEPlaintextReborrowBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBSGSMeta, SetBase2k, SetGLWEPlaintextInfos, SetK, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, Ring, Standard, ZnxWord};

use crate::{CKKSInfos, SetCKKSInfos, ckks_log_delta};

use super::CKKSScalar;

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintext<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    /// Raw GLWE plaintext limb storage; its plaintext metadata holds the CKKS
    /// scale and slot kind.
    pub(crate) inner: GLWEPlaintext<D, W, Rg>,
}

impl<D: Data, W: ZnxWord, Rg: Ring> CKKSPlaintext<D, W, Rg> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D, W, Rg>, meta: GLWEPlaintextMeta) -> Self {
        let mut pt = Self { inner };
        pt.set_meta(meta);
        pt
    }

    /// Rebuilds this backend-owned plaintext as a host-owned [`CKKSPlaintext<AlignedBuf, W, Rg>`].
    pub fn to_host_owned<BE>(&self) -> CKKSPlaintext<AlignedBuf, W, Rg>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        CKKSPlaintext {
            inner: self.inner.to_host_owned::<BE>(),
        }
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
    /// Normal CKKS operations update metadata themselves. `meta.log_sparsity` is
    /// not read: a plaintext is dense at its own degree.
    pub fn set_meta_checked(&mut self, meta: GLWEPlaintextMeta) -> Result<()> {
        anyhow::ensure!(
            self.k().as_usize() <= self.max_k().as_usize() && ckks_log_delta(&meta) <= self.k().as_usize(),
            crate::CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: ckks_log_delta(&meta),
                base2k: self.base2k().as_usize(),
                requested_limbs: self.max_size(),
            }
        );
        self.set_meta(meta);
        Ok(())
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEToBackendRef<BE, Rg> for CKKSPlaintext<D, BE::ZnxWord, Rg>
where
    GLWEPlaintext<D, BE::ZnxWord, Rg>: GLWEToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, Rg> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEToBackendMut<BE, Rg> for CKKSPlaintext<D, BE::ZnxWord, Rg>
where
    GLWEPlaintext<D, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, Rg> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    fn set_canonical(&mut self, _canonical: bool) {}

    fn set_noise(&mut self, _metadata: Option<poulpy_core::ComponentNoise>) {}
}

/// Backend-owned CKKS plaintext: the backend's buffer type and its coefficient word.
pub type CKKSPlaintextOwned<BE, Rg = Standard> = CKKSPlaintext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord, Rg>;

poulpy_core::view_wrapper!(
    /// Scratch-backed mutable CKKS plaintext view.
    ///
    /// This nominal wrapper contains an ordinary
    /// `CKKSPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg>`; both its polynomial storage and CKKS
    /// metadata therefore have exactly the same representation as other CKKS
    /// plaintexts.
    CKKSPlaintextViewMut,
    CKKSPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg>
);
poulpy_core::impl_glwe_infos!(CKKSPlaintextViewMut);
crate::impl_ckks_infos!(ckks_inner CKKSPlaintextViewMut);

impl<'a, BE: poulpy_hal::layouts::Backend + 'a, Rg: Ring> poulpy_core::layouts::IntPolyInfos
    for CKKSPlaintextViewMut<'a, BE, Rg>
{
    fn encoded_k(&self) -> TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<BE: Backend, Rg: Ring> SetBase2k for CKKSPlaintextViewMut<'_, BE, Rg> {
    fn set_base2k(&mut self, base2k: Base2K) {
        SetBase2k::set_base2k(&mut self.inner, base2k);
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEToBackendRef<BE, Rg> for CKKSPlaintextViewMut<'a, BE, Rg> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord, Rg> {
        <GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg> as GLWEPlaintextReborrowBackendRef<BE, Rg>>::reborrow_backend_ref(
            &self.inner.inner,
        )
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEToBackendMut<BE, Rg> for CKKSPlaintextViewMut<'a, BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord, Rg> {
        <GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg> as GLWEPlaintextReborrowBackendMut<BE, Rg>>::reborrow_backend_mut(
            &mut self.inner.inner,
        )
    }

    fn set_canonical(&mut self, _canonical: bool) {}

    fn set_noise(&mut self, _metadata: Option<poulpy_core::ComponentNoise>) {}
}

impl<D: Data, W: ZnxWord, Rg: Ring> Deref for CKKSPlaintext<D, W, Rg> {
    type Target = GLWEPlaintext<D, W, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> DerefMut for CKKSPlaintext<D, W, Rg> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for CKKSPlaintext<D, W, Rg> {
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

impl<D: Data, W: ZnxWord, Rg: Ring> IntPolyInfos for CKKSPlaintext<D, W, Rg> {
    fn encoded_k(&self) -> TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for CKKSPlaintext<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> SetK for CKKSPlaintext<D, W, Rg> {
    fn set_k(&mut self, k: TorusPrecision) {
        SetK::set_k(&mut self.inner, k);
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> SetBSGSMeta for CKKSPlaintext<D, W, Rg> {
    fn set_bsgs_log_budget(&mut self, log_budget: usize) {
        SetCKKSInfos::set_log_budget(self, log_budget);
    }
    fn set_bsgs_log_delta(&mut self, log_delta: usize) {
        SetCKKSInfos::set_log_delta(self, log_delta);
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> SetBase2k for CKKSPlaintext<D, W, Rg> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.set_base2k(base2k);
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for CKKSPlaintext<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPlaintextInfos for CKKSPlaintext<D, W, Rg> {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        self.inner.plaintext_meta()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> CKKSInfos for CKKSPlaintext<D, W, Rg> {}

impl<D: Data, W: ZnxWord, Rg: Ring> SetGLWEPlaintextInfos for CKKSPlaintext<D, W, Rg> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.inner.set_plaintext_meta(meta)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> BSGSMeta for CKKSPlaintext<D, W, Rg> {
    fn bsgs_log_budget(&self) -> usize {
        CKKSInfos::log_budget(self)
    }
    fn bsgs_log_delta(&self) -> usize {
        CKKSInfos::log_delta(self)
    }
}

pub trait CKKSPlaintextVecHostCodec<F: CKKSScalar>: CKKSInfos {
    /// Quantizes one value per coefficient of the plaintext.
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()>;

    /// Inverse of [`Self::encode_host_floats`].
    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()>;
}

/// Validates one coefficient per coefficient of a degree-`n` plaintext.
fn check_coefficient_count<F>(coeffs: &[F], n: usize) -> Result<()> {
    anyhow::ensure!(
        coeffs.len() == n,
        "coefficient count {} must equal the plaintext degree {n}",
        coeffs.len()
    );
    Ok(())
}

/// Quantizes `coeffs`, one per coefficient, into the plaintext.
fn quantize_into_plaintext<F, D, Rg: Ring>(pt: &mut CKKSPlaintext<D, i64, Rg>, coeffs: &[F]) -> Result<()>
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
    let k = pt.encoded_k();
    if log_delta + log_budget <= 63 {
        let data: Vec<i64> = coeffs
            .iter()
            .enumerate()
            .map(|(index, &x)| {
                x.ckks_quantize_i64(log_delta)
                    .with_context(|| format!("CKKS coefficient {index} is not representable as an i64 at scale 2^{log_delta}"))
            })
            .collect::<Result<_>>()?;
        pt.encode_vec_i64(&data, k);
    } else {
        let data: Vec<i128> = coeffs
            .iter()
            .enumerate()
            .map(|(index, &x)| {
                x.ckks_quantize(log_delta)
                    .with_context(|| format!("CKKS coefficient {index} is not representable as an i128 at scale 2^{log_delta}"))
            })
            .collect::<Result<_>>()?;
        pt.encode_vec_i128(&data, k);
    }
    Ok(())
}

/// Dequantizes every coefficient of `pt` into `coeffs`, inverse of [`quantize_into_plaintext`].
fn dequantize_from_plaintext<F, D, Rg: Ring>(pt: &CKKSPlaintext<D, i64, Rg>, coeffs: &mut [F]) -> Result<()>
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
    let k = pt.encoded_k();
    if log_delta + log_budget <= 63 {
        let mut data = vec![0i64; coeffs.len()];
        pt.decode_vec_i64(&mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::ckks_dequantize(i as i128, log_delta);
        }
    } else {
        let mut data = vec![0i128; coeffs.len()];
        pt.decode_vec_i128(&mut data, k);
        for (c, &i) in coeffs.iter_mut().zip(data.iter()) {
            *c = F::ckks_dequantize(i, log_delta);
        }
    }
    Ok(())
}

impl<F: CKKSScalar, D: HostDataMut + HostDataRef, Rg: Ring> CKKSPlaintextVecHostCodec<F> for CKKSPlaintext<D, i64, Rg> {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()> {
        check_coefficient_count(coeffs, self.n().as_usize())?;
        quantize_into_plaintext(self, coeffs)
    }

    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()> {
        check_coefficient_count(coeffs, self.n().as_usize())?;
        dequantize_from_plaintext(self, coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CKKSModuleInfos;
    use crate::layouts::CKKSPlaintextAlloc;
    use crate::{Scale, SlotLayout, SlotsKind};
    use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, Standard};

    #[test]
    fn plaintext_alloc_from_slots_takes_twice_the_slots() {
        let module = Module::<HostBytesBackend>::new(64);
        let (base2k, k) = (Base2K(12), TorusPrecision(24));
        assert_eq!(
            module
                .ckks_plaintext_alloc_from_slots::<Standard>(8, base2k, k)
                .n()
                .as_usize(),
            16
        );
        assert_eq!(
            module
                .ckks_plaintext_alloc_from_slots::<Standard>(module.ckks_max_slots::<Standard>(), base2k, k)
                .n()
                .as_usize(),
            64
        );
    }

    #[test]
    #[should_panic(expected = "ckks_plaintext_alloc_from_slots: the slot count needs a degree below the backend's minimum")]
    fn plaintext_alloc_from_slots_rejects_a_degree_below_the_floor() {
        let module = Module::<HostBytesBackend>::new(64);
        module.ckks_plaintext_alloc_from_slots::<Standard>(
            <HostBytesBackend as Backend>::MIN_DEGREE / 4,
            Base2K(12),
            TorusPrecision(24),
        );
    }

    #[test]
    fn plaintext_coeff_pack_allocates_requested_degree() {
        let module = Module::<HostBytesBackend>::new(16);
        let prec = GLWEPlaintextMeta {
            scale: Scale::Log(40),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        };
        let base2k: Base2K = 52usize.into();

        let mut pt = module.ckks_plaintext_alloc::<Standard>(3usize.into(), base2k, (12usize + 40).into());
        pt.set_meta(prec);

        assert_eq!(pt.n().as_usize(), 3);
        assert_eq!(pt.base2k(), base2k);
        assert_eq!(pt.meta(), prec);
        assert!(pt.k() <= pt.max_k());
    }
}
