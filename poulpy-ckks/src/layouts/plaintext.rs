use std::{
    fmt::{self},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetLWEInfos,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef};
use rand_distr::num_traits::{Float, ToPrimitive};

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
            meta.effective_k() <= self.max_k().as_usize(),
            crate::CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta(),
                base2k: self.base2k().as_usize(),
                requested_limbs: self.size(),
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

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn n(&self) -> Degree {
        self.inner.n()
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
}

impl<D: HostDataMut> SetLWEInfos for CKKSPlaintext<D> {
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
        self.meta.log_delta()
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget()
    }
}

pub trait CKKSPlaintextVecHostCodec<F: CKKSScalar>: CKKSInfos + LWEInfos {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()>;
    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()>;
}

impl<F: CKKSScalar, D: HostDataMut + HostDataRef> CKKSPlaintextVecHostCodec<F> for CKKSPlaintext<D> {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()> {
        let log_delta = self.log_delta();
        let log_budget = self.log_budget();
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        anyhow::ensure!(log_delta <= max_log_delta_prec_for::<F>());

        let scale = F::from_usize(log_delta).unwrap().exp2();
        let k = self.max_k();
        if log_delta + log_budget <= 63 {
            let data: Vec<i64> = coeffs.iter().map(|&x| (x * scale).round().to_i64().unwrap()).collect();
            self.encode_vec_i64(&data, k);
        } else {
            let data: Vec<i128> = coeffs.iter().map(|&x| (x * scale).round().to_i128().unwrap()).collect();
            self.encode_vec_i128(&data, k);
        }
        Ok(())
    }

    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()> {
        let log_delta = self.log_delta();
        let log_budget = self.log_budget();
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        anyhow::ensure!(log_delta <= max_log_delta_prec_for::<F>());
        anyhow::ensure!(log_delta + log_budget <= 127);

        let scale = (-F::from_usize(log_delta).unwrap()).exp2();
        let k = self.max_k();
        if log_delta + log_budget <= 63 {
            let mut data = vec![0i64; coeffs.len()];
            self.decode_vec_i64(&mut data, k);
            coeffs
                .iter_mut()
                .zip(data.iter())
                .for_each(|(f, i)| *f = F::from_i64(*i).unwrap() * scale);
        } else {
            let mut data = vec![0i128; coeffs.len()];
            self.decode_vec_i128(&mut data, k);
            coeffs
                .iter_mut()
                .zip(data.iter())
                .for_each(|(f, i)| *f = F::from_i128(*i).unwrap() * scale);
        }
        Ok(())
    }
}

fn max_log_delta_prec_for<F>() -> usize
where
    F: Float + ToPrimitive,
{
    ((-F::epsilon().log2()).round().to_usize().unwrap()) + 1
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
            log_budget: 12,
            log_delta: 40,
        };
        let base2k: Base2K = 52usize.into();

        let pt = module.ckks_pt_coeffs_alloc(3, base2k, prec);

        assert_eq!(pt.n().as_usize(), 3);
        assert_eq!(pt.base2k(), base2k);
        assert_eq!(pt.meta(), prec);
        assert!(pt.effective_k() <= pt.max_k().as_usize());
    }
}
