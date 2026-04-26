//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use std::ops::{Deref, DerefMut};

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module};

use crate::{CKKSInfos, CKKSMeta, error::CKKSCompositionError};

/// CKKS ciphertext storage plus semantic precision metadata.
///
/// `inner` contains the raw GLWE torus digits while `meta` describes the
/// semantic decimal scaling and remaining homomorphic capacity of the value.
pub struct CKKSCiphertext<D: Data> {
    /// Raw GLWE ciphertext storage.
    pub inner: GLWE<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub meta: CKKSMeta,
}

impl<D: Data> CKKSCiphertext<D> {
    pub(crate) fn from_inner(inner: GLWE<D>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }
}

impl<D: Data> Deref for CKKSCiphertext<D> {
    type Target = GLWE<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data> DerefMut for CKKSCiphertext<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data> LWEInfos for CKKSCiphertext<D> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<D: Data> GLWEInfos for CKKSCiphertext<D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data> CKKSInfos for CKKSCiphertext<D> {
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

impl<D: DataRef> GLWEToRef for CKKSCiphertext<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        self.inner.to_ref()
    }
}

impl<D: DataMut> GLWEToMut for CKKSCiphertext<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        self.inner.to_mut()
    }
}

impl CKKSCiphertext<Vec<u8>> {
    /// Allocates an owned ciphertext buffer with zeroed metadata.
    ///
    /// Inputs:
    /// - `n`: polynomial degree
    /// - `k`: torus storage precision in bits
    /// - `base2k`: limb radix
    ///
    /// Output:
    /// - a rank-1 ciphertext buffer ready to be populated by encryption or
    ///   evaluation code
    pub fn alloc(n: Degree, k: TorusPrecision, base2k: Base2K) -> Self {
        Self::from_inner(GLWE::alloc(n, base2k, k, Rank(1)), CKKSMeta::default())
    }

    /// Allocates an owned ciphertext from an existing GLWE layout descriptor.
    ///
    /// Inputs:
    /// - `infos`: degree, rank, `base2k`, and `max_k` of the target buffer
    ///
    /// Output:
    /// - a fresh ciphertext buffer with default metadata
    ///
    /// Errors:
    /// - propagates allocation/layout errors from the underlying GLWE type
    pub fn alloc_from_infos<A>(infos: &A) -> Result<Self>
    where
        A: GLWEInfos,
    {
        Ok(Self::from_inner(
            GLWE::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank()),
            CKKSMeta::default(),
        ))
    }
}

/// Maintenance operations for resizing ciphertext limb storage.
pub trait CKKSMaintainOps {
    /// Reallocates the owned backing buffer to exactly `size` limbs.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose owned limb buffer should be resized
    /// - `size`: requested number of limbs
    ///
    /// Output:
    /// - returns `Ok(())` after resizing `ct`
    ///
    /// Behavior:
    /// - preserves ciphertext metadata
    /// - rejects shrink operations that would make the buffer too small for the
    ///   current semantic precision
    ///
    /// Errors:
    /// - `LimbReallocationShrinksBelowMetadata` if the requested limb count
    ///   cannot represent the current metadata
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()>;

    /// Shrinks an owned ciphertext buffer to the minimum limb count that still
    /// preserves its current metadata.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose limb storage should be compacted
    ///
    /// Output:
    /// - returns `Ok(())` after compacting `ct`
    ///
    /// Errors:
    /// - propagates `ckks_reallocate_limbs_checked` if the computed compact
    ///   size would violate metadata constraints
    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()>;

    /// Returns a newly allocated owned ciphertext holding a compacted copy of
    /// `ct`.
    ///
    /// Inputs:
    /// - `ct`: ciphertext to copy and compact
    ///
    /// Output:
    /// - a fresh owned ciphertext with the same metadata and the minimum limb
    ///   count needed to preserve it
    ///
    /// Errors:
    /// - propagates allocation failures from the underlying GLWE type
    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: DataRef;
}

#[doc(hidden)]
pub trait CKKSMaintainOpsDefault {
    fn ckks_reallocate_limbs_checked_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        let base2k = ct.base2k().as_usize();
        let required_limbs = ct.effective_k().div_ceil(base2k);
        anyhow::ensure!(
            size >= required_limbs,
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: ct.max_k().as_usize(),
                log_delta: ct.log_delta(),
                base2k,
                requested_limbs: size,
            }
        );
        ct.data_mut().reallocate_limbs(size);
        Ok(())
    }

    fn ckks_compact_limbs_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        self.ckks_reallocate_limbs_checked_default(ct, size)?;
        Ok(())
    }

    fn ckks_compact_limbs_copy_default<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: DataRef,
    {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        let mut compact = CKKSCiphertext::alloc(ct.n(), (size * ct.base2k().as_usize()).into(), ct.base2k());
        compact.meta = ct.meta();
        let dst_len = compact.data().data.len();
        compact.data_mut().data[..].copy_from_slice(&ct.data().data.as_ref()[..dst_len]);
        Ok(compact)
    }
}

impl<BE: Backend> CKKSMaintainOpsDefault for Module<BE> {}

impl<BE: Backend> CKKSMaintainOps for Module<BE>
where
    Module<BE>: CKKSMaintainOpsDefault,
{
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        self.ckks_reallocate_limbs_checked_default(ct, size)
    }

    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        self.ckks_compact_limbs_default(ct)
    }

    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: DataRef,
    {
        self.ckks_compact_limbs_copy_default(ct)
    }
}

pub(crate) trait CKKSOffset {
    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos;
    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos;
}

impl<D: Data> CKKSOffset for CKKSCiphertext<D> {
    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos,
    {
        a.effective_k().min(b.effective_k()).saturating_sub(self.max_k().as_usize())
    }

    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos,
    {
        a.effective_k().saturating_sub(self.max_k().as_usize())
    }
}
