use super::CKKSRing;
use crate::{CKKSCtBounds, CKKSInfos, CKKSResult, api::CKKSModuleInfos};
use poulpy_core::layouts::{LWEInfos, prepared::PreparedDiagonal};
use poulpy_core::{LinearTransformationLayout, LinearTransformationPlan};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::Backend,
};
use std::ops::Deref;

/// Encoded or prepared linear transformation bound to its evaluation ring.
pub struct LinearTransformation<P> {
    pub(crate) inner: poulpy_core::LinearTransformation<P>,
    pub(crate) ring: CKKSRing,
}
impl<P> LinearTransformation<P> {
    /// Evaluation ring retained by this representation.
    pub fn ring(&self) -> CKKSRing {
        self.ring
    }

    /// Binds encoded diagonals to a module after checking their kind and embedding degree.
    pub fn from_plaintexts<M: CKKSModuleInfos>(module: &M, inner: poulpy_core::LinearTransformation<P>) -> CKKSResult<Self>
    where
        P: CKKSInfos,
    {
        let ring = module.ckks_ring();
        for step in &inner.giant_steps {
            for diagonal in &step.diagonals {
                ring.check_plaintext("linear transformation", &diagonal.plaintext)?;
            }
        }
        Ok(Self { inner, ring })
    }
}
impl<P> Deref for LinearTransformation<P> {
    type Target = poulpy_core::LinearTransformation<P>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Linear transformation with backend-prepared diagonals.
pub type LinearTransformationPrepared<BE> = LinearTransformation<PreparedDiagonal<<BE as Backend>::OwnedBuf, BE>>;

impl<BE: Backend> LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>> {
    /// Allocates a prepared destination in the module ring with the supplied diagonal shape.
    pub fn alloc_prepared<M, P>(module: &M, layout: &LinearTransformationLayout, pt: &P) -> Self
    where
        M: CnvPVecAlloc<BE> + CKKSModuleInfos,
        P: LWEInfos,
    {
        Self::alloc_prepared_from_index(module, &layout.index(), pt)
    }
    /// Allocates prepared diagonals from a resolved schedule and a plaintext shape.
    pub fn alloc_prepared_from_index<M, P>(module: &M, index: &LinearTransformationPlan, pt: &P) -> Self
    where
        M: CnvPVecAlloc<BE> + CKKSModuleInfos,
        P: LWEInfos,
    {
        Self {
            inner: poulpy_core::LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, index, pt),
            ring: module.ckks_ring(),
        }
    }
    /// Sets the common scale exponent carried by every prepared diagonal.
    pub fn set_log_scale(&mut self, log_scale: usize) {
        self.inner.set_log_scale(log_scale);
    }
}

/// Prepared baby rotations bound to their evaluation ring.
pub struct LinearTransformationBabySteps<BE: Backend> {
    pub(crate) inner: poulpy_core::LinearTransformationBabySteps<BE>,
    pub(crate) ring: CKKSRing,
}
impl<BE: Backend> Deref for LinearTransformationBabySteps<BE> {
    type Target = poulpy_core::LinearTransformationBabySteps<BE>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<BE: Backend> LinearTransformationBabySteps<BE> {
    /// Evaluation ring retained by this representation.
    pub fn ring(&self) -> CKKSRing {
        self.ring
    }
    /// Allocates baby rotations; panics if the source ring differs from the module.
    pub fn alloc<M, A>(module: &M, steps: &[i64], a: &A) -> Self
    where
        M: ModuleN + CnvPVecAlloc<BE> + CKKSModuleInfos,
        A: CKKSCtBounds,
    {
        let ring = module.ckks_ring();
        ring.check_ciphertext("baby-step allocation", a)
            .expect("incompatible ciphertext ring");
        Self {
            inner: poulpy_core::LinearTransformationBabySteps::alloc(module, steps, a),
            ring,
        }
    }
    /// Allocates the layout's baby rotations in the source and module ring.
    pub fn alloc_from_layout<M, A>(module: &M, layout: &LinearTransformationLayout, a: &A) -> Self
    where
        M: ModuleN + CnvPVecAlloc<BE> + CKKSModuleInfos,
        A: CKKSCtBounds,
    {
        Self::alloc(module, &layout.baby_steps(), a)
    }
}
