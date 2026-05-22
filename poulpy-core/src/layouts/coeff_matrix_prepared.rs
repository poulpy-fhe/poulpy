use std::marker::PhantomData;

use poulpy_hal::layouts::{Backend, CoeffGemmPanel, CoeffGemmPanelBackendRef, CoeffGemmPanelToBackendRef, Data};

use crate::layouts::{Base2K, CoeffBound};

/// A `CoeffMatrix` whose kernel-specific decomposition has been precomputed
/// once (into a backend [`CoeffGemmPanel`]), so it can be reused across many
/// [`crate::LWEMatrixMul::lwe_matrix_mul_bodies_prepared`] calls without
/// re-preparing `U`. `BU` is the compile-time entry bound that selected the
/// kernel width when preparing.
pub struct CoeffMatrixPrepared<D: Data, BE: Backend, BU: CoeffBound = i64> {
    pub(crate) panel: CoeffGemmPanel<D, BE>,
    pub(crate) base2k: Base2K,
    pub(crate) _bound: PhantomData<BU>,
}

pub type CoeffMatrixPreparedOwned<BE, BU = i64> = CoeffMatrixPrepared<<BE as Backend>::OwnedBuf, BE, BU>;

impl<BE: Backend, BU: CoeffBound> CoeffMatrixPrepared<BE::OwnedBuf, BE, BU> {
    pub fn from_parts(panel: CoeffGemmPanel<BE::OwnedBuf, BE>, base2k: Base2K) -> Self {
        Self {
            panel,
            base2k,
            _bound: PhantomData,
        }
    }

    /// Base-`2^base2k` of the prepared `U`.
    pub fn base2k(&self) -> Base2K {
        self.base2k
    }

    /// Output rows of `U` (`= rows_out`).
    pub fn rows_out(&self) -> usize {
        self.panel.rows_out()
    }

    /// Input rows of `U` (`= rows_in`).
    pub fn rows_in(&self) -> usize {
        self.panel.rows_in()
    }

    /// Number of base-`2^base2k` limbs of `U`.
    pub fn u_size(&self) -> usize {
        self.panel.u_size()
    }

    /// Backend-native view of the prepared panel, for the apply step.
    pub fn panel_ref(&self) -> CoeffGemmPanelBackendRef<'_, BE> {
        self.panel.to_backend_ref()
    }
}
