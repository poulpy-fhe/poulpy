//! Transform preparation reference implementations.
//!
//! Implements docs/linear_transformation.md: diagonals are encoded by the scheme-aware
//! caller (CKKS), then turned into right convolution operands (`CnvPVecR`).
//! The resident transform is allocated up-front as a
//! `LinearTransformation<PreparedDiagonal<…>>` via
//! [`LinearTransformation::alloc_prepared`] from a [`LinearTransformationLayout`]
//! and a plaintext-shape proxy; this module's `_into` function only fills the
//! pre-allocated `CnvPVecR` slots, performing zero `CnvPVecR` allocations.
//! Backends forward to them from their [`crate::oep::LinearTransformationDefault`]
//! impl.

use poulpy_hal::layouts::{CnvPVecROwned, CnvPVecRToBackendMut};
use poulpy_hal::{
    api::{Convolution, ModuleN, ModuleNew},
    layouts::{Backend, PrepareHint, ScratchArena},
};

use crate::layouts::{
    GLWEInfos, GLWEToBackendRef, LWEInfos, LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep,
    LinearTransformationLayout, LinearTransformationPlan, prepared::PreparedDiagonal,
};

impl<BE: Backend> LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>> {
    /// Pre-allocates a resident (prepared) linear transformation sized for the
    /// given BSGS `layout` and plaintext shape `pt_infos`.
    ///
    /// Convenience for the layout-driven flow: builds the BSGS index via
    /// `layout.index()` and forwards to [`Self::alloc_prepared_from_index`].
    pub fn alloc_prepared<P>(layout: &LinearTransformationLayout, pt_infos: &P) -> Self
    where
        P: LWEInfos,
    {
        Self::alloc_prepared_from_index(&layout.index(), pt_infos)
    }

    /// Pre-allocates a resident transform sized for an explicit BSGS `index`.
    ///
    /// Each slot takes the plaintext's degree, `base2k` and `k`, so a compact
    /// (degree-`n`) diagonal gets a degree-`n` slot; the convolution buffers are
    /// zeroed and populated by `glwe_prepare_linear_transformation_rhs`. The
    /// per-diagonal `log_scale` is left at `0` for the scheme layer to set.
    pub fn alloc_prepared_from_index<P>(index: &LinearTransformationPlan, pt_infos: &P) -> Self
    where
        P: LWEInfos,
    {
        let pt_n = pt_infos.n().as_usize();
        let pt_size = pt_infos.size();
        let base2k = pt_infos.base2k();
        let k = pt_infos.k();

        let mut giant_steps = Vec::with_capacity(index.giant_steps.len());
        for (g, &rot) in index.giant_steps.iter().enumerate() {
            let baby_rots = &index.index[g];
            let mut diagonals = Vec::with_capacity(baby_rots.len());
            for &baby in baby_rots {
                diagonals.push(LinearTransformationDiagonal {
                    baby,
                    plaintext: PreparedDiagonal {
                        cnv: CnvPVecROwned::<BE>::alloc(pt_n, 1, pt_size, PrepareHint::Reuse),
                        base2k,
                        k,
                        log_scale: 0,
                    },
                });
            }
            giant_steps.push(LinearTransformationGiantStep { rot, diagonals });
        }

        LinearTransformation {
            baby_steps: index.baby_steps.clone(),
            giant_steps,
        }
    }

    /// Sets the per-diagonal `log_scale` of every diagonal; called by the scheme
    /// layer during the populate step (mirrors the streamed plaintext's
    /// `log_delta`).
    pub fn set_log_scale(&mut self, log_scale: usize) {
        for gs in &mut self.giant_steps {
            for d in &mut gs.diagonals {
                d.plaintext.set_log_scale(log_scale);
            }
        }
    }

    /// Base-2 log of the plaintext scaling factor shared by every diagonal.
    pub fn log_scale(&self) -> usize {
        self.first_diagonal_plaintext()
            .expect("prepared linear transformation has no diagonals")
            .log_scale()
    }
}

/// Reference impl: scratch bytes for `glwe_prepare_linear_transformation_rhs`.
///
/// The budget is sized at the module degree, an upper bound for a compact
/// diagonal, which is prepared under a module of its own degree.
pub fn glwe_prepare_linear_transformation_rhs_tmp_bytes_default<BE, M, P>(module: &M, pt_infos: &P) -> usize
where
    BE: Backend,
    M: Convolution<BE>,
    P: LWEInfos,
{
    module.cnv_prepare_right_tmp_bytes(pt_infos.size(), pt_infos.size())
}

/// Reference impl: encodes every diagonal of `lt` into the matching
/// pre-allocated `CnvPVecR` slot in `prepared`.
///
/// `prepared` must have been sized via
/// [`LinearTransformation::alloc_prepared`](LinearTransformation::alloc_prepared)
/// for the same BSGS schedule (giant rotations and baby rotations) that `lt`
/// follows.
pub fn glwe_prepare_linear_transformation_rhs_default<BE, M, P>(
    module: &M,
    prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
    lt: &LinearTransformation<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: Convolution<BE> + ModuleN + ModuleNew<BE>,
    P: GLWEToBackendRef<BE> + GLWEInfos,
{
    if !lt.baby_steps.is_empty() {
        assert_eq!(
            lt.baby_steps.first(),
            Some(&0),
            "baby_steps must start with the identity rotation (0)"
        );
    }

    let first = prepared
        .first_diagonal_plaintext()
        .expect("prepared linear transformation has no diagonals");
    let pt_base2k = first.base2k();
    let pt_k = first.k();
    let pt_base2k_usize = pt_base2k.as_usize();
    let pt_k_usize = pt_k.as_usize();
    let pt_n = first.n().as_usize();
    // The diagonal is an integer poly encoded across its full physical width
    // (`max_k`), so it is consumed at `max_k`, not the (possibly smaller)
    // effective `k`, otherwise the low limb's data is truncated.
    // A compact diagonal is prepared under a module of its own degree: the
    // prepares are degree-blind, only the apply forms read a degree below the
    // module's. Built once per call; every diagonal shares the degree.
    let small: Option<M> = (pt_n < module.n()).then(|| M::new(pt_n as u64));
    let prepare_module: &M = small.as_ref().unwrap_or(module);

    for gs in &lt.giant_steps {
        if gs.diagonals.is_empty() {
            continue;
        }
        let prepared_gs = prepared
            .giant_steps
            .iter_mut()
            .find(|p| p.rot == gs.rot)
            .unwrap_or_else(|| panic!("prepared cache has no giant step for rotation {}", gs.rot));

        for d in &gs.diagonals {
            let plaintext = &d.plaintext;
            assert_eq!(
                plaintext.base2k(),
                pt_base2k,
                "linear transformation diagonal base2k does not match prepared cache"
            );
            assert_eq!(
                plaintext.k(),
                pt_k,
                "linear transformation diagonal k does not match prepared cache"
            );
            assert_eq!(
                pt_k_usize.div_ceil(pt_base2k_usize),
                plaintext.size(),
                "linear transformation plaintext size does not match its effective precision"
            );
            assert_eq!(
                plaintext.n().as_usize(),
                pt_n,
                "linear transformation diagonal degree does not match prepared cache"
            );

            let prepared_slot = prepared_gs
                .diagonals
                .iter_mut()
                .find(|p| p.baby == d.baby)
                .unwrap_or_else(|| panic!("prepared cache has no diagonal slot for baby {} at giant {}", d.baby, gs.rot));
            let plaintext_backend = plaintext.to_backend_ref();
            prepare_module.cnv_prepare_right(
                &mut prepared_slot.plaintext.cnv_mut().to_backend_mut(),
                &plaintext_backend.data,
                scratch,
            );
        }
    }
}
