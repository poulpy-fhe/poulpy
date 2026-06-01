//! Transform preparation.
//!
//! Implements docs/lt_bsgs.md §5: diagonals are encoded by the scheme-aware
//! caller (CKKS), then turned into right convolution operands (`CnvPVecR`).
//! The cache itself is allocated up-front via
//! [`GLWEPreparedLinearTransformationRhs::alloc`] from a [`LinearTransformationLayout`]
//! and a plaintext-shape proxy; this module's `_into` function only fills the
//! pre-allocated slots, performing zero `CnvPVecR` allocations.

//! Transform preparation reference implementations.
//!
//! Diagonals are encoded by the scheme-aware caller (CKKS), then turned into
//! right convolution operands (`CnvPVecR`). The cache itself is allocated
//! up-front via [`GLWEPreparedLinearTransformationRhs::alloc`]; the `*_default`
//! functions below only fill the pre-allocated slots, performing zero
//! `CnvPVecR` allocations. Backends forward to them from their
//! [`crate::oep::LinearTransformationDefault`] impl.

use std::collections::BTreeMap;

use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution},
    layouts::{Backend, CnvPVecRToBackendMut, ScratchArena},
};

use crate::{
    default::operations::msb_mask_bottom_limb,
    layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos},
};

use super::{
    GLWELinearTransform, GLWELinearTransformationSchedule, GLWEPreparedLinearTransformationRhs,
    GLWEPreparedLinearTransformationRhsGiantStep, LinearTransformationLayout,
};

impl<BE: Backend> GLWEPreparedLinearTransformationRhs<BE> {
    /// Pre-allocates a prepared linear-transformation cache sized for the
    /// given BSGS `layout` and plaintext shape `pt_infos`.
    ///
    /// Convenience for the layout-driven flow: builds the BSGS index via
    /// `layout.index()` and forwards to
    /// [`Self::alloc_from_index`](GLWEPreparedLinearTransformationRhs::alloc_from_index).
    pub fn alloc<M, P>(module: &M, layout: &LinearTransformationLayout, pt_infos: &P) -> Self
    where
        M: CnvPVecAlloc<BE>,
        P: LWEInfos,
    {
        Self::alloc_from_index(module, &layout.index(), pt_infos)
    }

    /// Pre-allocates a prepared cache sized for an explicit BSGS `index`.
    ///
    /// Stores `pt_base2k` / `pt_max_k` so the evaluator never needs the raw
    /// [`GLWELinearTransform`] again. Diagonal contents are populated by
    /// `glwe_prepare_linear_transformation_rhs`.
    pub fn alloc_from_index<M, P>(module: &M, index: &GLWELinearTransformationSchedule, pt_infos: &P) -> Self
    where
        M: CnvPVecAlloc<BE>,
        P: LWEInfos,
    {
        let baby_step_idx_by_rotation: BTreeMap<i64, usize> = index
            .baby_steps
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, rot)| (rot, idx))
            .collect();

        let pt_size = pt_infos.size();
        let mut giant_steps = Vec::with_capacity(index.giant_steps.len());
        for (g, &rot) in index.giant_steps.iter().enumerate() {
            let baby_rots = &index.index[g];
            let mut diagonals = BTreeMap::new();
            let mut baby_step_indexes = Vec::with_capacity(baby_rots.len());
            for &baby_rot in baby_rots {
                let baby_step_idx = *baby_step_idx_by_rotation
                    .get(&baby_rot)
                    .expect("index references missing baby step");
                diagonals.insert(baby_rot, module.cnv_pvec_right_alloc(1, pt_size));
                baby_step_indexes.push(baby_step_idx);
            }
            giant_steps.push(GLWEPreparedLinearTransformationRhsGiantStep {
                rot,
                baby_step_indexes,
                diagonals,
            });
        }

        Self {
            baby_steps: index.baby_steps.clone(),
            giant_steps,
            pt_base2k: pt_infos.base2k(),
            pt_max_k: pt_infos.max_k(),
            pt_log_scale: 0,
        }
    }
}

/// Reference impl: scratch bytes for `glwe_prepare_linear_transformation_rhs`.
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
/// `prepared` must have been sized via [`GLWEPreparedLinearTransformationRhs::alloc`]
/// for the same BSGS schedule (giant rotations and baby rotations) that `lt`
/// follows.
pub fn glwe_prepare_linear_transformation_rhs_default<BE, M, P>(
    module: &M,
    prepared: &mut GLWEPreparedLinearTransformationRhs<BE>,
    lt: &GLWELinearTransform<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecAlloc<BE> + Convolution<BE>,
    P: GLWEToBackendRef<BE> + GLWEInfos,
{
    if !lt.baby_steps.is_empty() {
        assert_eq!(
            lt.baby_steps.first(),
            Some(&0),
            "baby_steps must start with the identity rotation (0)"
        );
    }

    let pt_base2k = prepared.pt_base2k;
    let pt_max_k = prepared.pt_max_k;
    let pt_base2k_usize = pt_base2k.as_usize();
    let pt_max_k_usize = pt_max_k.as_usize();
    let mask = msb_mask_bottom_limb(pt_base2k_usize, pt_max_k_usize);

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
                plaintext.max_k(),
                pt_max_k,
                "linear transformation diagonal max_k does not match prepared cache"
            );
            assert_eq!(
                pt_max_k_usize.div_ceil(pt_base2k_usize),
                plaintext.size(),
                "linear transformation plaintext size does not match its effective precision"
            );

            let prepared_slot = prepared_gs
                .diagonals
                .get_mut(&d.baby)
                .unwrap_or_else(|| panic!("prepared cache has no diagonal slot for baby {} at giant {}", d.baby, gs.rot));
            let plaintext_backend = plaintext.to_backend_ref();
            module.cnv_prepare_right(&mut prepared_slot.to_backend_mut(), &plaintext_backend.data, mask, scratch);
        }
    }
}
