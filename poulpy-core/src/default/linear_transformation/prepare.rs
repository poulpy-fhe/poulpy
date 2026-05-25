//! Transform preparation.
//!
//! Implements docs/lt_bsgs.md §5.2: diagonals are already encoded and
//! pre-rotated by the CKKS layer; preparation turns each non-zero diagonal into a
//! right convolution operand (`CnvPVecR`) and prunes unused baby/giant metadata.

use std::collections::{BTreeMap, BTreeSet};

use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution},
    layouts::{Backend, CnvPVecRToBackendMut, Module, ScratchArena},
};

use crate::{
    default::operations::msb_mask_bottom_limb,
    layouts::{GLWEInfos, GLWEToBackendRef},
};

use super::{GLWELinearTransform, GLWEPreparedLinearTransform, GLWEPreparedLinearTransformGiantStep};

/// GLWE-level preparation of a [`GLWELinearTransform`].
pub trait GLWEPrepareLinearTransformOps<BE: Backend> {
    /// Scratch bytes required by [`Self::glwe_prepare_linear_transform`].
    fn glwe_prepare_linear_transform_tmp_bytes<P>(&self, lt: &GLWELinearTransform<P>) -> usize
    where
        P: GLWEInfos;

    /// Prepares plaintext diagonals as right convolution operands, pruning unused
    /// baby steps and empty giant steps.
    fn glwe_prepare_linear_transform<P>(
        &self,
        lt: &GLWELinearTransform<P>,
        prepared: &mut GLWEPreparedLinearTransform<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        P: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEPrepareLinearTransformOps<BE> for Module<BE>
where
    Module<BE>: CnvPVecAlloc<BE> + Convolution<BE>,
{
    fn glwe_prepare_linear_transform_tmp_bytes<P>(&self, lt: &GLWELinearTransform<P>) -> usize
    where
        P: GLWEInfos,
    {
        lt.giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| self.cnv_prepare_right_tmp_bytes(d.plaintext.size(), d.plaintext.size()))
            .max()
            .unwrap_or(0)
    }

    fn glwe_prepare_linear_transform<P>(
        &self,
        lt: &GLWELinearTransform<P>,
        prepared: &mut GLWEPreparedLinearTransform<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        if !lt.baby_steps.is_empty() {
            assert_eq!(
                lt.baby_steps.first(),
                Some(&0),
                "baby_steps must start with the identity rotation (0)"
            );
        }

        let baby_steps_set: BTreeSet<i64> = lt.baby_steps.iter().copied().collect();
        let mut used_babies = BTreeSet::new();
        for gs in &lt.giant_steps {
            for d in &gs.diagonals {
                assert!(
                    baby_steps_set.contains(&d.baby),
                    "linear transformation diagonal references missing baby step"
                );
                used_babies.insert(d.baby);
            }
        }

        let baby_steps = used_babies.into_iter().collect();

        let mut giant_steps = Vec::new();
        let mut first_plaintext: Option<&P> = None;
        for gs in &lt.giant_steps {
            if gs.diagonals.is_empty() {
                continue;
            }

            let mut diagonals = BTreeMap::new();
            for d in &gs.diagonals {
                let baby = d.baby;
                let plaintext = &d.plaintext;
                if let Some(first_plaintext) = first_plaintext {
                    assert_eq!(
                        plaintext.base2k(),
                        first_plaintext.base2k(),
                        "linear transformation diagonals must use the same base2k"
                    );
                    assert_eq!(
                        plaintext.max_k(),
                        first_plaintext.max_k(),
                        "linear transformation diagonals must use the same effective precision"
                    );
                } else {
                    first_plaintext = Some(plaintext);
                }
                assert!(
                    !diagonals.contains_key(&baby),
                    "linear transformation giant step contains duplicate baby-step rotation"
                );

                let mut prepared_plaintext = self.cnv_pvec_right_alloc(1, plaintext.size());
                let plaintext_effective_k = plaintext.max_k().as_usize();
                let plaintext_base2k = plaintext.base2k().as_usize();
                assert_eq!(
                    plaintext_effective_k.div_ceil(plaintext_base2k),
                    plaintext.size(),
                    "linear transformation plaintext size must match its effective precision"
                );
                let mask = msb_mask_bottom_limb(plaintext_base2k, plaintext_effective_k);
                {
                    let plaintext_backend = plaintext.to_backend_ref();
                    self.cnv_prepare_right(
                        &mut prepared_plaintext.to_backend_mut(),
                        &plaintext_backend.data,
                        mask,
                        scratch,
                    );
                }

                diagonals.insert(baby, prepared_plaintext);
            }

            giant_steps.push(GLWEPreparedLinearTransformGiantStep { rot: gs.rot, diagonals });
        }

        prepared.baby_steps = baby_steps;
        prepared.giant_steps = giant_steps;
    }
}
