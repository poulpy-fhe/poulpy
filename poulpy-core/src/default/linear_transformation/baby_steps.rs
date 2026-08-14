//! Baby-step rotation materialization.
//!
//! Implements docs/linear_transformation.md. The non-trivial baby rotations share one
//! DFT of the input mask columns, then each key performs VMP -> IDFT -> add body
//! -> normalize -> automorphism. The resulting SMALL ciphertexts are prepared as
//! `CnvPVecL` so every giant step can reuse them in convolution form.

//! Baby-step rotation materialization (the prepared LHS).
//!
//! Implements docs/linear_transformation.md. The non-trivial baby rotations share one
//! DFT of the input mask columns, then each key performs VMP -> IDFT -> add body
//! -> normalize -> automorphism. The resulting SMALL ciphertexts are prepared
//! as `CnvPVecL` and stored in a [`LinearTransformationBabySteps`] (whose
//! definition lives in [`crate::layouts`]); this module owns the HAL-dependent
//! allocator and population routines.

use poulpy_hal::layouts::CnvPVecLToBackendMut;
use std::collections::BTreeMap;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssignBackend, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, GaloisElement, ScratchArena, VecZnxBigToBackendRef, VecZnxDftBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    GLWEAutomorphism, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    default::{keyswitching::GGLWEProductDefault, operations::msb_mask_bottom_limb},
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        prepared::GGLWEPreparedToBackendRef,
    },
};

use super::{LinearTransformationBabySteps, LinearTransformationLayout};

impl<BE: Backend> LinearTransformationBabySteps<BE> {
    /// Pre-allocates a baby-step cache for the given `baby_steps` rotations
    /// and input ciphertext shape `a`.
    ///
    /// Each prepared baby rotation is a `CnvPVecL` with `a.rank() + 1` columns
    /// and `a.size()` limbs. The `baby_steps` slice typically comes from
    /// [`LinearTransformationLayout::baby_steps`] (before encoding) or the
    /// `baby_steps` field of a resident `LinearTransformation<PreparedDiagonal>` (after encoding).
    /// Duplicate rotations in `baby_steps` are de-duplicated.
    pub fn alloc<M, A>(module: &M, baby_steps: &[i64], a: &A) -> Self
    where
        M: CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        let cols = a.rank().as_usize() + 1;
        let size = a.size();
        let mut values = BTreeMap::new();
        for &rot in baby_steps {
            values.entry(rot).or_insert_with(|| module.cnv_pvec_left_alloc(cols, size));
        }
        Self { values }
    }

    /// Convenience: pre-allocates from a [`LinearTransformationLayout`].
    pub fn alloc_from_layout<M, A>(module: &M, layout: &LinearTransformationLayout, a: &A) -> Self
    where
        M: CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        Self::alloc(module, &layout.baby_steps(), a)
    }
}

pub(super) fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<BE, M, A, K>(
    module: &M,
    a_infos: &A,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let a_size = a_infos.size();
    let key_size = key_infos.size();
    let baby = module.glwe_bytes_of_from_infos(a_infos);
    let prepare = module.cnv_prepare_left_pvec_tmp_bytes(a_infos.size(), a_infos.size());

    let hoisted_a_dft = module.bytes_of_vec_znx_dft(cols - 1, a_size);
    let hoisted_rot = module.bytes_of_vec_znx_dft(cols, key_size)
        + module.bytes_of_vec_znx_big(cols, key_size)
        + module
            .gglwe_product_dft_tmp_bytes_default(key_size, a_size, key_infos)
            .max(module.vec_znx_idft_apply_tmp_bytes());
    let hoisted = hoisted_a_dft + baby + hoisted_rot.max(prepare);

    let fallback = baby + module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos).max(prepare);
    hoisted.max(fallback).max(prepare)
}

#[allow(clippy::too_many_arguments)]
fn glwe_hoisted_baby_rotation<BE, M, R, A, H, K>(
    module: &M,
    baby: &mut R,
    rot: i64,
    a: &A,
    a_dft_ref: &VecZnxDftBackendRef<'_, BE>,
    key_size: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GaloisElement
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let cols = a.rank().as_usize() + 1;
    let key: &K = keys
        .get_automorphism_key(module.galois_element(rot))
        .unwrap_or_else(|| panic!("missing automorphism key for baby-step rotation {rot}"));
    let key_ref = key.to_backend_ref();
    assert_eq!(key_ref.base2k(), a.base2k());

    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    if key_ref.dsize().as_usize() > 1 {
        // See `glwe_hoisted_baby_rotations`: multi-digit VMP accumulates into
        // top limbs that must not contain stale scratch contents.
    }
    module.gglwe_product_dft_default(&mut res_dft, a_dft_ref, &key_ref, &mut scratch_1.borrow());

    let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, cols, key_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch_2.borrow());
    }
    {
        let a_ref = a.to_backend_ref();
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a_ref.data, 0);
    }

    let res_big_ref = res_big.to_backend_ref();
    let baby_base2k = baby.base2k().as_usize();
    let a_base2k = a.base2k().as_usize();
    {
        let mut baby_ref = baby.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_big_normalize(
                &mut baby_ref.data,
                baby_base2k,
                0,
                col,
                &res_big_ref,
                a_base2k,
                col,
                &mut scratch_2.borrow(),
            );
        }
        for col in 0..cols {
            module.vec_znx_automorphism_assign_backend(key.p(), &mut baby_ref.data, col, &mut scratch_2.borrow());
        }
    }
}

/// Fills a pre-allocated baby-step cache with `rot(a, k)` for every `k` already
/// stored in `cache`.
///
/// The cache must have been sized via [`LinearTransformationBabySteps::alloc`].
/// This is the populating counterpart of the old returning variant: it
/// performs zero `CnvPVecL` allocations because the slots are owned by
/// `cache`.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_prepare_linear_transformation_baby_steps<BE, M, A, H, K>(
    module: &M,
    cache: &mut LinearTransformationBabySteps<BE>,
    a: &A,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + ModuleN
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let cols = a.rank().as_usize() + 1;
    let a_size = a.size();
    let mask = msb_mask_bottom_limb(a.base2k().as_usize(), a.k().as_usize());
    let has_nonzero_rotation = cache.values.keys().any(|&rot| rot != 0);
    let (use_hoisted, key_size) = if has_nonzero_rotation {
        let key_infos = keys.automorphism_key_infos();
        (a.base2k() == key_infos.base2k(), key_infos.work_size(a.k()))
    } else {
        (false, a.size())
    };

    if use_hoisted {
        let scratch = scratch.borrow();
        let (mut a_dft, mut loop_scratch) = scratch.take_vec_znx_dft_scratch(module, cols - 1, a_size);
        {
            let a_ref = a.to_backend_ref();
            for col_i in 0..cols - 1 {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, &a_ref.data, col_i + 1);
            }
        }
        let a_dft_ref = a_dft.to_backend_ref();

        for (&rot, prepared) in cache.values.iter_mut() {
            assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
            assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
            if rot == 0 {
                let a_ref = a.to_backend_ref();
                module.cnv_prepare_left_pvec(&mut prepared.to_backend_mut(), &a_ref.data, mask, &mut loop_scratch.borrow());
            } else {
                let (mut baby, mut baby_scratch) = loop_scratch.borrow().take_glwe_scratch(a);
                glwe_hoisted_baby_rotation(
                    module,
                    &mut baby,
                    rot,
                    a,
                    &a_dft_ref,
                    key_size,
                    keys,
                    &mut baby_scratch.borrow(),
                );
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left_pvec(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
        }
    } else {
        for (&rot, prepared) in cache.values.iter_mut() {
            assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
            assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
            if rot == 0 {
                let a_ref = a.to_backend_ref();
                module.cnv_prepare_left_pvec(&mut prepared.to_backend_mut(), &a_ref.data, mask, scratch);
            } else {
                let key: &K = keys
                    .get_automorphism_key(module.galois_element(rot))
                    .unwrap_or_else(|| panic!("missing automorphism key for baby-step rotation {rot}"));
                let (mut baby, mut baby_scratch) = scratch.borrow().take_glwe_scratch(a);
                module.glwe_automorphism(&mut baby, a, key, &mut baby_scratch.borrow());
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left_pvec(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
        }
    }
}
