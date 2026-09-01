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
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssignBackend, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes,
    },
    execution::{for_each_with_scratch, scratch_workers, worker_count, worker_scratch_bytes},
    layouts::{Backend, GaloisElement, ScratchArena, VecZnxDftBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWEShift, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    default::{
        keyswitching::{GGLWEProductDefault, gglwe_product_output_size},
        operations::msb_mask_bottom_limb,
    },
    layouts::{
        GGLWEInfos, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        prepared::{GGLWEPreparedBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
};

use super::{LinearTransformationBabySteps, LinearTransformationLayout};

const BABY_ROTATION_WORKERS: usize = 4;

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
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeConsumeTmpBytes
        + GLWEShift<BE>,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let a_size = a_infos.size();
    let key_size = gglwe_product_output_size::<BE, _, _, _>(a_infos, a_infos, key_infos);
    let baby = module.glwe_bytes_of_from_infos(a_infos);
    let prepare = module.cnv_prepare_left_tmp_bytes(a_infos.size(), a_infos.size());

    let hoisted_a_dft = module.bytes_of_vec_znx_dft(cols - 1, a_size);
    let hoisted_rot = module.bytes_of_vec_znx_dft(cols, key_size)
        + module
            .gglwe_product_dft_tmp_bytes_default(key_size, a_size, key_infos)
            .max(module.vec_znx_idft_normalize_consume_tmp_bytes(a_size, key_size));
    let hoisted_worker = worker_scratch_bytes::<BE>(baby + hoisted_rot.max(prepare));
    let hoisted = hoisted_a_dft + scratch_workers::<BE::TaskExecutor>(BABY_ROTATION_WORKERS) * hoisted_worker;

    let fallback = baby + module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos).max(prepare);
    let prepare_babies = hoisted.max(fallback).max(prepare);
    let padding =
        (a_infos.base2k().as_usize() - a_infos.k().as_usize() % a_infos.base2k().as_usize()) % a_infos.base2k().as_usize();
    if padding == 0 {
        prepare_babies
    } else {
        baby + prepare_babies.max(module.glwe_shift_tmp_bytes())
    }
}

#[allow(clippy::too_many_arguments)]
fn glwe_hoisted_baby_rotation<BE, M, R>(
    module: &M,
    baby: &mut R,
    a: &GLWEBackendRef<'_, BE>,
    a_dft_ref: &VecZnxDftBackendRef<'_, BE>,
    key_p: i64,
    key_ref: &GGLWEPreparedBackendRef<'_, BE>,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GaloisElement
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalizeConsume<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let cols = a.rank().as_usize() + 1;
    assert_eq!(key_ref.base2k(), a.base2k());

    // `key_size` is this key's own product width; limbs above it stay zeroed.
    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    module.gglwe_product_dft_default(&mut res_dft, a_dft_ref, key_ref, 1, &mut scratch_1.borrow());

    let baby_base2k = baby.base2k().as_usize();
    let a_base2k = a.base2k().as_usize();
    {
        let mut baby_ref = baby.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_idft_normalize_consume(
                &mut baby_ref.data,
                baby_base2k,
                col,
                &mut res_dft,
                col,
                a_base2k,
                (col == 0).then_some((&a.data, 0)),
                &mut scratch_1.borrow(),
            );
        }
        for col in 0..cols {
            module.vec_znx_automorphism_assign_backend(key_p, &mut baby_ref.data, col, &mut scratch_1.borrow());
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
pub(super) fn glwe_prepare_linear_transformation_baby_steps<BE, M, A, H>(
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
        + GLWECopy<BE>
        + GLWEShift<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalizeConsume<BE>
        + VecZnxIdftNormalizeConsumeTmpBytes
        + Sync,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
{
    let base2k = a.base2k().as_usize();
    let padding = (base2k - a.k().as_usize() % base2k) % base2k;
    if padding == 0 {
        glwe_prepare_linear_transformation_baby_steps_inner(module, cache, a, keys, scratch);
        return;
    }

    // The DFT and body-add paths below otherwise consume inactive bits from the partial bottom limb.
    let (mut a_clean, mut clean_scratch) = scratch.borrow().take_glwe_scratch(a);
    module.glwe_copy(&mut a_clean, a);
    module.glwe_rsh(padding, &mut a_clean, &mut clean_scratch.borrow());
    module.glwe_lsh_assign(&mut a_clean, padding, &mut clean_scratch.borrow());
    glwe_prepare_linear_transformation_baby_steps_inner(module, cache, &a_clean, keys, &mut clean_scratch);
}

#[allow(clippy::too_many_arguments)]
fn glwe_prepare_linear_transformation_baby_steps_inner<BE, M, A, H>(
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
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalizeConsume<BE>
        + VecZnxIdftNormalizeConsumeTmpBytes
        + Sync,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
{
    let cols = a.rank().as_usize() + 1;
    let a_size = a.size();
    let mask = msb_mask_bottom_limb(a.base2k().as_usize(), a.k().as_usize());
    // Baby rotations rotate the source, so their keys are the ones the source's
    // precision asks for. Every key is resolved once, in cache order, because
    // the rotations need not share a layout: the hoisted route is only taken
    // when all of them carry the source's base2k, and every rotation is sized by
    // its own key. A shared width would evaluate one key's product against
    // another key's output size, a pair the per-key scratch query never sizes.
    let key_refs: Vec<Option<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>> = cache
        .values
        .keys()
        .map(|&rot| {
            (rot != 0).then(|| {
                keys.get_automorphism_key(module.galois_element(rot), a.k())
                    .unwrap_or_else(|e| panic!("baby-step rotation {rot}: {e}"))
            })
        })
        .collect();
    let use_hoisted =
        key_refs.iter().flatten().next().is_some() && key_refs.iter().flatten().all(|key| key.base2k() == a.base2k());
    let key_sizes: Vec<usize> = key_refs
        .iter()
        .map(|key| {
            key.as_ref()
                .map_or(a_size, |key| gglwe_product_output_size::<BE, _, _, _>(a, a, key))
        })
        .collect();

    if use_hoisted {
        let scratch = scratch.borrow();
        let (mut a_dft, mut loop_scratch) = scratch.take_vec_znx_dft_scratch(module, cols - 1, a_size);
        let a_ref = a.to_backend_ref();
        for col_i in 0..cols - 1 {
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, &a_ref.data, col_i + 1);
        }
        let a_dft_ref = a_dft.to_backend_ref();
        let mut tasks: Vec<_> = cache.values.iter_mut().map(|(&rot, prepared)| (rot, prepared)).collect();
        let workers = worker_count::<BE::TaskExecutor>(BABY_ROTATION_WORKERS, tasks.len());
        // Sized exactly as the per-key scratch query sizes one rotation, so the
        // caller's max over its key layouts covers whatever this loop visits.
        let rotation_bytes = key_refs
            .iter()
            .zip(&key_sizes)
            .filter_map(|(key, &key_size)| key.as_ref().map(|key| (key, key_size)))
            .map(|(key, key_size)| {
                module.bytes_of_vec_znx_dft(cols, key_size)
                    + module
                        .gglwe_product_dft_tmp_bytes_default(key_size, a_size, key)
                        .max(module.vec_znx_idft_normalize_consume_tmp_bytes(a_size, key_size))
            })
            .max()
            .unwrap_or(0);
        let task_bytes = worker_scratch_bytes::<BE>(
            module.glwe_bytes_of_from_infos(a) + rotation_bytes.max(module.cnv_prepare_left_tmp_bytes(a_size, a_size)),
        );
        let (worker_scratch, _) = loop_scratch.borrow().split(workers, task_bytes);
        for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(&mut tasks, 0, worker_scratch, &|index, task, task_scratch| {
            let (rot, prepared) = task;
            assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
            assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
            if *rot == 0 {
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, mask, task_scratch);
            } else {
                let key = key_refs[index].as_ref().unwrap();
                let (mut baby, mut baby_scratch) = task_scratch.borrow().take_glwe_scratch(&a_ref);
                glwe_hoisted_baby_rotation(
                    module,
                    &mut baby,
                    &a_ref,
                    &a_dft_ref,
                    key.p,
                    &key.key,
                    key_sizes[index],
                    &mut baby_scratch.borrow(),
                );
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
        });
    } else {
        for ((&rot, prepared), key) in cache.values.iter_mut().zip(&key_refs) {
            assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
            assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
            if rot == 0 {
                let a_ref = a.to_backend_ref();
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, mask, scratch);
            } else {
                let (mut baby, mut baby_scratch) = scratch.borrow().take_glwe_scratch(a);
                let key = key.as_ref().unwrap();
                module.glwe_automorphism(&mut baby, a, key, &mut baby_scratch.borrow());
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
        }
    }
}
