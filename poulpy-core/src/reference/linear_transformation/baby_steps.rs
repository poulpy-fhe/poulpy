//! Baby-step rotation materialization.
//!
//! Implements docs/guides/linear_transformation.md. The non-trivial baby rotations share one
//! DFT of the input mask columns, then each key performs VMP -> IDFT -> add body
//! -> normalize -> automorphism. The resulting SMALL ciphertexts are prepared as
//! `CnvPVecL` so every giant step can reuse them in convolution form.

//! Baby-step rotation materialization (the prepared LHS).
//!
//! Implements docs/guides/linear_transformation.md. The non-trivial baby rotations share one
//! DFT of the input mask columns, then each key performs VMP -> IDFT -> add body
//! -> normalize -> automorphism. The resulting SMALL ciphertexts are prepared
//! as `CnvPVecL` and stored in a [`LinearTransformationBabySteps`] (whose
//! definition lives in [`crate::layouts`]); this module owns the HAL-dependent
//! allocator and population routines.

use poulpy_hal::layouts::CnvPVecLToBackendMut;
use std::collections::BTreeMap;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssign, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxDftZero, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    execution::{for_each_with_scratch, scratch_workers, worker_count, worker_scratch_bytes},
    layouts::{
        Backend, GaloisElement, PrepareHint, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, vec_znx_dft_backend_ref_from_mut,
    },
};

use crate::gadget::{gadget_dft_apply, gadget_dft_apply_tmp_bytes, gadget_input_size};
use crate::{
    GLWEAutomorphism, GLWENormalize, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    gadget::gglwe_product_output_size,
    layouts::{
        GGLWEInfos, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        prepared::{GGLWEPreparedBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
    reference::keyswitching::GGLWEProductReference,
};

use super::{BsgsRotations, LinearTransformationBabySteps, LinearTransformationLayout, LinearTransformationPlan};
use crate::layouts::{ModuleCoreAlloc, digits_aligned, operand_galois_element};

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
        M: ModuleN + CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        Self::alloc_with(module, baby_steps, BsgsRotations::PerRotation, a)
    }

    /// Pre-allocates a baby-step cache prepared as `rotations` says, holding `baby_steps` and,
    /// with [`BsgsRotations::PowersOfTwo`], every rotation a doubling pass reads to reach them.
    pub fn alloc_with<M, A>(module: &M, baby_steps: &[i64], rotations: BsgsRotations, a: &A) -> Self
    where
        M: ModuleN + CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        let cols = a.rank().as_usize() + 1;
        let size = a.size();
        let mut values = BTreeMap::new();
        for rot in rotations.baby_closure(baby_steps.iter().copied()) {
            values
                .entry(rot)
                .or_insert_with(|| module.cnv_pvec_left_alloc(a.n().as_usize(), cols, size, PrepareHint::Reuse));
        }
        Self { values, rotations }
    }

    /// Pre-allocates the baby-step cache the transformation `lt` reads, prepared as it says.
    pub fn alloc_for<M, A, P>(module: &M, lt: &crate::LinearTransformation<P>, a: &A) -> Self
    where
        M: ModuleN + CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        Self::alloc_with(module, lt.baby_steps(), lt.rotations, a)
    }

    /// Convenience: pre-allocates from a [`LinearTransformationLayout`].
    pub fn alloc_from_layout<M, A>(module: &M, layout: &LinearTransformationLayout, a: &A) -> Self
    where
        M: ModuleN + CnvPVecAlloc<BE>,
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
        + GGLWEProductReference<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + GLWENormalize<BE>,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let a_size = a_infos.size();
    let key_size = gglwe_product_output_size::<BE, _, _, _>(a_infos, a_infos, key_infos);
    let baby = module.glwe_bytes_of_from_infos(a_infos);
    let prepare = module.cnv_prepare_left_tmp_bytes(a_infos.size(), a_infos.size());

    let (base2k, k_digit, dnum) = (key_infos.base2k(), key_infos.k_digit(), key_infos.dnum());
    let operand_size = gadget_input_size(base2k, k_digit, dnum, a_infos.k(), a_size);
    let hoisted_a_dft = module.bytes_of_vec_znx_dft(module.n(), cols - 1, operand_size);
    let hoisted_rot = module.bytes_of_vec_znx_dft(module.n(), cols, key_size)
        + module
            .gglwe_product_dft_tmp_bytes_reference(key_size, operand_size, key_infos)
            .max(module.vec_znx_idft_normalize_tmp_bytes(a_size, key_size));
    let hoisted_worker = worker_scratch_bytes::<BE>(baby + hoisted_rot.max(prepare));
    let hoisted = hoisted_a_dft
        + (scratch_workers::<BE::TaskExecutor>(BABY_ROTATION_WORKERS) * hoisted_worker).max(gadget_dft_apply_tmp_bytes::<BE, _>(
            module,
            base2k,
            k_digit,
            dnum,
            a_infos.k(),
            a_size,
        ));

    let fallback = baby + module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos).max(prepare);
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a_infos))
        + hoisted.max(fallback).max(prepare).max(module.glwe_normalize_tmp_bytes())
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
        + GGLWEProductReference<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let cols = a.rank().as_usize() + 1;
    assert_eq!(key_ref.base2k(), a.base2k());

    // `key_size` is this key's own product width; limbs above it stay zeroed.
    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(a.n().as_usize(), cols, key_size);
    crate::layouts::assert_no_row_offset("glwe_hoisted_baby_rotation", &key_ref);
    module.gglwe_product_dft_reference(&mut res_dft, a_dft_ref, key_ref, 1, &mut scratch_1.borrow());

    let baby_base2k = baby.base2k().as_usize();
    let baby_k = baby.k().as_usize();
    let a_base2k = a.base2k().as_usize();
    {
        let mut baby_ref = baby.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_idft_normalize(
                &mut baby_ref.data,
                baby_base2k,
                baby_k,
                0,
                col,
                &mut res_dft,
                col,
                a_base2k,
                (col == 0).then_some((&a.data, 0)),
                &mut scratch_1.borrow(),
            );
        }
        for col in 0..cols {
            module.vec_znx_automorphism_assign(key_p, &mut baby_ref.data, col, &mut scratch_1.borrow());
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
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Convolution<BE>
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GGLWEProductReference<BE>
        + ModuleN
        + VecZnxAutomorphismAssign<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + GLWENormalize<BE>
        + Sync,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
{
    let (mut a_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(a);
    let a_ref = if a.is_canonical() {
        a.to_backend_ref()
    } else {
        module.glwe_normalize(&mut a_tmp, a, &mut scratch.borrow());
        a_tmp.to_backend_ref()
    };
    let (a, scratch) = (&&a_ref, &mut scratch);
    if cache.rotations != BsgsRotations::PerRotation {
        return glwe_prepare_baby_steps_doubling(module, cache, a, keys, scratch);
    }
    let cols = a.rank().as_usize() + 1;
    let a_size = a.size();
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
                keys.get_automorphism_key(operand_galois_element(module, a.n().as_usize(), rot), a.k())
                    .unwrap_or_else(|e| panic!("baby-step rotation {rot}: {e}"))
            })
        })
        .collect();
    // The hoisted operand is laid out once per class of keys that read the same
    // one: whole-limb digits read the input limbs whatever their width, and
    // other widths a digit grid of their width over the digits they reach. In
    // a class, the operand is the one each key's own scratch query sizes, and
    // each key reads exactly the digits it would decompose alone.
    let operand_class = |key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>| -> (u32, usize) {
        if digits_aligned(key.base2k(), key.k_digit()) {
            (0, a_size)
        } else {
            (
                key.k_digit().0,
                gadget_input_size(a.base2k(), key.k_digit(), key.dnum(), a.k(), a_size),
            )
        }
    };
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
        // Every class in turn, the unrotated entry with the first one.
        let classes: Vec<usize> = key_refs
            .iter()
            .map(|key| {
                key.as_ref().map_or(usize::MAX, |key| {
                    let class = operand_class(key);
                    key_refs
                        .iter()
                        .position(|other| other.as_ref().is_some_and(|other| operand_class(other) == class))
                        .unwrap()
                })
            })
            .collect();
        let mut representatives: Vec<usize> = classes.iter().copied().filter(|&c| c != usize::MAX).collect();
        representatives.sort_unstable();
        representatives.dedup();
        let mut entries: Vec<Option<_>> = cache
            .values
            .iter_mut()
            .map(|(&rot, prepared)| Some((rot, prepared)))
            .collect();
        for (pass, &representative) in representatives.iter().enumerate() {
            let class_key = key_refs[representative].as_ref().unwrap();
            let (k_digit, dnum) = (class_key.k_digit(), class_key.dnum());
            let mut tasks: Vec<_> = entries
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| classes[*index] == representative || (pass == 0 && classes[*index] == usize::MAX))
                .map(|(index, entry)| (index, entry.take().unwrap()))
                .collect();
            let scratch = scratch.borrow();
            let operand_size = gadget_input_size(a.base2k(), k_digit, dnum, a.k(), a_size);
            let (mut a_dft, mut loop_scratch) = scratch.take_vec_znx_dft_scratch(a.n().as_usize(), cols - 1, operand_size);
            for col_i in 0..cols - 1 {
                gadget_dft_apply(
                    module,
                    &mut a_dft,
                    col_i,
                    &a_ref.data,
                    col_i + 1,
                    a.base2k(),
                    k_digit,
                    dnum,
                    a.k(),
                    &mut loop_scratch.borrow(),
                );
            }
            let a_dft_ref = a_dft.to_backend_ref();
            let workers = worker_count::<BE::TaskExecutor>(BABY_ROTATION_WORKERS, tasks.len());
            // Sized exactly as the per-key scratch query sizes one rotation, so the
            // caller's max over its key layouts covers whatever this loop visits.
            let rotation_bytes = tasks
                .iter()
                .filter_map(|(index, _)| key_refs[*index].as_ref().map(|key| (key, key_sizes[*index])))
                .map(|(key, key_size)| {
                    module.bytes_of_vec_znx_dft(module.n(), cols, key_size)
                        + module
                            .gglwe_product_dft_tmp_bytes_reference(key_size, operand_size, key)
                            .max(module.vec_znx_idft_normalize_tmp_bytes(a_size, key_size))
                })
                .max()
                .unwrap_or(0);
            let task_bytes = worker_scratch_bytes::<BE>(
                module.glwe_bytes_of_from_infos(a) + rotation_bytes.max(module.cnv_prepare_left_tmp_bytes(a_size, a_size)),
            );
            let (worker_scratch, _) = loop_scratch.borrow().split(workers, task_bytes);
            for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(&mut tasks, 0, worker_scratch, &|_, task, task_scratch| {
                let (index, (rot, prepared)) = task;
                assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
                assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
                if *rot == 0 {
                    module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, task_scratch);
                } else {
                    let key = key_refs[*index].as_ref().unwrap();
                    let (mut baby, mut baby_scratch) = task_scratch.borrow().take_glwe_scratch(&a_ref);
                    glwe_hoisted_baby_rotation(
                        module,
                        &mut baby,
                        &a_ref,
                        &a_dft_ref,
                        key.p,
                        &key.key,
                        key_sizes[*index],
                        &mut baby_scratch.borrow(),
                    );
                    let baby_ref = baby.to_backend_ref();
                    module.cnv_prepare_left(&mut prepared.to_backend_mut(), &baby_ref.data, &mut baby_scratch.borrow());
                }
            });
        }
    } else {
        for ((&rot, prepared), key) in cache.values.iter_mut().zip(&key_refs) {
            assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
            assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
            if rot == 0 {
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, scratch);
            } else {
                let (mut baby, mut baby_scratch) = scratch.borrow().take_glwe_scratch(a);
                let key = key.as_ref().unwrap();
                module.glwe_automorphism(&mut baby, a, key, &mut baby_scratch.borrow());
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &baby_ref.data, &mut baby_scratch.borrow());
            }
        }
    }
}

/// [`glwe_prepare_linear_transformation_baby_steps`] for [`BsgsRotations::PowersOfTwo`]: the
/// rotations of a doubling pass rotate earlier ones with the pass's key, each source laid out
/// and switched on its own, a task each on a parallel executor. The rotations are held as
/// ciphertexts until the last pass.
#[allow(clippy::type_complexity)]
fn glwe_prepare_baby_steps_doubling<BE, M, H>(
    module: &M,
    cache: &mut LinearTransformationBabySteps<BE>,
    a: &GLWEBackendRef<'_, BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Convolution<BE>
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GGLWEProductReference<BE>
        + ModuleN
        + VecZnxAutomorphismAssign<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + GLWENormalize<BE>
        + Sync,
    H: GetAutomorphismKey<BE>,
{
    let n = a.n().as_usize();
    let cols = a.rank().as_usize() + 1;
    let a_size = a.size();
    let rotations: Vec<i64> = cache.values.keys().copied().collect();
    let passes = cache.rotations.baby_passes(rotations.iter().copied());
    let pass_keys: Vec<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> = passes
        .iter()
        .map(|&(key_rot, _)| {
            keys.get_automorphism_key(operand_galois_element(module, n, key_rot), a.k())
                .unwrap_or_else(|e| panic!("baby-step doubling {key_rot}: {e}"))
        })
        .collect();
    // A pass switches its sources together when its key has their base2k.
    let hoisted = |key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>| key.base2k() == a.base2k();
    // The rotations as ciphertexts and the buffers of the widest pass, carved from the scratch
    // beside the work of a rotation ([`glwe_prepare_baby_doubling_tmp_bytes`]).
    let need = baby_doubling_rotations_bytes::<BE, _, _>(module, a, rotations.len())
        + passes
            .iter()
            .zip(&pass_keys)
            .filter(|(_, key)| hoisted(key))
            .map(|((_, pairs), key)| baby_doubling_pass_bytes::<BE, _, _, _>(module, a, key, pairs.len()))
            .max()
            .unwrap_or(0);
    let work = pass_keys
        .iter()
        .map(|key| glwe_prepare_linear_transformation_baby_steps_tmp_bytes::<BE, _, _, _>(module, a, key))
        .max()
        .unwrap_or(0);
    assert!(
        scratch.available() >= need + work,
        "glwe_prepare_linear_transformation_baby_steps: scratch below its query"
    );
    let (mut rotated, mut work) = scratch.borrow().take_glwe_slice_scratch(rotations.len(), a);
    let index = |rot: i64| rotations.binary_search(&rot).unwrap();
    for ((_, pairs), key) in passes.iter().zip(&pass_keys) {
        if !hoisted(key) {
            for &(source, target) in pairs {
                let target = index(target);
                if source == 0 {
                    module.glwe_automorphism(&mut rotated[target], &a, key, &mut work);
                } else {
                    // A source always precedes its target.
                    let (sources, targets) = rotated.split_at_mut(target);
                    let source = sources[index(source)].to_backend_ref();
                    module.glwe_automorphism(&mut targets[0], &&source, key, &mut work);
                }
            }
            continue;
        }
        crate::layouts::assert_no_row_offset("glwe_hoisted_baby_rotation", &key.key);
        let m = pairs.len();
        let (k_digit, dnum) = (key.k_digit(), key.dnum());
        let operand_size = gadget_input_size(a.base2k(), k_digit, dnum, a.k(), a_size);
        let key_size = gglwe_product_output_size::<BE, _, _, _>(a, a, key);
        let pass = work.borrow();
        let (mut operands, pass) = pass.take_vec_znx_dft_slice_scratch(n, m, cols - 1, operand_size);
        let (mut products, pass) = pass.take_vec_znx_dft_slice_scratch(n, m, cols, key_size);
        let (mut normalized, mut pass) = pass.take_vec_znx_slice_scratch(m, n, cols, a_size);
        {
            // Every source reborrowed alike, the input as the rotations.
            let sources: Vec<GLWEBackendRef<'_, BE>> = pairs
                .iter()
                .map(|&(source, _)| {
                    if source == 0 {
                        crate::layouts::glwe_backend_ref_from_ref::<BE>(a)
                    } else {
                        rotated[index(source)].to_backend_ref()
                    }
                })
                .collect();
            // The rotation of `source` before its automorphism: its operand laid out, its
            // product by the key, normalized with its body.
            let (base2k, k) = (a.base2k().as_usize(), a.k().as_usize());
            let switch = |operand: &mut VecZnxDftBackendMut<'_, BE>,
                          product: &mut VecZnxDftBackendMut<'_, BE>,
                          normalized: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
                          source: &GLWEBackendRef<'_, BE>,
                          scratch: &mut ScratchArena<'_, BE>| {
                for col_i in 0..cols - 1 {
                    gadget_dft_apply(
                        module,
                        operand,
                        col_i,
                        &source.data,
                        col_i + 1,
                        a.base2k(),
                        k_digit,
                        dnum,
                        a.k(),
                        &mut scratch.borrow(),
                    );
                }
                module.gglwe_product_dft_reference(
                    product,
                    &vec_znx_dft_backend_ref_from_mut(operand),
                    &key.key,
                    1,
                    &mut scratch.borrow(),
                );
                for col in 0..cols {
                    module.vec_znx_idft_normalize(
                        normalized,
                        base2k,
                        k,
                        0,
                        col,
                        product,
                        col,
                        base2k,
                        (col == 0).then_some((&source.data, 0)),
                        &mut scratch.borrow(),
                    );
                }
            };
            let task_bytes = worker_scratch_bytes::<BE>(
                gadget_dft_apply_tmp_bytes::<BE, _>(module, a.base2k(), k_digit, dnum, a.k(), a_size)
                    .max(module.gglwe_product_dft_tmp_bytes_reference(key_size, operand_size, key))
                    .max(module.vec_znx_idft_normalize_tmp_bytes(a_size, key_size)),
            );
            let workers = worker_count::<BE::TaskExecutor>(BABY_ROTATION_WORKERS, m).min(pass.available() / task_bytes.max(1));
            let mut tasks: Vec<_> = operands
                .iter_mut()
                .zip(products.iter_mut())
                .zip(normalized.iter_mut())
                .zip(&sources)
                .collect();
            if workers > 1 {
                // On a parallel executor every source is a task of its own.
                let (worker_scratch, _) = pass.borrow().split(workers, task_bytes);
                for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(
                    &mut tasks,
                    0,
                    worker_scratch,
                    &|_, (((operand, product), normalized), source), task_scratch| {
                        switch(
                            &mut operand.to_backend_mut(),
                            &mut product.to_backend_mut(),
                            &mut normalized.to_backend_mut(),
                            source,
                            task_scratch,
                        );
                    },
                );
            } else {
                for (((operand, product), normalized), source) in tasks {
                    switch(
                        &mut operand.to_backend_mut(),
                        &mut product.to_backend_mut(),
                        &mut normalized.to_backend_mut(),
                        source,
                        &mut pass,
                    );
                }
            }
        }
        // Each rotation stored through the pass's automorphism.
        for (normalized, &(_, target)) in normalized.iter().zip(pairs) {
            let normalized = normalized.to_backend_ref();
            let mut target = rotated[index(target)].to_backend_mut();
            for col in 0..cols {
                module.vec_znx_automorphism(key.p, &mut target.data, col, &normalized, col);
            }
        }
    }
    // The prepared cache, a task per rotation on a parallel executor.
    let sources: Vec<GLWEBackendRef<'_, BE>> = rotations
        .iter()
        .map(|&rot| {
            if rot == 0 {
                crate::layouts::glwe_backend_ref_from_ref::<BE>(a)
            } else {
                rotated[index(rot)].to_backend_ref()
            }
        })
        .collect();
    let mut tasks: Vec<_> = cache.values.values_mut().zip(&sources).collect();
    let task_bytes = worker_scratch_bytes::<BE>(module.cnv_prepare_left_tmp_bytes(a_size, a_size));
    let workers = worker_count::<BE::TaskExecutor>(BABY_ROTATION_WORKERS, tasks.len())
        .min(work.available() / task_bytes.max(1))
        .max(1);
    let (worker_scratch, _) = work.borrow().split(workers, task_bytes);
    for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(&mut tasks, 0, worker_scratch, &|_, (prepared, source), task_scratch| {
        assert_eq!(prepared.cols(), cols, "prepared baby cache has wrong column count");
        assert_eq!(prepared.size(), a_size, "prepared baby cache has wrong size");
        module.cnv_prepare_left(&mut prepared.to_backend_mut(), &source.data, task_scratch);
    });
}

/// Bytes of the rotations a doubling holds as ciphertexts of the layout of `a`.
fn baby_doubling_rotations_bytes<BE, M, A>(module: &M, a: &A, rotations: usize) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>,
    A: GLWEInfos,
{
    rotations * BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
}

/// Bytes of the buffers of a doubling pass that switches `sources` sources of the layout of `a`
/// together with `key`: the operand, the product and the normalized rotation of each.
fn baby_doubling_pass_bytes<BE, M, A, K>(module: &M, a: &A, key: &K, sources: usize) -> usize
where
    BE: Backend,
    M: VecZnxDftBytesOf,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let (n, cols, a_size) = (a.n().as_usize(), a.rank().as_usize() + 1, a.size());
    let operand_size = gadget_input_size(a.base2k(), key.k_digit(), key.dnum(), a.k(), a_size);
    let key_size = gglwe_product_output_size::<BE, _, _, _>(a, a, key);
    sources
        * (BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols - 1, operand_size))
            + BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, key_size))
            + BE::scratch_aligned(BE::bytes_of_vec_znx(n, cols, a_size)))
}

/// Bytes the doubling of a cache on [`BsgsRotations::PowersOfTwo`] for the schedule `plan`
/// takes from the scratch on top of the per-rotation budget, with keys of the layout `key`:
/// the source normalized, the rotations as ciphertexts and the buffers of the widest pass,
/// beside the work of a rotation, which that budget covers. Zero for
/// [`BsgsRotations::PerRotation`].
pub(super) fn glwe_prepare_baby_doubling_tmp_bytes<BE, M, A, K>(
    module: &M,
    a: &A,
    plan: &LinearTransformationPlan,
    key: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + VecZnxDftBytesOf,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    if plan.rotations == BsgsRotations::PerRotation {
        return 0;
    }
    // A cache holds the identity whether or not the plan reads it.
    let rotations = plan.rotations.baby_closure(plan.baby_steps.iter().copied().chain([0]));
    let widest_pass = plan
        .rotations
        .baby_passes(rotations.iter().copied())
        .iter()
        .map(|(_, pairs)| baby_doubling_pass_bytes::<BE, _, _, _>(module, a, key, pairs.len()))
        .max()
        .unwrap_or(0);
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a))
        + baby_doubling_rotations_bytes::<BE, _, _>(module, a, rotations.len())
        + widest_pass
}
