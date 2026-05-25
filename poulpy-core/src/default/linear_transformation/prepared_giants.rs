//! Prepared giant-step evaluation.
//!
//! Main loop from docs/lt_bsgs.md §6.3: each giant bucket first runs PROD (a
//! DFT-domain inner product over prepared baby rotations and prepared
//! diagonals, IDFT'd into a `VecZnxBig`), then ROT rotates that bucket and
//! folds it into a BIG accumulator. The PROD result rides BIG end-to-end when
//! the key base matches the ciphertext base; only the mask columns are dropped
//! to SMALL inside ROT, because gadget decomposition needs limb-aligned input.

use poulpy_hal::{
    api::{
        Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, ZnxInfos},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain,
    default::{
        keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
        linear_transformation::{
            baby_steps::GLWEPreparedBabyStepHelper,
            inner_product::glwe_accumulate_prepared_inner_product_big,
            lazy::{glwe_big_add_big_assign, glwe_lazy_giant_automorphism_from_big, glwe_normalize_big_into},
        },
        operations::cnv_offset_to_limb_offset,
    },
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        ModuleCoreAlloc, prepared::GGLWEPreparedToBackendRef,
    },
};

use super::{GLWELinearTransform, GLWEPreparedLinearTransform};

#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_prepared_linear_transform_with_babies<BE, M, R, P, H, K, B>(
    module: &M,
    res: &mut R,
    lt: &GLWELinearTransform<P>,
    prepared: &GLWEPreparedLinearTransform<BE>,
    babies: &B,
    cnv_offset: usize,
    key_size: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + Convolution<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + GLWEKeyswitchInternal<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxIdftApplyTmpBytes
        + GLWEMulPlain<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    B: GLWEPreparedBabyStepHelper<BE>,
{
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();

    // PROD writes its result in the plaintext's base2k. All diagonals share
    // one base2k (asserted below), so the convolution alignment is uniform.
    let first_plaintext = lt
        .giant_steps
        .iter()
        .flat_map(|gs| gs.diagonals.iter())
        .map(|d| &d.plaintext)
        .next()
        .expect("linear transformation has no diagonals");
    let prod_base2k = first_plaintext.base2k();
    let (first_gs, _) = prepared
        .giant_steps
        .split_first()
        .expect("linear transformation has no giant steps");
    let (&first_baby, first_diag) = first_gs
        .diagonals
        .iter()
        .next()
        .expect("linear transformation giant step is empty");
    let baby_size = babies.baby_step(first_baby).size();
    let diagonal_size = first_diag.size();
    let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, prod_base2k.as_usize());
    let prod_size = baby_size + diagonal_size - cnv_offset_hi;

    let has_nonzero_giant_rotation = prepared.giant_steps.iter().any(|gs| gs.rot != 0);
    // `automorphism_key_infos()` panics on an empty key map (legitimate for
    // an identity-only transform), so only consult it when at least one giant
    // rotation actually needs a key.
    let (use_lazy_giant_rotation, key_size_effective) = if has_nonzero_giant_rotation {
        let key_infos = keys.automorphism_key_infos();
        let key_base2k = key_infos.base2k();
        let key_size_effective = key_size.min(key_infos.size());
        (res_base2k == key_base2k && prod_base2k == key_base2k, key_size_effective)
    } else {
        // No giant rotation: BIG-flow accumulator is always valid (no key required).
        (true, key_size)
    };
    let use_final_lazy_accumulator = !has_nonzero_giant_rotation || use_lazy_giant_rotation;
    let lazy_size = if use_lazy_giant_rotation {
        key_size_effective.max(prod_size)
    } else {
        res.size().max(prod_size)
    };

    // `prod_big` is per-step scratch: PROD writes a full BIG result every
    // giant step, overwriting via IDFT, so it doesn't need zero-init.
    let scratch = scratch.borrow();
    let (mut prod_big, mut scratch_phase) = scratch.take_vec_znx_big_scratch(module, cols, prod_size);

    // The cross-step BIG accumulator and the ROT temp must start zeroed (we
    // use `vec_znx_big_add_assign` on the first iteration). The HAL has no
    // `vec_znx_big_zero` primitive yet, so we lean on `vec_znx_big_alloc`'s
    // `alloc_zeroed_bytes` to get a clean buffer. These are one-per-call,
    // not per-step (Pass 1 already eliminated the per-step `rot_big` churn).
    let mut lazy_acc = use_final_lazy_accumulator.then(|| module.vec_znx_big_alloc(cols, lazy_size));
    let mut rot_big =
        (use_lazy_giant_rotation && has_nonzero_giant_rotation).then(|| module.vec_znx_big_alloc(cols, key_size_effective));
    let mut fallback_acc: Option<GLWE<BE::OwnedBuf>> = (!use_final_lazy_accumulator).then(|| module.glwe_alloc_from_infos(res));

    let mut res_initialized = false;
    let mut terms: Vec<(i64, &B::BabyStep, _)> = Vec::with_capacity(prepared.baby_steps.len());

    for gs in &prepared.giant_steps {
        // PROD in lt_bsgs.md §6.3: multiply the prepared baby rotations by the
        // prepared diagonals for this giant bucket and sum in DFT space.
        terms.clear();
        for (&baby, prepared_plaintext) in &gs.diagonals {
            assert_eq!(prepared_plaintext.size(), diagonal_size);
            terms.push((baby, babies.baby_step(baby), prepared_plaintext));
        }
        {
            let mut prod_big_backend = prod_big.to_backend_mut();
            glwe_accumulate_prepared_inner_product_big(module, cnv_offset_hi, &mut prod_big_backend, &terms, &mut scratch_phase);
        }

        if let Some(lazy_acc) = lazy_acc.as_mut() {
            // ROT in lt_bsgs.md §6.3: keep giant contributions in BIG and defer
            // the single final normalize to §6.4. `lazy_acc` was zero-initialized
            // by its `vec_znx_big_alloc`, so the first `add_assign` cleanly
            // initializes the accumulator.
            if gs.rot == 0 {
                let prod_big_ref = prod_big.to_backend_ref();
                let mut lazy_acc_mut = lazy_acc.to_backend_mut();
                glwe_big_add_big_assign(module, &mut lazy_acc_mut, &prod_big_ref);
                res_initialized = true;
            } else {
                let rot_big = rot_big.as_mut().expect("rot_big allocated");
                let key: &K = keys
                    .get_automorphism_key(gs.rot)
                    .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {}", gs.rot));
                {
                    let mut rot_big_backend = rot_big.to_backend_mut();
                    let prod_big_ref = prod_big.to_backend_ref();
                    glwe_lazy_giant_automorphism_from_big(
                        module,
                        &mut rot_big_backend,
                        &prod_big_ref,
                        prod_base2k.as_usize(),
                        key,
                        key_size,
                        &mut scratch_phase,
                    );
                }
                let rot_big_ref = rot_big.to_backend_ref();
                let mut lazy_acc_mut = lazy_acc.to_backend_mut();
                glwe_big_add_big_assign(module, &mut lazy_acc_mut, &rot_big_ref);
                res_initialized = true;
            }
        } else {
            // Fallback for incompatible bases: normalize PROD into the SMALL
            // `fallback_acc` (applying `cnv_offset_lo`), then use the regular
            // normalized path.
            let acc = fallback_acc.as_mut().expect("fallback_acc allocated");
            {
                let prod_big_ref = prod_big.to_backend_ref();
                let mut acc_backend = <GLWE<BE::OwnedBuf> as GLWEToBackendMut<BE>>::to_backend_mut(acc);
                for col in 0..cols {
                    module.vec_znx_big_normalize(
                        &mut acc_backend.data,
                        res_base2k.as_usize(),
                        cnv_offset_lo,
                        col,
                        &prod_big_ref,
                        prod_base2k.as_usize(),
                        col,
                        &mut scratch_phase.borrow(),
                    );
                }
            }

            if gs.rot != 0 {
                let key: &K = keys
                    .get_automorphism_key(gs.rot)
                    .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {}", gs.rot));
                module.glwe_automorphism_assign(acc, key, key_size, &mut scratch_phase);
            }

            if res_initialized {
                module.glwe_add_assign(res, acc);
            } else {
                module.glwe_copy(res, acc);
                res_initialized = true;
            }
        }
    }

    if let Some(lazy_acc) = lazy_acc.as_ref() {
        // Finalize in lt_bsgs.md §6.4: one BIG -> SMALL normalization, folding
        // `cnv_offset_lo` (PROD never applied it) into this last rounding.
        let lazy_acc_ref = lazy_acc.to_backend_ref();
        glwe_normalize_big_into(
            module,
            res,
            &lazy_acc_ref,
            prod_base2k.as_usize(),
            cnv_offset_lo,
            &mut scratch_phase,
        );
    }
}
