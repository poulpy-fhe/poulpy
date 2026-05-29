//! Prepared giant-step evaluation.
//!
//! Main loop from docs/lt_bsgs.md §6.3: each giant bucket first runs PROD (a
//! DFT-domain inner product over prepared baby rotations and prepared diagonals),
//! then ROT rotates that bucket and folds it into a DFT accumulator. The lazy
//! path keeps body add, DFT automorphism, and cross-giant accumulation in DFT;
//! only mask preparation uses scratch BIG/SMALL before key-switching. Incompatible
//! bases fall back to the regular normalized GLWE automorphism path.

use poulpy_hal::{
    api::{
        Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf,
        VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
        ZnxInfos,
    },
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain,
    default::{
        keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
        linear_transformation::{
            baby_steps::GLWEPreparedBabyStepHelper,
            inner_product::glwe_accumulate_prepared_baby_steps_dft,
            lazy::{
                glwe_dft_add_dft_assign, glwe_dft_copy_dft, glwe_idft_dft_into_big, glwe_lazy_giant_automorphism_from_dft,
                glwe_normalize_big_into,
            },
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
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
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
    let first_baby_rot = prepared.baby_step_rotation(first_gs.first_baby_step_index());
    let sizing_diagonal_operand = first_gs.diagonal(first_baby_rot);
    let baby_size = babies.baby_step(first_baby_rot).size();
    let diagonal_size = sizing_diagonal_operand.size();
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

    let scratch = scratch.borrow();

    if use_final_lazy_accumulator {
        // Lazy path: PROD, body add, giant automorphism, and cross-giant
        // accumulation all stay in DFT. The only IDFT is the final one before
        // the single BIG -> SMALL normalization.
        let (mut prod_dft, scratch_phase) = scratch.take_vec_znx_dft_scratch(module, cols, prod_size);
        let (mut lazy_acc_dft, mut scratch_phase) = scratch_phase.take_vec_znx_dft_scratch(module, cols, lazy_size);
        for col in 0..cols {
            module.vec_znx_dft_zero(&mut lazy_acc_dft, col);
        }

        let mut res_initialized = false;
        for gs in &prepared.giant_steps {
            {
                let mut prod_dft_backend = prod_dft.to_backend_mut();
                glwe_accumulate_prepared_baby_steps_dft(
                    module,
                    cnv_offset_hi,
                    &mut prod_dft_backend,
                    prepared,
                    gs,
                    babies,
                    &mut scratch_phase,
                );
            }

            if gs.rot == 0 {
                let prod_dft_ref = prod_dft.to_backend_ref();
                let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
                if res_initialized {
                    glwe_dft_add_dft_assign(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                } else {
                    glwe_dft_copy_dft(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                }
            } else {
                let key: &K = keys
                    .get_automorphism_key(gs.rot)
                    .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {}", gs.rot));
                {
                    let (mut rot_dft, mut scratch_rot) =
                        scratch_phase
                            .borrow()
                            .take_vec_znx_dft_scratch(module, cols, key_size_effective);
                    {
                        let mut rot_dft_backend = rot_dft.to_backend_mut();
                        let prod_dft_ref = prod_dft.to_backend_ref();
                        glwe_lazy_giant_automorphism_from_dft(
                            module,
                            &mut rot_dft_backend,
                            &prod_dft_ref,
                            prod_base2k.as_usize(),
                            key,
                            key_size,
                            &mut scratch_rot,
                        );
                    }

                    let rot_dft_ref = rot_dft.to_backend_ref();
                    let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
                    if res_initialized {
                        glwe_dft_add_dft_assign(module, &mut lazy_acc_dft_backend, &rot_dft_ref);
                    } else {
                        glwe_dft_copy_dft(module, &mut lazy_acc_dft_backend, &rot_dft_ref);
                    }
                }
            }
            res_initialized = true;
        }
        assert!(res_initialized, "linear transformation has no giant steps");

        let (mut lazy_acc_big, mut scratch_phase) = scratch_phase.take_vec_znx_big_scratch(module, cols, lazy_size);
        {
            let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
            let mut lazy_acc_big_backend = lazy_acc_big.to_backend_mut();
            glwe_idft_dft_into_big(module, &mut lazy_acc_big_backend, &mut lazy_acc_dft_backend);
        }

        let lazy_acc_ref = lazy_acc_big.to_backend_ref();
        glwe_normalize_big_into(
            module,
            res,
            &lazy_acc_ref,
            prod_base2k.as_usize(),
            cnv_offset_lo,
            &mut scratch_phase,
        );
        return;
    }

    // Fallback for incompatible bases: PROD is still computed in DFT, then each
    // column is IDFT'd through a one-column BIG scratch only where it is
    // normalized into the temporary SMALL ciphertext.
    let (mut prod_dft, scratch_phase) = scratch.take_vec_znx_dft_scratch(module, cols, prod_size);
    let (mut prod_col_big, mut scratch_phase) = scratch_phase.take_vec_znx_big_scratch(module, 1, prod_size);
    let mut fallback_acc: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(res);
    let mut res_initialized = false;

    for gs in &prepared.giant_steps {
        {
            let mut prod_dft_backend = prod_dft.to_backend_mut();
            glwe_accumulate_prepared_baby_steps_dft(
                module,
                cnv_offset_hi,
                &mut prod_dft_backend,
                prepared,
                gs,
                babies,
                &mut scratch_phase,
            );
            let mut acc_backend = <GLWE<BE::OwnedBuf> as GLWEToBackendMut<BE>>::to_backend_mut(&mut fallback_acc);
            for col in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut prod_col_big, 0, &mut prod_dft_backend, col);
                let prod_col_big_ref = prod_col_big.to_backend_ref();
                module.vec_znx_big_normalize(
                    &mut acc_backend.data,
                    res_base2k.as_usize(),
                    cnv_offset_lo,
                    col,
                    &prod_col_big_ref,
                    prod_base2k.as_usize(),
                    0,
                    &mut scratch_phase.borrow(),
                );
            }
        }

        if gs.rot != 0 {
            let key: &K = keys
                .get_automorphism_key(gs.rot)
                .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {}", gs.rot));
            module.glwe_automorphism_assign(&mut fallback_acc, key, key_size, &mut scratch_phase);
        }

        if res_initialized {
            module.glwe_add_assign(res, &fallback_acc);
        } else {
            module.glwe_copy(res, &fallback_acc);
            res_initialized = true;
        }
    }
}
