//! Prepared giant-step evaluation.
//!
//! Main loop from docs/linear_transformation.md: each giant bucket first runs PROD (a
//! DFT-domain inner product over prepared baby rotations and prepared diagonals),
//! then ROT rotates that bucket and folds it into a DFT accumulator. The lazy
//! path keeps body add, DFT automorphism, and cross-giant accumulation in DFT;
//! only mask preparation uses scratch BIG/SMALL before key-switching. Incompatible
//! bases fall back to the regular normalized GLWE automorphism path.

use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf,
        VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation, LinearTransformationGiantStep,
    default::{
        keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
        linear_transformation::{
            inner_product::{glwe_accumulate_prepared_baby_steps_dft, glwe_accumulate_unprepared_baby_steps_dft},
            lazy::{
                glwe_dft_add_dft_assign, glwe_dft_copy_dft, glwe_idft_dft_into_big, glwe_lazy_giant_automorphism_from_dft,
                glwe_normalize_big_into,
            },
        },
        operations::cnv_offset_to_limb_offset,
    },
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, GetGaloisElement, LWEInfos, ModuleCoreAlloc, WithEffectiveDsize,
        prepared::{GGLWEPreparedToBackendRef, PreparedDiagonal},
    },
};

use super::LinearTransformationBabySteps;

/// Per-giant PROD, provided by the diagonal representation itself.
///
/// The shared evaluator `glwe_eval_giant_steps` is generic over the diagonal
/// type `P` stored in the [`LinearTransformation`]; the only per-flavor step is
/// how one giant step's `Σ_k ũ_{j,k} ⊙ rot(v,k)` is computed. Each concrete
/// diagonal type implements this once:
///
/// - [`PreparedDiagonal`] (resident): diagonals are already in convolution
///   domain, so the whole giant step is one fused accumulation
///   (`glwe_accumulate_prepared_baby_steps_dft`).
/// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
///   through one reused scratch slot
///   (`glwe_accumulate_unprepared_baby_steps_dft`); implemented by the scheme
///   layer for its plaintext type.
///
/// Dispatching per concrete type — rather than via a blanket keyed on
/// [`GLWEToBackendRef`] — is what keeps the impls coherent: the backend type
/// parameter prevents the compiler from ruling out a downstream
/// `GLWEToBackendRef` impl for [`PreparedDiagonal`], so a blanket would clash
/// with the resident impl.
pub trait DiagonalProd<BE: Backend>: LWEInfos + Sized {
    /// Runs the PROD inner product of one giant step into `prod_dft`.
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN;
}

impl<BE: Backend> DiagonalProd<BE> for PreparedDiagonal<BE::OwnedBuf, BE> {
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        glwe_accumulate_prepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
    }
}

/// Reference streamed PROD, exposed for the scheme layer to wire into its
/// [`DiagonalProd`] impl for its own plaintext diagonal type.
pub fn glwe_accumulate_streamed_baby_steps_dft<BE, M, P>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationBabySteps<BE>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    P: GLWEToBackendRef<BE> + crate::layouts::IntPolyInfos + GLWEInfos,
{
    glwe_accumulate_unprepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
}

/// The shared Phase B/C driver: runs the BSGS giant-step loop and finalizes.
///
/// Generic over the diagonal type `P: DiagonalProd`, so the *same* loop drives
/// both the resident transform (`P = PreparedDiagonal`) and the streamed
/// unprepared transform (`P` a plaintext); only the per-giant PROD block
/// ([`DiagonalProd::accumulate_giant_prod`]) differs. Implements the giant-step
/// products, rotations, and final normalization of docs/linear_transformation.md.
///
/// For each giant step `j` it computes `PROD = Σ_k ũ_{j,k} ⊙ rot(v,k)` in DFT,
/// then rotates by `n1·j` and folds the result into a single accumulator. One of
/// two strategies is chosen up front from the
/// `base2k` agreement of `res`, the PROD output, and the keys:
///
/// - **Lazy DFT path** (hot path; bases agree, or no giant rotation): body add,
///   giant automorphism, and cross-giant accumulation all stay in `VecZnxDft`.
///   The only IDFT and the only normalize happen once, at the end. `j == 0`
///   skips ROT entirely.
/// - **Fallback path** (base mismatch): PROD is still in DFT, but each giant
///   contribution is normalized to SMALL and rotated with the public normalized
///   `glwe_automorphism`. Correct, but gives up the lazy savings.
///
/// Writes the encryption of `M·v` into `res`.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_eval_giant_steps<BE, M, R, P, H, K>(
    module: &M,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
        + GaloisElement
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
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
    P: DiagonalProd<BE>,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
{
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();

    // PROD writes its result in the diagonals' base2k.
    let first_diagonal = rhs
        .first_diagonal_plaintext()
        .expect("linear transformation has no diagonals");
    let prod_base2k = first_diagonal.base2k();
    let baby_size = lhs.size();
    let diagonal_size = first_diagonal.size();
    let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, prod_base2k.as_usize());
    let prod_size = baby_size + diagonal_size - cnv_offset_hi;

    let num_giant_steps = rhs.giant_steps.len();
    let nonzero_giant_rotations = rhs.giant_steps.iter().filter(|gs| gs.rot != 0).count();
    let has_nonzero_giant_rotation = nonzero_giant_rotations != 0;
    // Keys may differ per giant rotation, so the sizing is the widest of the
    // rotations actually used. An identity-only transform consults none.
    let (use_lazy_giant_rotation, key_output_size) = if has_nonzero_giant_rotation {
        let mut key_base2k: Option<crate::layouts::Base2K> = None;
        let mut output_size: usize = 0;
        for gs in rhs.giant_steps.iter().filter(|gs| gs.rot != 0) {
            let (layout, effective_dsize) = keys
                .get_automorphism_key_layout_for(module.galois_element(gs.rot), lhs.k())
                .unwrap_or_else(|e| panic!("giant-step rotation {}: {e}", gs.rot));
            key_base2k = Some(layout.base2k());
            output_size = output_size.max(
                crate::default::keyswitching::gglwe_product_accumulation_output_size_with_tail::<BE, _, _, _>(
                    res,
                    res,
                    &layout.with_dsize(effective_dsize),
                    nonzero_giant_rotations,
                    prod_size.saturating_sub(res.size()),
                ),
            );
        }
        let key_base2k = key_base2k.expect("at least one nonzero giant rotation");
        let bases_match = res_base2k == key_base2k && prod_base2k == key_base2k;
        (bases_match, output_size)
    } else {
        // No giant rotation: BIG-flow accumulator is always valid (no key required).
        (true, res.size())
    };
    let use_final_lazy_accumulator = !has_nonzero_giant_rotation || use_lazy_giant_rotation;
    let lazy_size = if use_lazy_giant_rotation {
        key_output_size.max(prod_size)
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
        for g in 0..num_giant_steps {
            {
                let mut prod_dft_backend = prod_dft.to_backend_mut();
                P::accumulate_giant_prod(
                    module,
                    cnv_offset_hi,
                    &mut prod_dft_backend,
                    lhs,
                    &rhs.giant_steps[g],
                    &mut scratch_phase,
                );
            }

            let rot = rhs.giant_steps[g].rot;
            if rot == 0 {
                let prod_dft_ref = prod_dft.to_backend_ref();
                let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
                if res_initialized {
                    glwe_dft_add_dft_assign(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                } else {
                    glwe_dft_copy_dft(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                }
            } else {
                let (key, effective_dsize) = keys
                    .get_automorphism_key_for(module.galois_element(rot), lhs.k())
                    .unwrap_or_else(|e| panic!("giant-step rotation {rot}: {e}"));
                let key = &key.with_dsize(effective_dsize);
                {
                    let (mut rot_dft, mut scratch_rot) =
                        scratch_phase.borrow().take_vec_znx_dft_scratch(module, cols, key_output_size);
                    {
                        let mut rot_dft_backend = rot_dft.to_backend_mut();
                        let prod_dft_ref = prod_dft.to_backend_ref();
                        glwe_lazy_giant_automorphism_from_dft(
                            module,
                            &mut rot_dft_backend,
                            &prod_dft_ref,
                            prod_base2k.as_usize(),
                            key,
                            key_output_size,
                            nonzero_giant_rotations,
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
    let mut fallback_acc: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(res);
    let mut res_initialized = false;

    for g in 0..num_giant_steps {
        {
            let mut prod_dft_backend = prod_dft.to_backend_mut();
            P::accumulate_giant_prod(
                module,
                cnv_offset_hi,
                &mut prod_dft_backend,
                lhs,
                &rhs.giant_steps[g],
                &mut scratch_phase,
            );
            let mut acc_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut fallback_acc);
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

        let rot = rhs.giant_steps[g].rot;
        if rot != 0 {
            let (key, effective_dsize) = keys
                .get_automorphism_key_for(module.galois_element(rot), lhs.k())
                .unwrap_or_else(|e| panic!("giant-step rotation {rot}: {e}"));
            module.glwe_automorphism_assign(&mut fallback_acc, &key.with_dsize(effective_dsize), &mut scratch_phase);
        }

        if res_initialized {
            module.glwe_add_assign(res, &fallback_acc);
        } else {
            module.glwe_copy(res, &fallback_acc);
            res_initialized = true;
        }
    }
}
