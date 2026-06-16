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
        GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        ModuleCoreAlloc,
        prepared::{GGLWEPreparedToBackendRef, PreparedDiagonal},
    },
};

use super::LinearTransformationBabySteps;

/// Per-giant PROD, provided by the diagonal representation itself.
///
/// The shared evaluator [`glwe_eval_giant_steps`] is generic over the diagonal
/// type `P` stored in the [`LinearTransformation`]; the only per-flavor step is
/// how one giant step's `Σ_k ũ_{j,k} ⊙ rot(v,k)` is computed. Each concrete
/// diagonal type implements this once:
///
/// - [`PreparedDiagonal`] (resident): diagonals are already in convolution
///   domain, so the whole giant step is one fused accumulation
///   ([`glwe_accumulate_prepared_baby_steps_dft`]).
/// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
///   through one reused scratch slot
///   ([`glwe_accumulate_unprepared_baby_steps_dft`]); implemented by the scheme
///   layer for its plaintext type.
///
/// Dispatching per concrete type — rather than via a blanket keyed on
/// [`GLWEToBackendRef`] — is what keeps the impls coherent: the backend type
/// parameter prevents the compiler from ruling out a downstream
/// `GLWEToBackendRef` impl for [`PreparedDiagonal`], so a blanket would clash
/// with the resident impl.
pub trait DiagonalProd<BE: Backend>: LWEInfos + Sized {
    /// The encoding scale (`log2` of the scaling factor) of this diagonal's
    /// plaintext. The CKKS scale/key-size bookkeeping reads it (together with
    /// [`LWEInfos::max_k`]) off the transform's first diagonal, uniformly across
    /// the resident ([`PreparedDiagonal::log_scale`]) and streamed (the plaintext
    /// `log_delta`) representations.
    fn diag_log_scale(&self) -> usize;

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
    fn diag_log_scale(&self) -> usize {
        self.log_scale()
    }

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
    P: GLWEToBackendRef<BE> + GLWEInfos,
{
    glwe_accumulate_unprepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
}

/// The shared Phase B/C driver: runs the BSGS giant-step loop and finalizes.
///
/// Generic over the diagonal type `P: DiagonalProd`, so the *same* loop drives
/// both the resident transform (`P = PreparedDiagonal`) and the streamed
/// unprepared transform (`P` a plaintext); only the per-giant PROD block
/// ([`DiagonalProd::accumulate_giant_prod`]) differs. Implements docs/lt_bsgs.md
/// §6.3-§6.4 and the implementation walkthrough in docs/lt_bsgs_impl.md §4.
///
/// For each giant step `j` it computes `PROD = Σ_k ũ_{j,k} ⊙ rot(v,k)` in DFT
/// (§4.2), then rotates by `n1·j` and folds the result into a single
/// accumulator (§4.3-§4.4). One of two strategies is chosen up front from the
/// `base2k` agreement of `res`, the PROD output, and the keys:
///
/// - **Lazy DFT path** (hot path; bases agree, or no giant rotation): body add,
///   giant automorphism, and cross-giant accumulation all stay in `VecZnxDft`.
///   The only IDFT and the only normalize happen once, at the end (savings
///   #4-#6, #9). `j == 0` skips ROT entirely.
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
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
        + GaloisElement
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
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
    H: GLWEAutomorphismKeyHelper<K, BE>,
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
    let has_nonzero_giant_rotation = rhs.giant_steps.iter().any(|gs| gs.rot != 0);
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
                let key: &K = keys
                    .get_automorphism_key(module.galois_element(rot))
                    .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {rot}"));
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

        let rot = rhs.giant_steps[g].rot;
        if rot != 0 {
            let key: &K = keys
                .get_automorphism_key(module.galois_element(rot))
                .unwrap_or_else(|| panic!("missing automorphism key for giant-step rotation {rot}"));
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
