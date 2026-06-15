//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the scale-derived convolution parameters (`a_effective_k`,
//! `cnv_offset`) and the result `log_delta` / `log_budget`, delegates the actual
//! evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformations`](poulpy_core::GLWELinearTransformations), and stamps the
//! CKKS metadata onto the result. See `docs/lt_bsgs.md`.

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWELinearTransformations, LinearTransformationBabySteps, LinearTransformationGiantStep,
    LinearTransformationPrepared,
    default::{
        keyswitching::truncated_keyswitch_size,
        linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
    },
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        LWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, ScratchArena, VecZnxDftBackendMut, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{LinearTransformation, LinearTransformationOps},
    checked_log_budget_sub, checked_mul_pt_log_budget,
    layouts::{CKKSModuleAlloc, CKKSPlaintext},
};

/// Per-giant streamed PROD for CKKS plaintext diagonals.
///
/// The scheme-side half of [`DiagonalProd`]: where the resident path (core's
/// [`PreparedDiagonal`](poulpy_core::layouts::prepared::PreparedDiagonal)) fuses
/// already-prepared diagonals, the streamed path prepares each
/// [`CKKSPlaintext`] diagonal on the fly. Implementing it here (per concrete
/// plaintext type) is what lets the resident and streamed transforms share the
/// single `LinearTransformation<P>` container without overlapping impls.
impl<BE: Backend, D: Data> DiagonalProd<BE> for CKKSPlaintext<D>
where
    CKKSPlaintext<D>: GLWEToBackendRef<BE>,
{
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
        glwe_accumulate_streamed_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
    }
}

/// Output `(log_budget, log_delta, cnv_offset)` for `dst = M·src`, given the
/// matrix diagonals' scale exponent (`pt_log_scale`) and storage precision
/// (`pt_max_k`).
///
/// This is `get_mul_pt_params(res, a, pt)` specialized to a plaintext operand
/// described by just those two integers (its `log_budget` is dead in this math),
/// which avoids materializing a throwaway plaintext-proxy value at eval time.
/// The prepared path reads them off the cache (`pt_log_scale()` / `pt_max_k()`),
/// the streamed path off the first plaintext diagonal (`log_delta()` /
/// `max_k()`); both share this body.
fn lt_mul_params<R, A>(res: &R, a: &A, pt_log_scale: usize, pt_max_k: usize) -> Result<(usize, usize, usize)>
where
    R: LWEInfos,
    A: CKKSInfos,
{
    let res_log_budget = checked_mul_pt_log_budget("mul", a.log_budget(), 0, a.log_delta(), pt_log_scale)?;
    let res_log_delta = a.log_delta();
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = pt_max_k + res_offset;
    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

/// Truncated keyswitch output size for a linear-transformation giant rotation.
///
/// Errors introduced by the giant rotations sit under the diagonal scale (the
/// convolution happened first), minus whatever slack `res_offset` already
/// consumed to fit the result precision in `max_k`. `pt_log_scale` / `pt_max_k`
/// describe the diagonals (the cache for the prepared path, the first plaintext
/// for the streamed one).
fn truncated_lt_key_size<H, K, BE>(
    n: usize,
    dst_size: usize,
    src_size: usize,
    keys: &H,
    pt_log_scale: usize,
    pt_max_k: usize,
    cnv_offset: usize,
) -> usize
where
    BE: Backend,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let res_offset = cnv_offset.saturating_sub(pt_max_k);
    let allowance = pt_log_scale.saturating_sub(res_offset);
    truncated_keyswitch_size(n, dst_size, src_size + 1, &keys.automorphism_key_infos(), allowance)
}

impl<BE: Backend> LinearTransformationOps<BE> for Module<BE>
where
    Module<BE>: GLWELinearTransformations<BE> + GLWECopy<BE> + CKKSModuleAlloc<BE> + CyclotomicOrder,
{
    // ---------- tmp_bytes ----------

    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        self.glwe_prepare_linear_transformation_rhs_tmp_bytes(pt_infos)
    }

    fn ckks_prepare_linear_transformation_lhs_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_linear_transformation_lhs_tmp_bytes(ct, key)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy: it bounds the convolution
        // sizes from above, so the result is a safe upper bound.
        self.glwe_eval_linear_transformation_tmp_bytes(ct, ct, ct, key)
    }

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy (upper bound on diagonal shape).
        self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct, ct, ct, key)
    }

    // ---------- populate ----------

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        // Stash the plaintext scale exponent while filling the diagonals so eval
        // no longer needs `lt` for `cnv_offset` math.
        if let Some(first_pt) = lt.first_diagonal_plaintext() {
            prepared.set_log_scale(first_pt.log_delta());
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
    }

    fn ckks_prepare_linear_transformation_lhs<Src, H, K>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut has_nonzero = false;
        let cyclotomic_order = self.cyclotomic_order();
        for rotation in babies.baby_steps().filter(|&rotation| rotation != 0) {
            has_nonzero = true;
            if keys
                .get_automorphism_key(galois_element(rotation, cyclotomic_order))
                .is_none()
            {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }
        let key_size = if has_nonzero {
            // Baby-step rotation errors are amplified by the diagonal scale in
            // PROD, so no allowance: keep every limb that can reach the output.
            truncated_keyswitch_size(self.n(), src.size(), src.size(), &keys.automorphism_key_infos(), 0)
        } else {
            src.size()
        };
        self.glwe_prepare_linear_transformation_lhs(babies, src, src.effective_k(), keys, key_size, scratch);
        Ok(())
    }

    // ---------- eval (prepared) ----------

    fn ckks_eval_prepared_linear_transformation_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        prepared: &LinearTransformationPrepared<BE>,
        babies: &LinearTransformationBabySteps<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        check_required_keys(prepared, babies, keys, self.cyclotomic_order())?;

        let first = prepared
            .first_diagonal_plaintext()
            .expect("prepared linear transformation has no diagonals");
        let (res_log_budget, res_log_delta, cnv_offset) = lt_mul_params(dst, src, first.log_scale(), first.max_k().as_usize())?;
        let key_size = key_size_for_prepared(self.n(), prepared, dst, src, keys, cnv_offset);
        self.glwe_eval_linear_transformation_into(cnv_offset, dst, babies, prepared, keys, key_size, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_eval_prepared_linear_transformation_assign<Dst, H, K>(
        &self,
        dst: &mut Dst,
        prepared: &LinearTransformationPrepared<BE>,
        babies: &LinearTransformationBabySteps<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_prepared_linear_transformation_into(&mut tmp, dst, prepared, babies, keys, scratch)?;
        self.glwe_copy(dst, &tmp);
        dst.set_meta(tmp.meta());
        Ok(())
    }

    // ---------- one-shot ----------

    fn ckks_eval_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let first_plaintext = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;

        let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(self, &lt.index(), first_plaintext);
        self.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);

        let mut babies = LinearTransformationBabySteps::alloc(self, prepared.baby_steps(), src);
        self.ckks_prepare_linear_transformation_lhs(&mut babies, src, keys, scratch)?;

        self.ckks_eval_prepared_linear_transformation_into(dst, src, &prepared, &babies, keys, scratch)
    }

    fn ckks_eval_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_linear_transformation_into(&mut tmp, dst, lt, keys, scratch)?;
        self.glwe_copy(dst, &tmp);
        dst.set_meta(tmp.meta());
        Ok(())
    }

    // ---------- unprepared RHS, caller-supplied baby cache ----------

    fn ckks_eval_linear_transformation_unprepared_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let first_plaintext = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;

        // All non-zero giant rotations must have an automorphism key (keyed by
        // Galois element); the caller is responsible for `babies` covering the
        // baby rotations.
        let cyclotomic_order = self.cyclotomic_order();
        let has_nonzero_giant_rotation = lt.giant_steps.iter().any(|gs| gs.rot != 0 && !gs.diagonals.is_empty());
        for gs in &lt.giant_steps {
            if gs.rot != 0
                && !gs.diagonals.is_empty()
                && keys.get_automorphism_key(galois_element(gs.rot, cyclotomic_order)).is_none()
            {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation: gs.rot,
                }
                .into());
            }
        }

        let (res_log_budget, res_log_delta, cnv_offset) =
            lt_mul_params(dst, src, first_plaintext.log_delta(), first_plaintext.max_k().as_usize())?;
        let key_size = if has_nonzero_giant_rotation {
            truncated_lt_key_size(
                self.n(),
                dst.size(),
                src.size(),
                keys,
                first_plaintext.log_delta(),
                first_plaintext.max_k().as_usize(),
                cnv_offset,
            )
        } else {
            src.size()
        };
        self.glwe_eval_linear_transformation_unprepared_rhs_into(cnv_offset, dst, babies, lt, keys, key_size, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_eval_linear_transformation_unprepared_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_linear_transformation_unprepared_into(&mut tmp, dst, babies, lt, keys, scratch)?;
        self.glwe_copy(dst, &tmp);
        dst.set_meta(tmp.meta());
        Ok(())
    }

    // ---------- streamed (unprepared RHS, self-allocated baby cache) ----------

    fn ckks_eval_linear_transformation_streamed_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        // Only the (small) input baby cache is materialized; the matrix streams.
        let plan = lt.index();
        let mut babies = LinearTransformationBabySteps::alloc(self, &plan.baby_steps, src);
        self.ckks_prepare_linear_transformation_lhs(&mut babies, src, keys, scratch)?;
        self.ckks_eval_linear_transformation_unprepared_into(dst, src, &babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_streamed_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_linear_transformation_streamed_into(&mut tmp, dst, lt, keys, scratch)?;
        self.glwe_copy(dst, &tmp);
        dst.set_meta(tmp.meta());
        Ok(())
    }
}

/// Verifies that all automorphism keys required by `prepared` are present
/// (keyed by Galois element) and that `babies` covers every baby rotation
/// `prepared` needs.
fn check_required_keys<BE: Backend, H, K>(
    prepared: &LinearTransformationPrepared<BE>,
    babies: &LinearTransformationBabySteps<BE>,
    keys: &H,
    cyclotomic_order: i64,
) -> Result<()>
where
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    for rotation in prepared.baby_steps().iter().copied() {
        anyhow::ensure!(
            babies.contains_baby_step(rotation),
            "missing prepared baby-step rotation {rotation}"
        );
    }
    for rotation in prepared.giant_steps.iter().map(|gs| gs.rot).filter(|&r| r != 0) {
        let gal_el = galois_element(rotation, cyclotomic_order);
        if keys.get_automorphism_key(gal_el).is_none() {
            return Err(CKKSCompositionError::MissingAutomorphismKey {
                op: "linear_transformation",
                rotation,
            }
            .into());
        }
    }
    Ok(())
}

/// Resolves the `key_size` argument used by the core eval entry point. Falls
/// back to `src.size()` when no giant rotation is needed (identity-only
/// transforms with an empty key map). When giant rotations are needed, the
/// keyswitch output is truncated: errors introduced by the giant rotations sit
/// under the diagonal scale (the convolution happened first), minus whatever
/// slack `res_offset` already consumed to fit the result precision in `max_k`.
fn key_size_for_prepared<BE: Backend, Dst, Src, H, K>(
    n: usize,
    prepared: &LinearTransformationPrepared<BE>,
    dst: &Dst,
    src: &Src,
    keys: &H,
    cnv_offset: usize,
) -> usize
where
    Dst: LWEInfos,
    Src: CKKSCtBounds,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let has_nonzero_giant_rotation = prepared.giant_steps.iter().any(|gs| gs.rot != 0);
    if has_nonzero_giant_rotation {
        let first = prepared
            .first_diagonal_plaintext()
            .expect("prepared linear transformation has no diagonals");
        truncated_lt_key_size(
            n,
            dst.size(),
            src.size(),
            keys,
            first.log_scale(),
            first.max_k().as_usize(),
            cnv_offset,
        )
    } else {
        src.size()
    }
}
