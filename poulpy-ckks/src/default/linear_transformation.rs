//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the scale-derived convolution parameters (`a_effective_k`,
//! `cnv_offset`) and the result `log_delta` / `log_budget`, delegates the actual
//! evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformOps`](poulpy_core::GLWELinearTransformOps), and stamps the
//! CKKS metadata onto the result. See `docs/lt_bsgs.md`.

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWELinearTransformations, GLWEPreparedLinearTransformationLhs, GLWEPreparedLinearTransformationRhs,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        LWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{LinearTransformation, LinearTransformationOps, PreparedLinearTransformationRhs},
    checked_log_budget_sub, checked_mul_pt_log_budget,
    layouts::CKKSModuleAlloc,
};

/// Output `(log_budget, log_delta, cnv_offset)` for `dst = M·src` where `M` is a
/// prepared linear transformation.
///
/// This is `get_mul_pt_params(res, a, pt)` specialized to a plaintext operand
/// whose shape is stashed in the prepared cache: `pt.max_k() == pt_max_k()` and
/// `pt.log_delta() == pt_log_scale()` (the plaintext `log_budget` is dead in
/// this math). Reading those two integers directly avoids materializing a
/// throwaway plaintext-proxy value at eval time.
fn prepared_lt_mul_params<R, A, BE>(
    res: &R,
    a: &A,
    prepared: &PreparedLinearTransformationRhs<BE>,
) -> Result<(usize, usize, usize)>
where
    R: LWEInfos,
    A: CKKSInfos,
    BE: Backend,
{
    let res_log_budget = checked_mul_pt_log_budget("mul", a.log_budget(), 0, a.log_delta(), prepared.pt_log_scale())?;
    let res_log_delta = a.log_delta();
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = prepared.pt_max_k().as_usize() + res_offset;
    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

impl<BE: Backend> LinearTransformationOps<BE> for Module<BE>
where
    Module<BE>: GLWELinearTransformations<BE> + GLWECopy<BE> + CKKSModuleAlloc<BE>,
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

    // ---------- populate ----------

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut PreparedLinearTransformationRhs<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        // Stash the plaintext scale exponent while filling the diagonals so eval
        // no longer needs `lt` for `cnv_offset` math.
        if let Some(first_pt) = lt
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| &d.plaintext)
            .next()
        {
            prepared.set_pt_log_scale(first_pt.log_delta());
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
    }

    fn ckks_prepare_linear_transformation_lhs<Src, H, K>(
        &self,
        babies: &mut PreparedLinearTransformationLhs<BE>,
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
        for rotation in babies.baby_steps().filter(|&rotation| rotation != 0) {
            has_nonzero = true;
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }
        let key_size = if has_nonzero {
            keys.automorphism_key_infos().size()
        } else {
            src.size()
        };
        self.glwe_prepare_linear_transformation_lhs(babies, src, src.effective_k(), key_size, keys, scratch);
        Ok(())
    }

    // ---------- eval (prepared) ----------

    fn ckks_eval_prepared_linear_transformation_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        prepared: &PreparedLinearTransformationRhs<BE>,
        babies: &PreparedLinearTransformationLhs<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        check_required_rotations(prepared, babies, keys)?;

        let (res_log_budget, res_log_delta, cnv_offset) = prepared_lt_mul_params(dst, src, prepared)?;
        let key_size = key_size_for_prepared(prepared, src, keys);
        self.glwe_eval_linear_transformation_into(dst, babies, prepared, cnv_offset, key_size, keys, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_eval_prepared_linear_transformation_assign<Dst, H, K>(
        &self,
        dst: &mut Dst,
        prepared: &PreparedLinearTransformationRhs<BE>,
        babies: &PreparedLinearTransformationLhs<BE>,
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

    fn ckks_eval_many_prepared_linear_transformations_into<Dst, Src, H, K>(
        &self,
        dsts: &mut [Dst],
        src: &Src,
        prepared_transforms: &[PreparedLinearTransformationRhs<BE>],
        babies: &PreparedLinearTransformationLhs<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        anyhow::ensure!(
            dsts.len() == prepared_transforms.len(),
            "linear transformation output count ({}) does not match transform count ({})",
            dsts.len(),
            prepared_transforms.len()
        );
        if prepared_transforms.is_empty() {
            return Ok(());
        }

        // All required keys present, baby cache covers every needed rotation.
        let mut required_rotations: Vec<i64> = prepared_transforms.iter().flat_map(|p| p.required_rotations()).collect();
        required_rotations.sort_unstable();
        required_rotations.dedup();
        for rotation in required_rotations {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }
        for prepared in prepared_transforms {
            for rotation in prepared.baby_steps().iter().copied() {
                anyhow::ensure!(
                    babies.contains_baby_step(rotation),
                    "missing prepared baby-step rotation {rotation}"
                );
            }
        }

        // All transforms share one cnv_offset (same pt scale) by construction.
        let mut shared_cnv_offset: Option<usize> = None;
        let mut output_meta = Vec::with_capacity(prepared_transforms.len());
        for (dst, prepared) in dsts.iter().zip(prepared_transforms) {
            let (res_log_budget, res_log_delta, cnv_offset) = prepared_lt_mul_params(dst, src, prepared)?;
            if let Some(shared) = shared_cnv_offset {
                anyhow::ensure!(
                    shared == cnv_offset,
                    "linear transformation convolution offsets are incompatible across outputs"
                );
            } else {
                shared_cnv_offset = Some(cnv_offset);
            }
            output_meta.push((res_log_budget, res_log_delta));
        }

        let key_size = key_size_for_prepared(&prepared_transforms[0], src, keys);
        for (dst, prepared) in dsts.iter_mut().zip(prepared_transforms) {
            self.glwe_eval_linear_transformation_into(dst, babies, prepared, shared_cnv_offset.unwrap(), key_size, keys, scratch);
        }
        for (dst, (res_log_budget, res_log_delta)) in dsts.iter_mut().zip(output_meta) {
            dst.set_log_budget(res_log_budget);
            dst.set_log_delta(res_log_delta);
        }
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let first_plaintext = lt
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| &d.plaintext)
            .next()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;

        let mut prepared = GLWEPreparedLinearTransformationRhs::alloc_from_index(self, &lt.index(), first_plaintext);
        self.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);

        let mut babies = GLWEPreparedLinearTransformationLhs::alloc(self, prepared.baby_steps(), src);
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
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
}

/// Verifies that all keys required by `prepared` are present and that `babies`
/// covers every baby rotation `prepared` needs.
fn check_required_rotations<BE: Backend, H, K>(
    prepared: &PreparedLinearTransformationRhs<BE>,
    babies: &PreparedLinearTransformationLhs<BE>,
    keys: &H,
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
    for rotation in prepared.giant_steps().iter().map(|gs| gs.rot()).filter(|&r| r != 0) {
        if keys.get_automorphism_key(rotation).is_none() {
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
/// transforms with an empty key map).
fn key_size_for_prepared<BE: Backend, Src, H, K>(prepared: &PreparedLinearTransformationRhs<BE>, src: &Src, keys: &H) -> usize
where
    Src: CKKSCtBounds,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let has_nonzero_giant_rotation = prepared.giant_steps().iter().any(|gs| gs.rot() != 0);
    if has_nonzero_giant_rotation {
        keys.automorphism_key_infos().size()
    } else {
        src.size()
    }
}

// Bring `PreparedLinearTransformationLhs` into scope under its CKKS name for the impl.
use crate::api::PreparedLinearTransformationLhs;
