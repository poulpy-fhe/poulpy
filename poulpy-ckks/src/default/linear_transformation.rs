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
    default::linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        LWEInfos,
        prepared::{GLWEAutomorphismKeyPreparedToBackendRef, PreparedDiagonal},
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, ScratchArena, VecZnxDftBackendMut, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSCopyOps, LinearTransformation, LinearTransformationOps, LtDiagonalScale},
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

/// Streamed-diagonal scale: a [`CKKSPlaintext`] carries its scale as `log_delta`.
impl<D: Data> LtDiagonalScale for CKKSPlaintext<D> {
    fn lt_log_scale(&self) -> usize {
        self.log_delta()
    }
}

/// Resident-diagonal scale: a core [`PreparedDiagonal`] carries the (opaque to the
/// core engine) scale the CKKS prepare step stashed on it via `set_log_scale`.
impl<D: Data, BE: Backend> LtDiagonalScale for PreparedDiagonal<D, BE> {
    fn lt_log_scale(&self) -> usize {
        self.log_scale()
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
fn lt_mul_params<R, A>(res: &R, a: &A, pt_log_scale: usize, pt_k: usize) -> Result<(usize, usize, usize)>
where
    R: LWEInfos,
    A: CKKSInfos,
{
    let res_log_budget = checked_mul_pt_log_budget("mul", a.log_budget(), 0, a.log_delta(), pt_log_scale)?;
    let res_log_delta = a.log_delta();
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = pt_k + res_offset;
    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

impl<BE: Backend> LinearTransformationOps<BE> for Module<BE>
where
    Module<BE>: GLWELinearTransformations<BE> + GLWECopy<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE> + CyclotomicOrder,
{
    // ---------- tmp_bytes ----------

    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        self.glwe_prepare_linear_transformation_rhs_tmp_bytes(pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(ct, key)
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

    fn ckks_prepare_linear_transformation_baby_steps<Src, H, K>(
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
        // Canonical keyswitch output size: the operand plus `key.dsize()` guard
        // limbs (the keyswitch adds ~`dsize·base2k` bits of noise; see
        // `ckks_rotate_into_default`). Only read `dsize` when a baby rotation
        // actually needs a key — `automorphism_key_infos()` panics on an empty key
        // map, and with no rotation the value is unused anyway.
        let key_size = if has_nonzero {
            src.max_size() + keys.automorphism_key_infos().dsize().as_usize()
        } else {
            src.max_size()
        };
        self.glwe_prepare_linear_transformation_baby_steps(babies, src, src.effective_k(), keys, key_size, scratch);
        Ok(())
    }

    // ---------- eval (caller-supplied baby cache) ----------

    fn ckks_eval_linear_transformation_into<Dst, Src, P, H, K>(
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
        P: DiagonalProd<BE> + LtDiagonalScale,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        check_required_keys(lt, babies, keys, self.cyclotomic_order())?;

        let first = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;
        // The diagonal scale (`lt_log_scale`, the CKKS-layer accessor) and storage
        // precision (the scheme-agnostic `max_k`) are read uniformly off the first
        // diagonal, regardless of `P` (resident or streamed) — the only
        // representation-dependent step in this wrapper.
        let (pt_log_scale, pt_max_k) = (first.lt_log_scale(), first.max_k().as_usize());
        let (res_log_budget, res_log_delta, cnv_offset) = lt_mul_params(dst, src, pt_log_scale, pt_max_k)?;
        let key_size = lt_key_size(lt, dst, keys);
        self.glwe_eval_linear_transformation_into(cnv_offset, dst, babies, lt, keys, key_size, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.compact_in_place();
        Ok(())
    }

    fn ckks_eval_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalScale,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_linear_transformation_into(&mut tmp, dst, babies, lt, keys, scratch)?;
        // `ckks_copy` moves both the limbs and the CKKS metadata the eval consumed
        // into `dst` (a plain `glwe_copy` would leave the budget/scale stale).
        self.ckks_copy(dst, &tmp, scratch)?;
        Ok(())
    }

    // ---------- eval (self-allocated baby cache) ----------

    fn ckks_eval_linear_transformation_self_into<Dst, Src, P, H, K>(
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
        P: DiagonalProd<BE> + LtDiagonalScale,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        // Only the (small) input baby cache is materialized here; with a plaintext
        // `lt` the matrix RHS itself is streamed inside the eval.
        let mut babies = LinearTransformationBabySteps::alloc(self, lt.baby_steps(), src);
        self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
        self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_self_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalScale,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(dst.meta());
        self.ckks_eval_linear_transformation_self_into(&mut tmp, dst, lt, keys, scratch)?;
        // `ckks_copy` moves both the limbs and the CKKS metadata the eval consumed
        // into `dst` (a plain `glwe_copy` would leave the budget/scale stale).
        self.ckks_copy(dst, &tmp, scratch)?;
        Ok(())
    }
}

/// Verifies that all automorphism keys required by `lt` are present (keyed by
/// Galois element) and that `babies` covers every baby rotation `lt` needs.
fn check_required_keys<BE: Backend, P, H, K>(
    lt: &LinearTransformation<P>,
    babies: &LinearTransformationBabySteps<BE>,
    keys: &H,
    cyclotomic_order: i64,
) -> Result<()>
where
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    for rotation in lt.baby_steps().iter().copied() {
        anyhow::ensure!(
            babies.contains_baby_step(rotation),
            "missing prepared baby-step rotation {rotation}"
        );
    }
    for rotation in lt
        .giant_steps
        .iter()
        .filter(|gs| !gs.diagonals.is_empty())
        .map(|gs| gs.rot)
        .filter(|&r| r != 0)
    {
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

/// Resolves the `key_size` (giant-rotation keyswitch output size) for the core
/// eval entry point.
///
/// Canonical sizing, consistent with the rest of the crate
/// (`ckks_rotate_into_default` etc.): the result operand plus `key.dsize()` guard
/// limbs, since a keyswitch adds ~`dsize·base2k` bits of noise. The core eval caps
/// this at the key's own size, so over-asking is safe. Falls back to `dst.size()`
/// when no giant rotation is needed (the eval skips ROT for `rot == 0`, so
/// `key_size` is unused — and `automorphism_key_infos()` would panic on the
/// possibly-empty key map of such an identity-only transform).
fn lt_key_size<BE: Backend, P, Dst, H, K>(lt: &LinearTransformation<P>, dst: &Dst, keys: &H) -> usize
where
    Dst: LWEInfos,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let has_nonzero_giant_rotation = lt.giant_steps.iter().any(|gs| gs.rot != 0 && !gs.diagonals.is_empty());
    if has_nonzero_giant_rotation {
        dst.max_size() + keys.automorphism_key_infos().dsize().as_usize()
    } else {
        dst.max_size()
    }
}
