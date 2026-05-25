//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the scale-derived convolution parameters (`a_effective_k`,
//! `cnv_offset`) and the result `log_delta` / `log_budget`, delegates the actual
//! evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformOps`](poulpy_core::GLWELinearTransformOps), and stamps the
//! CKKS metadata onto the result. See `docs/lt_bsgs.md`.

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWELinearTransformOps, GLWEPrepareLinearTransformOps, GLWEPreparedBabyRotations,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement, LWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{LinearTransformation, LinearTransformationOps, PreparedLinearTransformation},
    default::mul::get_mul_pt_params,
    layouts::CKKSModuleAlloc,
};

fn first_diagonal_plaintext<P>(lt: &LinearTransformation<P>) -> Result<&P> {
    lt.giant_steps
        .iter()
        .flat_map(|gs| gs.diagonals.iter())
        .map(|d| &d.plaintext)
        .next()
        .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))
}

impl<BE: Backend> LinearTransformationOps<BE> for Module<BE>
where
    Module<BE>: GLWELinearTransformOps<BE> + GLWEPrepareLinearTransformOps<BE> + GLWECopy<BE> + CKKSModuleAlloc<BE>,
{
    fn ckks_prepare_linear_transformation_tmp_bytes<P>(&self, lt: &LinearTransformation<P>) -> usize
    where
        P: CKKSCtBounds,
    {
        self.glwe_prepare_linear_transform_tmp_bytes(lt)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy: it bounds the convolution
        // sizes from above, so the result is a safe upper bound.
        self.glwe_prepared_linear_transform_tmp_bytes(ct, ct, ct, key)
    }

    fn ckks_prepare_baby_rotations_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_baby_rotations_tmp_bytes(ct, key)
    }

    fn ckks_prepare_linear_transformation<P>(
        &self,
        lt: &LinearTransformation<P>,
        prepared: &mut PreparedLinearTransformation<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.glwe_prepare_linear_transform(lt, prepared, scratch)
    }

    fn ckks_prepare_baby_rotations<Src, H, K>(
        &self,
        baby_steps: &[i64],
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<GLWEPreparedBabyRotations<BE>>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        for rotation in baby_steps.iter().copied().filter(|&rotation| rotation != 0) {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }

        let key_size = if baby_steps.iter().all(|&rotation| rotation == 0) {
            src.size()
        } else {
            keys.automorphism_key_infos().size()
        };
        Ok(self.glwe_prepare_baby_rotations(baby_steps, src, src.effective_k(), key_size, keys, scratch))
    }

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
        let first_plaintext = first_diagonal_plaintext(lt)?;

        // Every required automorphism key must be present.
        let required_rotations = lt.required_rotations();
        for rotation in required_rotations.iter().copied() {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }

        // All diagonals share one encoded scale, so the convolution alignment and
        // result metadata are uniform; derive them from the first diagonal.
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, src, first_plaintext)?;
        let a_effective_k: usize = src.effective_k();
        let key_size: usize = if required_rotations.is_empty() {
            src.size()
        } else {
            keys.automorphism_key_infos().size()
        };

        let mut prepared = PreparedLinearTransformation::default();
        self.glwe_prepare_linear_transform(lt, &mut prepared, scratch);
        let babies = self.glwe_prepare_baby_rotations(&prepared.baby_steps, src, a_effective_k, key_size, keys, scratch);
        self.glwe_prepared_linear_transform(dst, lt, &prepared, &babies, cnv_offset, key_size, keys, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_eval_prepared_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
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
        let first_plaintext = first_diagonal_plaintext(lt)?;

        let required_rotations = prepared.required_rotations();
        for rotation in required_rotations.iter().copied() {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, src, first_plaintext)?;
        let a_effective_k: usize = src.effective_k();
        let key_size: usize = if required_rotations.is_empty() {
            src.size()
        } else {
            keys.automorphism_key_infos().size()
        };

        let babies = self.glwe_prepare_baby_rotations(&prepared.baby_steps, src, a_effective_k, key_size, keys, scratch);
        self.glwe_prepared_linear_transform(dst, lt, prepared, &babies, cnv_offset, key_size, keys, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_eval_prepared_linear_transformation_with_babies_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
        babies: &GLWEPreparedBabyRotations<BE>,
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
        let first_plaintext = first_diagonal_plaintext(lt)?;

        for rotation in prepared.baby_steps.iter().copied() {
            anyhow::ensure!(
                babies.contains_baby_step(rotation),
                "missing prepared baby-step rotation {rotation}"
            );
        }

        for rotation in prepared.giant_steps.iter().map(|gs| gs.rot).filter(|&rotation| rotation != 0) {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, src, first_plaintext)?;
        let has_nonzero_giant_rotation = prepared.giant_steps.iter().any(|gs| gs.rot != 0);
        let key_size = if has_nonzero_giant_rotation {
            keys.automorphism_key_infos().size()
        } else {
            src.size()
        };

        self.glwe_prepared_linear_transform(dst, lt, prepared, babies, cnv_offset, key_size, keys, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
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

    fn ckks_eval_prepared_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
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
        self.ckks_eval_prepared_linear_transformation_into(&mut tmp, dst, lt, prepared, keys, scratch)?;
        self.glwe_copy(dst, &tmp);
        dst.set_meta(tmp.meta());
        Ok(())
    }

    fn ckks_eval_many_prepared_linear_transformations_into<Dst, Src, P, H, K>(
        &self,
        dsts: &mut [Dst],
        src: &Src,
        transforms: &[LinearTransformation<P>],
        prepared_transforms: &[PreparedLinearTransformation<BE>],
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
        anyhow::ensure!(
            dsts.len() == transforms.len(),
            "linear transformation output count ({}) does not match transform count ({})",
            dsts.len(),
            transforms.len()
        );
        anyhow::ensure!(
            transforms.len() == prepared_transforms.len(),
            "linear transformation count ({}) does not match prepared transform count ({})",
            transforms.len(),
            prepared_transforms.len()
        );

        if transforms.is_empty() {
            return Ok(());
        }

        let mut required_rotations = transforms
            .iter()
            .zip(prepared_transforms)
            .flat_map(|(_, prepared)| prepared.required_rotations())
            .collect::<Vec<_>>();
        required_rotations.sort_unstable();
        required_rotations.dedup();
        for rotation in required_rotations.iter().copied() {
            if keys.get_automorphism_key(rotation).is_none() {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                }
                .into());
            }
        }

        let mut output_meta = Vec::with_capacity(transforms.len());
        let mut shared_cnv_offset = None;
        for (dst, transform) in dsts.iter().zip(transforms) {
            let first_plaintext = first_diagonal_plaintext(transform)?;
            let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, src, first_plaintext)?;
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

        let key_size: usize = if required_rotations.is_empty() {
            src.size()
        } else {
            keys.automorphism_key_infos().size()
        };
        let mut baby_steps = transforms
            .iter()
            .zip(prepared_transforms)
            .flat_map(|(_, prepared)| prepared.baby_steps.iter().copied())
            .collect::<Vec<_>>();
        baby_steps.sort_unstable();
        baby_steps.dedup();
        let babies = self.glwe_prepare_baby_rotations(&baby_steps, src, src.effective_k(), key_size, keys, scratch);
        for ((dst, transform), prepared) in dsts.iter_mut().zip(transforms).zip(prepared_transforms) {
            self.glwe_prepared_linear_transform(
                dst,
                transform,
                prepared,
                &babies,
                shared_cnv_offset.unwrap(),
                key_size,
                keys,
                scratch,
            );
        }

        for (dst, (res_log_budget, res_log_delta)) in dsts.iter_mut().zip(output_meta) {
            dst.set_log_budget(res_log_budget);
            dst.set_log_delta(res_log_delta);
        }
        Ok(())
    }

    fn ckks_eval_sequential_prepared_linear_transformations_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        transforms: &[LinearTransformation<P>],
        prepared_transforms: &[PreparedLinearTransformation<BE>],
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
        anyhow::ensure!(
            transforms.len() == prepared_transforms.len(),
            "linear transformation count ({}) does not match prepared transform count ({})",
            transforms.len(),
            prepared_transforms.len()
        );

        if transforms.is_empty() {
            self.glwe_copy(dst, src);
            dst.set_meta(src.meta());
            return Ok(());
        }

        let mut tmp = self.ckks_ciphertext_alloc_from_infos(dst);
        tmp.set_meta(src.meta());

        self.ckks_eval_prepared_linear_transformation_into(dst, src, &transforms[0], &prepared_transforms[0], keys, scratch)?;
        for (transform, prepared) in transforms[1..].iter().zip(&prepared_transforms[1..]) {
            tmp.set_meta(dst.meta());
            self.ckks_eval_prepared_linear_transformation_into(&mut tmp, dst, transform, prepared, keys, scratch)?;
            self.glwe_copy(dst, &tmp);
            dst.set_meta(tmp.meta());
        }

        Ok(())
    }
}
