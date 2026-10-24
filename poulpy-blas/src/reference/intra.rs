use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformationBabySteps, LinearTransformationPrepared},
};
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::lt_tmp_bytes;
use crate::layouts::{ciphertext::layout_of, validate::assert_maps_scratch};

/// Reference of the map hook of [`crate::oep::BlasMatIntraImpl`], under which the transpose and the shifts are
/// provided, and through which the reference cold preparations of products run, and the hot sides of the reference
/// products, and the first map of the reference JKLS products by public matrices. It runs on the inputs of an operation
/// its validator accepted ([`crate::layouts::validate_blas_transpose`], [`crate::layouts::validate_blas_rotate_rows`],
/// [`crate::layouts::validate_blas_rotate_cols`], [`crate::layouts::validate_blas_matmul_prepare_left`],
/// [`crate::layouts::validate_blas_matmul_prepare_right`], [`crate::layouts::validate_blas_matmul_prepared`],
/// [`crate::layouts::validate_blas_matmul`], [`crate::layouts::validate_blas_matmul_pt_right`],
/// [`crate::layouts::validate_blas_matmul_pt_left`]): it checks nothing but its scratch, and returns poulpy-ckks's
/// budget errors before any write. It runs source after source on one scratch and one baby-step cache, and the first
/// error stops it.
pub trait BlasMatIntraReference<BE: Backend> {
    /// Filling a baby-step cache from a source of layout `src`, and the evaluation of a map from it into a block of
    /// layout `dst`.
    fn blas_map_blocks_tmp_bytes_reference<D, S, K>(
        &self,
        dst: &D,
        src: &S,
        _sources: usize,
        _lts: &[&LinearTransformationPrepared<BE>],
        atk: &K,
    ) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, dst, src, atk, false)
    }

    /// `dst[t * lts.len() + m] = lts[m](src[t])`, stamped with the label of `src[t]`. Per source, one baby-step cache
    /// covers the baby steps of every map, then each map is evaluated from it with `ckks_eval_linear_transformation_into`.
    ///
    /// # Panics
    ///
    /// Panics unless `lts` is non-empty and `dst` holds `src.len() * lts.len()` blocks, or if `scratch` holds less than
    /// the query reports for the keys it looks up.
    fn blas_map_blocks_reference<D, S, K>(
        &self,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<BE>],
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized,
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_map_blocks";
        assert!(
            !lts.is_empty() && dst.len() == src.len() * lts.len(),
            "{op}: {} blocks for {} sources and {} maps",
            dst.len(),
            src.len(),
            lts.len()
        );
        let (Some(&first), Some(first_dst)) = (src.first(), dst.first()) else {
            return Ok(());
        };
        // At the blocks' widths, as the callers' queries read them, not their allocations.
        let (dst_layout, src_layout) = (layout_of(first_dst), layout_of(first));
        let bytes = |key: &GGLWELayout| self.blas_map_blocks_tmp_bytes_reference(&dst_layout, &src_layout, src.len(), lts, key);
        assert_maps_scratch(op, self, lts, atk, first.k(), first_dst.k(), first.rank(), bytes, scratch);
        let rotations: Vec<i64> = lts.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, first);
        for (&src, dst) in src.iter().zip(dst.chunks_mut(lts.len())) {
            self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, atk, scratch)?;
            for (dst, lt) in dst.iter_mut().zip(lts) {
                self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, atk, scratch)?;
                dst.set_log_sparsity(src.log_sparsity());
            }
        }
        Ok(())
    }
}

impl<BE: Backend> BlasMatIntraReference<BE> for Module<BE> {}
