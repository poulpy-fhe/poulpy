use poulpy_ckks::{
    CKKSCtBounds, CKKSModuleInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSDotProductOps, CKKSLeftOperandPreparedFactory, CKKSMulOps, LinearTransformationPrepared},
    layouts::{CKKSCiphertextAlloc, CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
};
use poulpy_hal::{
    api::{ScratchArenaTakeVecZnxDft, VecZnxDftBytesOf},
    layouts::{Backend, Module, PrepareHint, ScratchArena, VecZnxDftToBackendMut},
};

use crate::{
    layouts::{
        ciphertext::layout_of,
        encoding::mapped_block,
        validate::{KeyLayouts, assert_scratch, ensure_lt_keys, ensure_tensor_key},
    },
    oep::BlasMatIntraImpl,
};

/// Reference of the JKLS kernel of [`crate::oep::BlasJklsImpl`], which
/// [`impl_blas_jkls_reference_full!`](crate::impl_blas_jkls_reference_full) forwards to and the reference products of
/// [`crate::oep::BlasMatMulImpl`] call once per block of `res`. It runs on inputs the prepared product's validator,
/// [`crate::layouts::validate_blas_matmul_prepared`], accepted, and checks nothing but its scratch.
pub trait BlasJklsReference<BE: Backend> {
    /// poulpy-ckks's `ckks_dot_product_prepared` query for `terms` pairs prepared from terms of these layouts.
    fn blas_jkls_block_tmp_bytes_reference<R, L, Rh, T>(&self, terms: usize, res: &R, lhs: &L, rhs: &Rh, tsk: &T) -> usize
    where
        Self: CKKSDotProductOps<BE>,
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_dot_product_prepared_tmp_bytes(terms, res, lhs, rhs, tsk)
    }

    /// The JKLS kernel: `res = sum_t lhs[t] rhs[t]` by poulpy-ckks's `ckks_dot_product_prepared`, which is
    /// `ckks_dot_product_ct` of the terms the pairs were prepared from, bit for bit, with no operand transformed. On
    /// pairs prepared from terms that share one layout per side, it relinearizes once, with the tensor key at
    /// `max(lhs.k, rhs.k)`. On rank 1, with one `base2k` across `res`, the terms and the key, and an offset of at least
    /// `base2k` in poulpy-ckks's ct x ct rule, that is poulpy-core's `glwe_mul_prepared_relinearize_sum`: the pairs'
    /// products are added in the transform domain, in slice order, before `res` is stamped. Otherwise it rounds each
    /// `glwe_tensor_apply_prepared` at that width, sums them exactly, normalizes the sum once, right before the
    /// relinearization reads it, and stamps `res` before relinearizing.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds less than the query reports for the tensor key it looks up.
    fn blas_jkls_block_reference<R, T>(
        &self,
        res: &mut R,
        lhs: &[&CKKSLeftOperandPrepared<BE>],
        rhs: &[&CKKSRightOperandPrepared<BE>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDotProductOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<BE> + Sync,
    {
        if let (Some(&l), Some(&r)) = (lhs.first(), rhs.first()) {
            let op = "blas_jkls_block";
            let used = KeyLayouts::new(tsk);
            let _ = ensure_tensor_key(op, &used, l.k().max(r.k()), l.rank());
            let bytes = used.max_bytes(|tsk| self.blas_jkls_block_tmp_bytes_reference(lhs.len(), &*res, l, r, tsk));
            assert_scratch(op, bytes, scratch);
        }
        self.ckks_dot_product_prepared(res, lhs, rhs, tsk, scratch)
    }

    /// The scratch of [`Self::blas_jkls_pt_unit_reference`]: one accumulator per block of `res`, then the largest
    /// step: the map hook through `first` from one source into a block of the layout `first` leaves and through `masks`
    /// from that block into blocks of the terms' layout, the copy of that block into a term, a term's preparation, and
    /// poulpy-ckks's DFT forms of the summed product.
    ///
    /// # Panics
    ///
    /// Panics if the accumulators' bytes overflow `usize`.
    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit_tmp_bytes_reference<R, S, K>(
        &self,
        outputs: usize,
        res: &R,
        src: &S,
        first: &LinearTransformationPrepared<BE>,
        masks: &[&LinearTransformationPrepared<BE>],
        plaintext: &CKKSRightOperandPrepared<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSLeftOperandPreparedFactory<BE> + CKKSDotProductOps<BE> + VecZnxDftBytesOf,
        R: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos;

    /// One unit of a product by public matrices under `JklsStored` or `JklsGalois`, a source at a time: for each source
    /// in order, `first` by the map hook into one block, its masks by the map hook into `masks.len()` blocks and the
    /// copy of the first into a term, each term prepared into one of `masks.len() + 1` operands by
    /// `ckks_left_operand_prepare`, then, for each block of `res`, poulpy-ckks's `ckks_dot_product_prepared_pt_vec_dft` of the
    /// first source's terms into the block's accumulator, carved from scratch, and `ckks_dot_product_prepared_pt_vec_dft_add`
    /// of the others'; each block of `res` is then finished by `ckks_dot_product_prepared_pt_vec_from_dft`. The heap holds
    /// one source's blocks and prepared terms, allocated once and refilled.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds less than the query reports for the rotation keys it looks up.
    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit_reference<D, S, K>(
        &self,
        res: &mut [&mut D],
        sources: &[&S],
        first: &LinearTransformationPrepared<BE>,
        masks: &[&LinearTransformationPrepared<BE>],
        plaintexts: &[&[CKKSRightOperandPrepared<BE>]],
        galois: &[i64],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + CKKSLeftOperandPreparedFactory<BE>
            + CKKSDotProductOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSModuleInfos
            + VecZnxDftBytesOf,
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;
}

impl<BE: Backend> BlasJklsReference<BE> for Module<BE> {
    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit_tmp_bytes_reference<R, S, K>(
        &self,
        outputs: usize,
        res: &R,
        src: &S,
        first: &LinearTransformationPrepared<BE>,
        masks: &[&LinearTransformationPrepared<BE>],
        plaintext: &CKKSRightOperandPrepared<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSLeftOperandPreparedFactory<BE> + CKKSDotProductOps<BE> + VecZnxDftBytesOf,
        R: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        let staged = mapped_block(layout_of(src), first);
        let term = mapped_block(staged, masks[0]);
        let cols = res.rank().as_usize() + 1;
        let acc = BE::scratch_aligned(self.vec_znx_dft_bytes_of(res.n().as_usize(), cols, term.size() + plaintext.size()));
        let step = BE::blas_map_blocks_tmp_bytes(self, 1, &staged, src, &[first], keys)
            .max(BE::blas_map_blocks_tmp_bytes(self, 1, &term, &staged, masks, keys))
            .max(self.ckks_copy_tmp_bytes(&term, &staged))
            .max(self.ckks_left_operand_prepare_tmp_bytes(&term))
            .max(self.ckks_dot_product_prepared_pt_vec_dft_tmp_bytes(res, &term, plaintext))
            .max(self.ckks_dot_product_prepared_pt_vec_dft_add_tmp_bytes(res, &term, plaintext))
            .max(self.ckks_dot_product_prepared_pt_vec_from_dft_tmp_bytes(res, &term, plaintext));
        acc.checked_mul(outputs)
            .and_then(|accs| accs.checked_add(step))
            .expect("blas_jkls_pt_unit: scratch bytes overflow usize")
    }

    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit_reference<D, S, K>(
        &self,
        res: &mut [&mut D],
        sources: &[&S],
        first: &LinearTransformationPrepared<BE>,
        masks: &[&LinearTransformationPrepared<BE>],
        plaintexts: &[&[CKKSRightOperandPrepared<BE>]],
        galois: &[i64],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + CKKSLeftOperandPreparedFactory<BE>
            + CKKSDotProductOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSModuleInfos
            + VecZnxDftBytesOf,
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        let op = "blas_jkls_pt_unit";
        let (Some(&src0), Some(res0)) = (sources.first(), res.first()) else {
            return Ok(());
        };
        let (inner, per) = (sources.len(), masks.len() + 1);
        let (src, dst) = (layout_of(src0), layout_of(&**res0));
        let staged = mapped_block(src, first);
        let term = mapped_block(staged, masks[0]);
        let plaintext = &plaintexts[0][0];
        let used = KeyLayouts::new(keys);
        let _ = ensure_lt_keys(op, self, first, &used, src.k(), staged.k(), src.rank());
        for mask in masks {
            let _ = ensure_lt_keys(op, self, mask, &used, staged.k(), term.k(), src.rank());
        }
        let bytes =
            used.max_bytes(|key| self.blas_jkls_pt_unit_tmp_bytes_reference(res.len(), &dst, &src, first, masks, plaintext, key));
        assert_scratch(op, bytes, scratch);

        let mut sigma = vec![self.ckks_ciphertext_alloc_from_infos(&staged)];
        let mut shifted: Vec<_> = masks.iter().map(|_| self.ckks_ciphertext_alloc_from_infos(&term)).collect();
        let mut narrowed = self.ckks_ciphertext_alloc_from_infos(&term);
        let mut lefts: Vec<CKKSLeftOperandPrepared<BE>> = (0..per)
            .map(|_| self.ckks_left_operand_prepared_alloc_from_infos(&term, PrepareHint::OneShot))
            .collect();
        let (n, cols, size) = (dst.n().as_usize(), dst.rank().as_usize() + 1, term.size() + plaintext.size());
        let mut rest = scratch.borrow();
        let mut accs = Vec::with_capacity(res.len());
        for _ in 0..res.len() {
            let (acc, r) = rest.take_vec_znx_dft_scratch(n, cols, size);
            accs.push(acc);
            rest = r;
        }
        let mut pts = Vec::with_capacity(per);
        for (l, &source) in sources.iter().enumerate() {
            BE::blas_map_blocks(self, &mut sigma, &[source], &[first], keys, &mut rest)?;
            BE::blas_map_blocks(self, &mut shifted, &[&sigma[0]], masks, keys, &mut rest)?;
            self.ckks_copy(&mut narrowed, &sigma[0], &mut rest)?;
            let (head, tail) = lefts.split_at_mut(1);
            self.ckks_left_operand_prepare(&mut head[0], &narrowed, &mut rest)?;
            for (left, shift) in tail.iter_mut().zip(&shifted) {
                self.ckks_left_operand_prepare(left, shift, &mut rest)?;
            }
            let terms: Vec<&CKKSLeftOperandPrepared<BE>> = lefts.iter().collect();
            for (o, (block, acc)) in res.iter().zip(&mut accs).enumerate() {
                let block_pts = plaintexts[o * inner + l];
                pts.clear();
                pts.extend((0..per).map(|k| &block_pts[k % block_pts.len()]));
                let mut acc = acc.to_backend_mut();
                if l == 0 {
                    self.ckks_dot_product_prepared_pt_vec_dft(&mut acc, &**block, &terms, &pts, galois, &mut rest)?;
                } else {
                    self.ckks_dot_product_prepared_pt_vec_dft_add(&mut acc, &**block, &terms, &pts, galois, &mut rest)?;
                }
            }
        }
        for (o, (block, acc)) in res.iter_mut().zip(&mut accs).enumerate() {
            let pt = &plaintexts[o * inner][0];
            self.ckks_dot_product_prepared_pt_vec_from_dft(&mut **block, &mut acc.to_backend_mut(), &lefts[0], pt, &mut rest)?;
        }
        Ok(())
    }
}
