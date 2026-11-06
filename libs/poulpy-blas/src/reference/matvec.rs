use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSModuleInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSDotProductOps, CKKSLeftOperandPreparedFactory, CKKSLinearTransformationOps, CKKSMulOps, CKKSRotateOps},
    layouts::{CKKSRightOperandPrepared, ScratchArenaTakeCKKSCiphertext},
};
use poulpy_core::{
    GLWEBytesOf, GLWELinearTransformationBabyStepsBytesOf, ScratchArenaTakeGLWELinearTransformationBabySteps,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, PrepareHint, ScratchArena},
};

use super::{LtEval, baby_caches, lt_tmp_bytes, shared_giants};
use crate::layouts::{
    BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextMap, BlasGrid, BlasGridInfos, BlasGridLine,
    BlasGridMut, BlasMatVecMaps, BlasVectorInfos, BlasVectorLayout, BlasVectorMut, BlasVectorRef,
    diagonal_ciphertext::{self, AppendAxis, append_rotations},
    matvec_operand::output_label,
    validate::{
        KeyLayouts, append_uses, assert_scratch, ensure_rotation, ensure_tensor_key, matvec_diagonal_tensor,
        matvec_diagonal_uses, matvec_pt_uses, rotates, validate_blas_append_col_diagonal, validate_blas_append_row_diagonal,
        validate_blas_grid_to_vector, validate_blas_matvec_diagonal, validate_blas_matvec_pt, validate_blas_vector_to_grid,
    },
};

/// Reference compositions of [`crate::api::BlasMatVecOps`]. Each runs its validator,
/// [`crate::layouts::validate_blas_matvec_pt`], before any write, so its errors leave `res` untouched, then asserts its
/// scratch and returns poulpy-ckks's budget errors before any write. It runs ciphertext after ciphertext on one
/// scratch, and the first error stops it.
pub trait BlasMatVecReference<BE: Backend> {
    /// The baby steps of one input ciphertext and one summed linear transformation from ciphertexts of `a`'s layout into
    /// one of `res`'s, by `w`'s diagonals: resident, or streamed for an unprepared operand. On maps that rotate nothing,
    /// `keys` is never read and the smallest key covering `a`'s layout stands in for it.
    fn blas_matvec_pt_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// One baby-step cache per input ciphertext `a_i`, covering the baby steps of `w`'s maps, which every output
    /// ciphertext shares, then every output ciphertext `res_o` is poulpy-ckks's summed linear transformation of the `K`
    /// terms `(a_i, w_oi)`, in ascending `i`, which shares the giant steps across them, stamped with the label
    /// `log2(S / output_period)`, for `S` the operand's slots.
    fn blas_matvec_pt_reference<R, W, A, K>(
        &self,
        res: &mut R,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// [`Self::blas_matvec_pt_tmp_bytes_reference`] from `a`'s blocks.
    fn blas_grid_to_vector_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// [`Self::blas_matvec_pt_reference`] with the blocks holding `line`, in entry order, as the input ciphertexts.
    fn blas_grid_to_vector_reference<R, W, A, K>(
        &self,
        res: &mut R,
        w: &W,
        a: &A,
        line: BlasGridLine,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// [`Self::blas_matvec_pt_tmp_bytes_reference`] into `res`'s blocks.
    fn blas_vector_to_grid_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// [`Self::blas_matvec_pt_reference`] with the blocks of `res`, which all hold `line`, in entry order, as the output
    /// ciphertexts, then the batch `output_period / d^2` recorded on `res`.
    fn blas_vector_to_grid_reference<R, W, A, K>(
        &self,
        res: &mut R,
        line: BlasGridLine,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// The baby-step cache of `a` beside the larger of its preparation and poulpy-ckks's transform of `m`'s ciphertext
    /// diagonals from `a`'s layout and from `res`'s. On a schedule that rotates nothing, `keys` is never read and
    /// `tsk`'s layout stands in for it.
    fn blas_matvec_diagonal_tmp_bytes_reference<R, M, A, T, K>(&self, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// One baby-step cache of `a`, prepared as `m`'s schedule reaches its rotations, then poulpy-ckks's
    /// `ckks_linear_transformation_eval_into` of `m`'s ciphertext diagonals into `res`, resident or prepared at each use,
    /// each giant step's products relinearized once with `tsk`.
    fn blas_matvec_diagonal_reference<R, M, A, K>(
        &self,
        res: &mut R,
        m: &M,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + Sized,
        R: BlasVectorMut<BE>,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// The largest of a rotation of `v` into a scratch ciphertext and its preparation as a left operand, and
    /// poulpy-ckks's masked accumulate into a diagonal of `m`; the rotation only when the row append rotates `v`.
    fn blas_append_row_diagonal_tmp_bytes_reference<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSDotProductOps<BE> + GLWEBytesOf<BE>,
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// Appends row `row` of `m`, `mask` marking the slots it covers: for each baby step `b` the diagonals read, in
    /// ascending order, `rot(v, b)` prepared as a left operand, then every entry `(b, g)` receives `mask` read through the
    /// Galois element of `-(row + g)` times `rot(v, b)`, in storage order, by poulpy-ckks's
    /// `ckks_dot_product_prepared_pt_vec_add`.
    fn blas_append_row_diagonal_reference<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        row: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSRotateOps<BE>
            + CKKSMulOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEBytesOf<BE>
            + Sized,
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_append_row_diagonal_tmp_bytes_reference`], for a column append.
    fn blas_append_col_diagonal_tmp_bytes_reference<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSDotProductOps<BE> + GLWEBytesOf<BE>,
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// Appends column `col` of `m` as [`Self::blas_append_row_diagonal_reference`] appends a row: for each giant step `g`, in
    /// ascending order of `-g`, `rot(v, -g)` prepared as a left operand, then every entry `(b, g)` receives `mask` read
    /// through the Galois element of `b - col` times `rot(v, -g)`.
    fn blas_append_col_diagonal_reference<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        col: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSRotateOps<BE>
            + CKKSMulOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEBytesOf<BE>
            + Sized,
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatVecReference<BE> for Module<BE>
where
    Module<BE>: CKKSLeftOperandPreparedFactory<BE>,
{
    fn blas_matvec_pt_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos,
    {
        maps_tmp_bytes(self, w, res.ct_layout(), a.ct_layout(), keys)
    }

    fn blas_matvec_pt_reference<R, W, A, K>(
        &self,
        res: &mut R,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        let op = "blas_matvec_pt";
        validate_blas_matvec_pt(self, &*res, w, a, keys)?;
        let used = KeyLayouts::new(keys);
        let _ = matvec_pt_uses(&*res, w.blocks(), a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes_or_unused(|key| {
            let key = key.copied().unwrap_or_else(|| covering_key(&a.ct_layout()));
            self.blas_matvec_pt_tmp_bytes_reference(&*res, w, a, &key)
        });
        assert_scratch(op, bytes, scratch);
        let cts = a.cts();
        apply_maps(
            self,
            res.cts_mut().iter_mut().collect(),
            cts.len(),
            |i| &cts[i],
            w,
            keys,
            scratch,
        )
    }

    fn blas_grid_to_vector_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        maps_tmp_bytes(self, w, res.ct_layout(), a.block_layout(), keys)
    }

    fn blas_grid_to_vector_reference<R, W, A, K>(
        &self,
        res: &mut R,
        w: &W,
        a: &A,
        line: BlasGridLine,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        let op = "blas_grid_to_vector";
        validate_blas_grid_to_vector(self, &*res, w, a, line, keys)?;
        let used = KeyLayouts::new(keys);
        let view = line.vector_layout(a);
        let _ = matvec_pt_uses(&*res, w.blocks(), &view, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes_or_unused(|key| {
            let key = key.copied().unwrap_or_else(|| covering_key(&a.block_layout()));
            self.blas_grid_to_vector_tmp_bytes_reference(&*res, w, a, &key)
        });
        assert_scratch(op, bytes, scratch);
        let blocks = a.blocks();
        let source = |t| &blocks[line.block(a, t)];
        apply_maps(
            self,
            res.cts_mut().iter_mut().collect(),
            view.shape.count(),
            source,
            w,
            keys,
            scratch,
        )
    }

    fn blas_vector_to_grid_tmp_bytes_reference<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos,
    {
        maps_tmp_bytes(self, w, res.block_layout(), a.ct_layout(), keys)
    }

    fn blas_vector_to_grid_reference<R, W, A, K>(
        &self,
        res: &mut R,
        line: BlasGridLine,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        let op = "blas_vector_to_grid";
        let batch = validate_blas_vector_to_grid(self, &*res, line, w, a, keys)?;
        let used = KeyLayouts::new(keys);
        let view = line.vector_layout(&*res);
        let _ = matvec_pt_uses(&view, w.blocks(), a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes_or_unused(|key| {
            let key = key.copied().unwrap_or_else(|| covering_key(&a.ct_layout()));
            self.blas_vector_to_grid_tmp_bytes_reference(&*res, w, a, &key)
        });
        assert_scratch(op, bytes, scratch);
        let cts = a.cts();
        apply_maps(
            self,
            res.blocks_mut().iter_mut().collect(),
            cts.len(),
            |i| &cts[i],
            w,
            keys,
            scratch,
        )?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matvec_diagonal_tmp_bytes_reference<R, M, A, T, K>(&self, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        let shape = diagonal_ciphertext::shape(&m.blas_diagonal_ciphertext_layout());
        let keys = rotates(self, |f| matvec_diagonal_uses(res, &shape, a, f)).then(|| keys.gglwe_layout());
        matvec_diagonal_tmp_bytes(self, M::STREAMED, res, m, a, &tsk.gglwe_layout(), keys.as_ref())
    }

    fn blas_matvec_diagonal_reference<R, M, A, K>(
        &self,
        res: &mut R,
        m: &M,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + ModuleN + Sized,
        R: BlasVectorMut<BE>,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        let op = "blas_matvec_diagonal";
        validate_blas_matvec_diagonal(self, &*res, m, a, keys)?;
        let (used_keys, used_tsk) = (KeyLayouts::new(keys), KeyLayouts::new(keys));
        let _ = matvec_diagonal_uses(&*res, m.map(), a, &mut |u| ensure_rotation(op, self, &used_keys, u));
        let _ = ensure_tensor_key(
            op,
            &used_tsk,
            matvec_diagonal_tensor(&m.blas_diagonal_ciphertext_layout(), a),
            a.ct_layout().rank(),
        );
        let bytes = used_tsk.max_bytes(|tensor| {
            used_keys.max_bytes_or_unused(|key| matvec_diagonal_tmp_bytes(self, M::STREAMED, &*res, m, a, tensor, key))
        });
        assert_scratch(op, bytes, scratch);
        let src = &a.cts()[0];
        let (mut babies, mut scratch) =
            scratch
                .borrow()
                .take_glwe_linear_transformation_baby_steps_scratch(self, &m.map().index(), src);
        self.ckks_linear_transformation_baby_steps_prepare(&mut babies, src, keys, &mut scratch.borrow())?;
        self.ckks_linear_transformation_eval_into(&mut res.cts_mut()[0], src, &babies, m.map(), keys, &mut scratch)
    }

    fn blas_append_row_diagonal_tmp_bytes_reference<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSDotProductOps<BE> + GLWEBytesOf<BE>,
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos,
    {
        append_query(self, AppendAxis::Row, m, v, mask, keys)
    }

    fn blas_append_row_diagonal_reference<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        row: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSRotateOps<BE>
            + CKKSMulOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEBytesOf<BE>
            + Sized,
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_append_row_diagonal(self, m, v, mask, keys)?;
        append(self, AppendAxis::Row, m, row, v, mask, keys, scratch)
    }

    fn blas_append_col_diagonal_tmp_bytes_reference<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSDotProductOps<BE> + GLWEBytesOf<BE>,
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos,
    {
        append_query(self, AppendAxis::Col, m, v, mask, keys)
    }

    fn blas_append_col_diagonal_reference<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        col: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSRotateOps<BE>
            + CKKSMulOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEBytesOf<BE>
            + Sized,
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_append_col_diagonal(self, m, v, mask, keys)?;
        append(self, AppendAxis::Col, m, col, v, mask, keys, scratch)
    }
}

/// [`BlasMatVecReference::blas_matvec_diagonal_tmp_bytes_reference`] of diagonals prepared at each use when `streamed`,
/// on rotation keys of layout `keys`, or of `tsk` where the schedule rotates nothing (`None`).
fn matvec_diagonal_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    streamed: bool,
    res: &impl BlasVectorInfos,
    m: &impl BlasDiagonalCiphertextInfos,
    a: &impl BlasVectorInfos,
    tsk: &GGLWELayout,
    keys: Option<&GGLWELayout>,
) -> usize
where
    Module<BE>: CKKSLinearTransformationOps<BE>,
{
    let (layout, res, a) = (m.blas_diagonal_ciphertext_layout(), res.ct_layout(), a.ct_layout());
    let (plan, keys) = (layout.lt.index(), keys.unwrap_or(tsk));
    let eval = |ct| {
        if streamed {
            module.ckks_linear_transformation_eval_streamed_tmp_bytes(ct, &layout.ct, &plan, tsk, keys)
        } else {
            module.ckks_linear_transformation_eval_tmp_bytes(ct, &layout.ct, &plan, tsk, keys)
        }
    };
    // The baby-step cache of `a` beside the preparation and either evaluation.
    module.glwe_linear_transformation_baby_steps_bytes_of(&plan, &a)
        + module
            .ckks_linear_transformation_baby_steps_prepare_tmp_bytes(&a, &plan, keys)
            .max(eval(&a))
            .max(eval(&res))
}

/// The append query along `axis` on rotation keys of layout `keys`, read only when the append rotates `v`.
fn append_query<BE: Backend>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &impl BlasDiagonalCiphertextInfos,
    v: &impl BlasVectorInfos,
    mask: &impl GLWEInfos,
    keys: &impl GGLWEInfos,
) -> usize
where
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + CKKSLeftOperandPreparedFactory<BE>
        + CKKSDotProductOps<BE>
        + GLWEBytesOf<BE>,
{
    let shape = diagonal_ciphertext::shape(&m.blas_diagonal_ciphertext_layout());
    let keys = rotates(module, |f| append_uses(axis, &shape, v, f)).then(|| keys.gglwe_layout());
    append_tmp_bytes(module, m, v, mask, keys.as_ref())
}

/// The append query on rotation keys of layout `keys`, `None` for an append that rotates nothing: a rotation of `v`
/// into a scratch ciphertext then its preparation, or `v`'s preparation alone, a masked accumulate and the
/// preparation of the diagonals.
fn append_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    m: &impl BlasDiagonalCiphertextInfos,
    v: &impl BlasVectorInfos,
    mask: &impl GLWEInfos,
    keys: Option<&GGLWELayout>,
) -> usize
where
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + CKKSLeftOperandPreparedFactory<BE>
        + CKKSDotProductOps<BE>
        + GLWEBytesOf<BE>,
{
    let (diagonal, v) = (m.blas_diagonal_ciphertext_layout().ct, v.ct_layout());
    let left = module.ckks_left_operand_prepare_tmp_bytes(&v);
    let rotated = keys.map_or(0, |keys| {
        BE::scratch_aligned(module.glwe_bytes_of_from_infos(&v)) + module.ckks_rotate_tmp_bytes(&v, keys).max(left)
    });
    rotated
        .max(left)
        .max(module.ckks_dot_product_prepared_pt_vec_add_tmp_bytes(&diagonal, &v, mask))
}

/// The body of the row (`axis` [`AppendAxis::Row`]) and column appends at `at`, after their validators: one prepared
/// left operand, refilled for each rotation of `v` ([`append_rotations`]), the rotation made in scratch, and read by the
/// entries of that rotation.
#[allow(clippy::too_many_arguments)]
fn append<BE, V, K>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &mut BlasDiagonalCiphertext<BE>,
    at: i64,
    v: &V,
    mask: &CKKSRightOperandPrepared<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + CKKSLeftOperandPreparedFactory<BE>
        + CKKSDotProductOps<BE>
        + GLWEBytesOf<BE>,
    V: BlasVectorRef<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let op = axis.op();
    let used = KeyLayouts::new(keys);
    let _ = append_uses(axis, &m.diagonals, v, &mut |u| ensure_rotation(op, module, &used, u));
    let bytes = used.max_bytes_or_unused(|key| append_tmp_bytes(module, &*m, v, mask, key));
    assert_scratch(op, bytes, scratch);
    let (src, layout, slots) = (&v.cts()[0], v.ct_layout(), module.ckks_max_slots() as i64);
    let at = at.rem_euclid(slots);
    let mut left = module.ckks_left_operand_prepared_alloc_from_infos(&layout, PrepareHint::Reuse);
    for r in append_rotations(axis, &m.diagonals) {
        if r.rem_euclid(slots) == 0 {
            module.ckks_left_operand_prepare(&mut left, src, scratch)?;
        } else {
            let (mut rotated, mut s) = scratch.borrow().take_ckks_ciphertext_like_scratch(&layout);
            module.ckks_rotate_into(r, &mut rotated, src, keys, &mut s)?;
            module.ckks_left_operand_prepare(&mut left, &rotated, &mut s)?;
        }
        for gs in &mut m.diagonals.giant_steps {
            let giant = gs.rot;
            for d in &mut gs.diagonals {
                let shift = match axis {
                    AppendAxis::Row if d.baby == r => -(at + giant),
                    AppendAxis::Col if -giant == r => d.baby - at,
                    _ => continue,
                };
                let galois = module.ckks_galois_element(shift.rem_euclid(slots));
                module.ckks_dot_product_prepared_pt_vec_add(&mut d.plaintext, &[&left], &[mask], &[galois], scratch)?;
            }
        }
    }
    Ok(())
}

/// The smallest rotation key covering an input of layout `a`, one digit per limb, which sizes the scratch of maps that
/// look up no key: any key covering `a` sizes at least as much.
fn covering_key(a: &CKKSLayout) -> GGLWELayout {
    let k_digit = TorusPrecision(a.base2k().0);
    GGLWELayout {
        n: a.n(),
        base2k: a.base2k(),
        dnum: GGLWELayout::dnum_for_input(a.k(), k_digit),
        k_aux: k_digit,
        rank_in: a.rank(),
        rank_out: a.rank(),
        k_digit,
        k_offset: TorusPrecision(0),
        stride: 1,
    }
}

/// One baby-step cache of an input of layout `a` and one summed transform from it into an output of layout `res` by
/// `w`'s diagonals: resident, or streamed for an unprepared operand, at `keys`' layout, or at [`covering_key`] for maps
/// that look up no key.
fn maps_tmp_bytes<BE, W>(module: &Module<BE>, w: &W, res: CKKSLayout, a: CKKSLayout, keys: &impl GGLWEInfos) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    W: BlasMatVecMaps<BE>,
{
    let views = (BlasVectorLayout::new(res, w.output()), BlasVectorLayout::new(a, w.input()));
    let key = if rotates(module, |f| matvec_pt_uses(&views.0, w.blocks(), &views.1, f)) {
        keys.gglwe_layout()
    } else {
        covering_key(&a)
    };
    let eval = if W::STREAMED { LtEval::StreamedSum } else { LtEval::Sum };
    lt_tmp_bytes(module, &res, &a, &key, eval)
}

/// The products by `w`'s maps: one baby-step cache per input `source(i)`, `i < inputs`, which every output shares, then
/// output `o` of `dsts` is poulpy-ckks's summed linear transformation of the terms `(source(i), w_oi)` in ascending
/// `i`, stamped with the label `w`'s output period proves.
fn apply_maps<'s, BE, M, D, S, W, K>(
    module: &M,
    dsts: Vec<&mut D>,
    inputs: usize,
    source: impl Fn(usize) -> &'s S,
    w: &W,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + 's,
    W: BlasMatVecMaps<BE>,
    K: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    let maps = w.blocks();
    // Every map shares one schedule, so the first one's baby steps serve them all.
    let mut caches = baby_caches(module, &[&maps[0]], source(0), inputs);
    shared_giants(
        module,
        &mut caches,
        vec![dsts],
        |_, i| source(i),
        |_, o, i| &maps[o * inputs + i],
        output_label(w),
        keys,
        scratch,
    )
}
