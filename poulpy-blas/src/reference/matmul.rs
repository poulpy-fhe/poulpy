use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformationBabySteps,
        LinearTransformationPrepared,
    },
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Data, Module, Ring, ScratchArena, ZnxWord},
};

use super::{
    KeyLayouts, assert_scratch, ensure_grid, ensure_inner, ensure_lt_keys, ensure_maps, ensure_shape, ensure_tensor_key,
    lt_output_k, lt_tmp_bytes, reject, stamp_sparsity,
};
use crate::layouts::{BlasCiphertext, BlasContext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand};

/// Reference compositions of [`crate::api::BlasMatMulOps`].
pub trait BlasMatMulReference<BE: Backend> {
    /// `2 (m + n) + 1` block ciphertexts: sigma and its shifts of a block column of `a`, tau and its shifts of a block
    /// row of `b`, and the product.
    fn blas_matmul_tmp_bytes_reference<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        let (a_layout, b_layout) = (a.block_layout(), b.block_layout());
        let ct_a = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&a_layout));
        let ct_b = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&b_layout));
        a.rows() * ct_a + b.cols() * ct_b + self.blas_matmul_prepared_tmp_bytes_reference(res, a, b, tsk, atk)
    }

    /// For `l` in `0..K`: `S_i = sigma(a_il)` for every `i`, `T_j = tau(b_lj)` for every `j`, then the terms of
    /// [`Self::blas_matmul_prepared_reference`] on `S` and `T`.
    fn blas_matmul_reference<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSModuleInfos
            + GLWEBytesOf<BE>
            + GLWENormalize<BE>,
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
    {
        let op = "blas_matmul";
        ensure_product(op, self, res, a, b, ctx)?;
        // `S_i` and `T_j` are carved at their inputs' widths, so sigma and tau narrow them by their scale only.
        let (a_k, b_k) = (a.block_layout().k(), b.block_layout().k());
        let used = KeyLayouts::new(keys);
        ensure_lt_keys(op, self, ctx.sigma(), &used, a_k, a_k)?;
        ensure_lt_keys(op, self, ctx.tau(), &used, b_k, b_k)?;
        ensure_jkls_keys(
            op,
            self,
            ctx,
            &used,
            lt_output_k(ctx.sigma(), a_k, a_k),
            lt_output_k(ctx.tau(), b_k, b_k),
        )?;
        let bytes = used.max_pair_bytes(|tsk, atk| self.blas_matmul_tmp_bytes_reference(&*res, a, b, tsk, atk));
        assert_scratch(op, bytes, scratch);
        let (res_layout, a_layout, b_layout) = (res.block_layout(), a.block_layout(), b.block_layout());
        scratch.scope(|s| -> Result<()> {
            let (mut sl, s) = s.take_ckks_ciphertext_slice_scratch(a.rows(), &a_layout, a_layout.meta);
            let (mut tl, mut s) = s.take_ckks_ciphertext_slice_scratch(b.cols(), &b_layout, b_layout.meta);
            let mut held = 0;
            for l in 0..a.cols() {
                for (i, si) in sl.iter_mut().enumerate() {
                    self.ckks_eval_linear_transformation_self_into(si, a.at(i, l), ctx.sigma(), keys, &mut s)?;
                }
                for (j, tj) in tl.iter_mut().enumerate() {
                    self.ckks_eval_linear_transformation_self_into(tj, b.at(l, j), ctx.tau(), keys, &mut s)?;
                }
                let (si, tj): (Vec<_>, Vec<_>) = (sl.iter().collect(), tl.iter().collect());
                held = jkls_terms(self, res, &si, &tj, held, &res_layout, ctx, keys, &mut s)?;
            }
            Ok(())
        })
    }

    fn blas_matmul_prepare_left_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), key)
    }

    /// `res_il = sigma(a_il)`, one linear transformation per block.
    fn blas_matmul_prepare_left_reference<DR, DA, K>(
        &self,
        res: &mut BlasMatMulLeft<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos,
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        map_blocks(
            self,
            "blas_matmul_prepare_left",
            res.as_grid_mut(),
            a,
            ctx,
            ctx.sigma(),
            keys,
            scratch,
        )
    }

    fn blas_matmul_prepare_right_tmp_bytes_reference<R, B, K>(&self, res: &R, b: &B, key: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, &res.block_layout(), &b.block_layout(), key)
    }

    /// `res_lj = tau(b_lj)`, one linear transformation per block.
    fn blas_matmul_prepare_right_reference<DR, DB, K>(
        &self,
        res: &mut BlasMatMulRight<DR, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos,
        DR: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        map_blocks(
            self,
            "blas_matmul_prepare_right",
            res.as_grid_mut(),
            b,
            ctx,
            ctx.tau(),
            keys,
            scratch,
        )
    }

    /// `m + n + 1` block ciphertexts: the shifts of a block column of `a` and of a block row of `b`, and the product.
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        let (res_layout, a_layout, b_layout) = (res.block_layout(), a.block_layout(), b.block_layout());
        let ct_a = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&a_layout));
        let ct_b = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&b_layout));
        let ct_res = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res_layout));
        a.rows() * ct_a
            + b.cols() * ct_b
            + ct_res
            + lt_tmp_bytes(self, &a_layout, &a_layout, atk)
                .max(lt_tmp_bytes(self, &b_layout, &b_layout, atk))
                .max(self.ckks_mul_tmp_bytes(&res_layout, &a_layout, &b_layout, tsk))
                .max(self.ckks_add_tmp_bytes(res_layout.size()))
                .max(self.glwe_normalize_tmp_bytes())
    }

    /// For `l` in `0..K` and `k` in `0..d`: `P_i = phi^k(a_il)` for every `i`, `Q_j = psi^k(b_lj)` for every `j`
    /// (the blocks themselves at `k = 0`), then for every `(i, j)` in row-major order `res_ij = P_i (.) Q_j` on the first
    /// term and `res_ij += P_i (.) Q_j` through a temporary otherwise. Every block of `res` is normalized before an `l`
    /// whose `d` terms would take it past `2^(63 - base2k)` lazy terms.
    fn blas_matmul_prepared_reference<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasMatMulLeft<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasMatMulRight<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSModuleInfos
            + GLWEBytesOf<BE>
            + GLWENormalize<BE>,
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
    {
        let op = "blas_matmul_prepared";
        let (a, b) = (a.as_grid(), b.as_grid());
        ensure_product(op, self, res, a, b, ctx)?;
        let used = KeyLayouts::new(keys);
        ensure_jkls_keys(op, self, ctx, &used, a.block_layout().k(), b.block_layout().k())?;
        let bytes = used.max_pair_bytes(|tsk, atk| self.blas_matmul_prepared_tmp_bytes_reference(&*res, a, b, tsk, atk));
        assert_scratch(op, bytes, scratch);
        let res_layout = res.block_layout();
        let mut held = 0;
        for l in 0..a.cols() {
            let si: Vec<_> = (0..a.rows()).map(|i| a.at(i, l)).collect();
            let tj: Vec<_> = (0..b.cols()).map(|j| b.at(l, j)).collect();
            held = jkls_terms(self, res, &si, &tj, held, &res_layout, ctx, keys, scratch)?;
        }
        Ok(())
    }

    /// One block-shaped temporary on top of the linear transformation, the addition and the normalization.
    fn blas_matmul_pt_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (res, a) = (res.block_layout(), a.block_layout());
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + lt_tmp_bytes(self, &res, &a, key)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.glwe_normalize_tmp_bytes())
    }

    /// One baby-step cache per call, holding the baby steps of every block of `b`; for `i` in `0..m`, `l` in `0..K`:
    /// the cache prepared on `a_il` and shared by every `b_lj`, then `res_ij = b_lj(a_il)` for `l = 0`, otherwise
    /// `b_lj(a_il)` into a temporary added to `res_ij`. Row `i` of `res` is normalized before an `l` whose term would take
    /// it past `2^(63 - base2k)` lazy terms.
    fn blas_matmul_pt_reference<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasPtOperand<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSAddOps<BE>
            + CKKSModuleInfos
            + GLWEBytesOf<BE>
            + GLWENormalize<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        let op = "blas_matmul_pt";
        ensure_grid::<BE::Ring>(op, b.log_d(), a, a.blocks())?;
        ensure_grid::<BE::Ring>(op, b.log_d(), res, res.blocks())?;
        ensure_inner(op, a.cols(), b.rows())?;
        ensure_shape(op, res, a.rows(), b.cols())?;
        ensure_maps(op, self, b.slots(), b.base2k(), &[a])?;
        if !a.mat_slots.is_multiple_of(b.count()) {
            return Err(reject(
                op,
                format!("{} public matrices do not divide the {} matrix slots", b.count(), a.mat_slots),
            ));
        }
        let maps = || (0..b.rows()).flat_map(|l| (0..b.cols()).map(move |j| b.at(l, j)));
        let res_layout = res.block_layout();
        ensure_lazy_sums(op, res_layout.base2k().as_usize(), a.cols(), 1)?;
        let used = KeyLayouts::new(keys);
        for lt in maps() {
            ensure_lt_keys(op, self, lt, &used, a.block_layout().k(), res_layout.k())?;
        }
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        let rotations: Vec<i64> = maps().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, a.at(0, 0));
        for i in 0..a.rows() {
            let mut held = 0;
            for l in 0..a.cols() {
                let src = a.at(i, l);
                self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
                held = make_room(self, &mut res.blocks_mut()[i * b.cols()..][..b.cols()], held, 1, scratch);
                for j in 0..b.cols() {
                    if l == 0 {
                        self.ckks_eval_linear_transformation_into(res.at_mut(i, j), src, &babies, b.at(l, j), keys, scratch)?;
                        continue;
                    }
                    scratch.scope(|s| -> Result<()> {
                        let (mut tmp, mut s) = s.take_ckks_ciphertext_like_scratch(&res_layout);
                        self.ckks_eval_linear_transformation_into(&mut tmp, src, &babies, b.at(l, j), keys, &mut s)?;
                        self.ckks_add_assign(res.at_mut(i, j), &tmp, &mut s)
                    })?;
                }
            }
        }
        stamp_sparsity(res.blocks_mut(), a);
        Ok(())
    }
}

impl<BE: Backend> BlasMatMulReference<BE> for Module<BE> {}

/// `res` is the `m x n` product grid of the `m x K` grid `a` and the `K x n` grid `b`, all read at the context's dimension,
/// whose blocks take the `d` terms of an inner index (`ensure_lazy_sums`); the context's maps read `a` and `b`
/// (`ensure_maps`).
fn ensure_product<M, DR, DA, DB, W, R, BE>(
    op: &'static str,
    module: &M,
    res: &BlasCiphertext<DR, W, R>,
    a: &BlasCiphertext<DA, W, R>,
    b: &BlasCiphertext<DB, W, R>,
    ctx: &BlasContext<BE>,
) -> Result<()>
where
    M: CKKSModuleInfos + ?Sized,
    DR: Data,
    DA: Data,
    DB: Data,
    W: ZnxWord,
    R: Ring,
    BE: Backend,
{
    ensure_grid::<R>(op, ctx.log_d(), a, a.blocks())?;
    ensure_grid::<R>(op, ctx.log_d(), b, b.blocks())?;
    ensure_grid::<R>(op, ctx.log_d(), res, res.blocks())?;
    ensure_inner(op, a.cols(), b.rows())?;
    ensure_shape(op, res, a.rows(), b.cols())?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[a, b])?;
    ensure_lazy_sums(op, res.block_layout().base2k().as_usize(), a.cols(), 1 << ctx.log_d())
}

/// Most lazy terms an `i64` digit at radix `base2k` holds: poulpy-ckks's `2^(63 - base2k)`.
fn max_lazy_terms(base2k: usize) -> usize {
    63usize.checked_sub(base2k).map_or(0, |shift| 1 << shift)
}

/// Rejects `inner` steps of `step` lazy terms each into blocks at radix `base2k` when one step passes [`max_lazy_terms`]
/// even on a normalized block, which holds one term from the second step on.
fn ensure_lazy_sums(op: &'static str, base2k: usize, inner: usize, step: usize) -> Result<()> {
    let terms = step + usize::from(inner > 1);
    if terms > max_lazy_terms(base2k) {
        return Err(reject(op, format!("{terms} lazy terms risk i64 overflow at base2k={base2k}")));
    }
    Ok(())
}

/// The lazy terms each of `blocks` holds once `step` more land on its `held`, `0` before the first: blocks the step would
/// take past [`max_lazy_terms`] are normalized first, then hold one.
fn make_room<BE, M, D>(
    module: &M,
    blocks: &mut [CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
    held: usize,
    step: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> usize
where
    BE: Backend,
    M: GLWENormalize<BE> + ?Sized,
    D: Data,
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
{
    if held + step <= max_lazy_terms(blocks[0].base2k().as_usize()) {
        return held + step;
    }
    for block in blocks {
        module.glwe_normalize_assign(block, scratch);
    }
    1 + step
}

/// `res_ij = lt(a_ij)` for every block, after the checks of a same-shape operation.
#[allow(clippy::too_many_arguments)]
fn map_blocks<BE, M, DR, DA, K>(
    module: &M,
    op: &'static str,
    res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
    a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
    ctx: &BlasContext<BE>,
    lt: &LinearTransformationPrepared<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ?Sized,
    DR: Data,
    DA: Data,
    CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
    CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
    K: GetAutomorphismKey<BE>,
{
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, ctx.log_d(), res, res.blocks())?;
    ensure_shape(op, res, a.rows(), a.cols())?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[a])?;
    let used = KeyLayouts::new(keys);
    ensure_lt_keys(op, module, lt, &used, a.block_layout().k(), res.block_layout().k())?;
    let bytes = used.max_bytes(|key| lt_tmp_bytes(module, &res.block_layout(), &a.block_layout(), key));
    assert_scratch(op, bytes, scratch);
    for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
        module.ckks_eval_linear_transformation_self_into(dst, src, lt, keys, scratch)?;
    }
    stamp_sparsity(res.blocks_mut(), a);
    Ok(())
}

/// The keys of the JKLS terms from `s_i` of width `s_k` and `t_j` of width `t_k`, at the widths `jkls_terms` uses
/// them on inputs it evaluates: the shifts `P_i` and `Q_j` are carved at `s_k` and `t_k` and each one is evaluated into
/// the previous one, so shift `k` evaluates into the width shift `k - 1` produced; every product looks the tensor key up
/// at its wider operand.
fn ensure_jkls_keys<BE, M, K>(
    op: &'static str,
    module: &M,
    ctx: &BlasContext<BE>,
    keys: &K,
    s_k: TorusPrecision,
    t_k: TorusPrecision,
) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
{
    ensure_tensor_key(op, keys, s_k.max(t_k))?;
    let (mut p_k, mut q_k) = (s_k, t_k);
    for k in 1..1usize << ctx.log_d() {
        let (phi, psi) = (ctx.rotate_cols(k), ctx.rotate_rows(k));
        ensure_lt_keys(op, module, phi, keys, s_k, p_k)?;
        ensure_lt_keys(op, module, psi, keys, t_k, q_k)?;
        (p_k, q_k) = (lt_output_k(phi, s_k, p_k), lt_output_k(psi, t_k, q_k));
        ensure_tensor_key(op, keys, p_k.max(q_k))?;
    }
    Ok(())
}

/// The JKLS terms of one inner index `l`, from `s_i = sigma(a_il)` and `t_j = tau(b_lj)`: for `k` in `0..d`, the shifts
/// `P_i = phi^k(s_i)` and `Q_j = psi^k(t_j)` (`s_i` and `t_j` at `k = 0`), then `accumulate`, after `make_room` for the
/// `d` terms on the `held` every block of `res` holds, `0` at `l = 0`. Returns the terms the blocks hold after.
#[allow(clippy::too_many_arguments)]
fn jkls_terms<BE, M, DR, S, T, K>(
    module: &M,
    res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
    s: &[&S],
    t: &[&T],
    held: usize,
    res_layout: &CKKSLayout,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<usize>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + GLWENormalize<BE> + ?Sized,
    DR: Data,
    CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
{
    let d = 1usize << ctx.log_d();
    scratch.scope(|sc| -> Result<usize> {
        let (mut p, sc) = sc.take_ckks_ciphertext_slice_scratch(s.len(), s[0], s[0].meta());
        let (mut q, sc) = sc.take_ckks_ciphertext_slice_scratch(t.len(), t[0], t[0].meta());
        let (mut prod, mut sc) = sc.take_ckks_ciphertext_like_scratch(res_layout);
        let terms = make_room(module, res.blocks_mut(), held, d, &mut sc);
        accumulate(module, res, s, t, held == 0, &mut prod, keys, &mut sc)?;
        for k in 1..d {
            for (pi, si) in p.iter_mut().zip(s) {
                module.ckks_eval_linear_transformation_self_into(pi, *si, ctx.rotate_cols(k), keys, &mut sc)?;
            }
            for (qj, tj) in q.iter_mut().zip(t) {
                module.ckks_eval_linear_transformation_self_into(qj, *tj, ctx.rotate_rows(k), keys, &mut sc)?;
            }
            let (pi, qj): (Vec<_>, Vec<_>) = (p.iter().collect(), q.iter().collect());
            accumulate(module, res, &pi, &qj, false, &mut prod, keys, &mut sc)?;
        }
        Ok(terms)
    })
}

/// For every `(i, j)` in row-major order: `res_ij = lhs_i (.) rhs_j` when `first`, otherwise `prod = lhs_i (.) rhs_j`
/// and `res_ij += prod`.
#[allow(clippy::too_many_arguments)]
fn accumulate<BE, M, DR, L, Rh, P, K>(
    module: &M,
    res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
    lhs: &[&L],
    rhs: &[&Rh],
    first: bool,
    prod: &mut P,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSMulOps<BE> + CKKSAddOps<BE> + ?Sized,
    DR: Data,
    CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
    L: GLWEToBackendRef<BE> + CKKSCtBounds,
    Rh: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    K: GetTensorKey<BE>,
{
    for (i, l) in lhs.iter().enumerate() {
        for (j, r) in rhs.iter().enumerate() {
            let dst = res.at_mut(i, j);
            if first {
                module.ckks_mul_into(dst, *l, *r, keys, scratch)?;
            } else {
                module.ckks_mul_into(prod, *l, *r, keys, scratch)?;
                module.ckks_add_assign(dst, &*prod, scratch)?;
            }
        }
    }
    Ok(())
}
