//! The products of a linear transformation whose diagonals are of the rank of its source.
//!
//! A giant step's product is the dot product of its baby rotations by its diagonals, summed over the terms of a sum
//! and relinearized once: the baby-step kernel sums the tensor columns of every pair in DFT, then the one-pass
//! relinearization brings back to digits the column it key-switches only, leaving the product in DFT before the
//! normalization that `glwe_mul_prepared_relinearize_sum` would end with, so the giant-step loop rotates and sums it
//! with one final normalization. Where the one-pass relinearization does not apply, the columns are normalized into a
//! tensor, relinearized, then transformed.

use crate::layouts::BabyStepOperand;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftBytesOf, VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes,
        VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    ScratchArenaTakeCore,
    api::GLWEBytesOf,
    gadget::{cnv_offset_to_limb_offset, normalize_input_limb_bound_with_offset},
    layouts::{
        DiagonalProd, GGLWEInfos, GLWEInfos, GLWELayout, GetTensorKey, LWEInfos, LinearTransformation, TorusPrecision,
        prepared::{
            GGLWEPreparedToBackendRef, GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, LinearTransformationBabySteps,
        },
    },
    oep::{GGLWEProductDigitsStridedImpl, GLWETensoringImpl},
    reference::{
        keyswitching::GGLWEProductReference,
        linear_transformation::giant_steps::GiantProducts,
        operations::{
            GLWETensoringReference, OnePassShape, glwe_relinearize_columns, glwe_relinearize_columns_tmp_bytes, one_pass_shape,
        },
    },
};

/// The products of the transforms `lts` from the baby caches `babies`, into a result of the layout `res`: giant step
/// `g`'s is the dot product of the baby rotations of every term by its diagonals of giant step `g`.
pub(super) struct CiphertextProducts<'a, BE: Backend, P, L> {
    babies: &'a [&'a LinearTransformationBabySteps<BE, L>],
    lts: &'a [&'a LinearTransformation<P>],
    diagonal: &'a P,
    tsk: GLWETensorKeyPreparedBackendRef<'a, BE>,
    res: GLWELayout,
    tensor: GLWELayout,
    cnv_offset: usize,
    shape: OnePassShape,
}

impl<'a, BE, P, L> CiphertextProducts<'a, BE, P, L>
where
    BE: Backend,
    P: DiagonalProd<BE>,
    L: BabyStepOperand<BE>,
{
    pub(super) fn new<T: GetTensorKey<BE>>(
        babies: &'a [&'a LinearTransformationBabySteps<BE, L>],
        lts: &'a [&'a LinearTransformation<P>],
        tsk_source: &'a T,
        res: GLWELayout,
        cnv_offset: usize,
    ) -> Self {
        let (lhs, diagonal) = (babies[0], lts[0].first_diagonal_plaintext().expect("checked non-empty"));
        let cols = res.rank.as_usize() + 1;
        // 1. Checks, once per evaluation and before any write: every baby cache and diagonal at the degree, base2k and
        //    rank of `res` and at the first one's width.
        assert!(
            babies.iter().all(|b| b.cols() == cols && b.k() == lhs.k()),
            "every baby cache must hold the rank of res and the first cache's width"
        );
        for (&b, lt) in babies.iter().zip(lts) {
            for gs in &lt.giant_steps {
                for d in &gs.diagonals {
                    assert!(
                        b.baby_step(d.baby).n() == res.n.as_usize(),
                        "a baby rotation is not at the degree of res"
                    );
                    let p = &d.plaintext;
                    assert!(
                        (p.n(), p.base2k(), p.rank(), p.k(), p.size())
                            == (res.n, res.base2k, res.rank, diagonal.k(), diagonal.size()),
                        "a ciphertext diagonal is not at the degree, base2k and rank of res and the first diagonal's width"
                    );
                }
            }
        }
        // 2. The tensor at the wider operand's width, and its key looked up once (`GetTensorKey::get_tensor_key`).
        let tensor = GLWELayout {
            k: lhs.k().max(diagonal.k()),
            ..res
        };
        let tsk = tsk_source.get_tensor_key(tensor.k).unwrap_or_else(|e| panic!("{e}"));
        // 3. The route and the column limbs (`one_pass_shape`): one-pass at rank 1, `cnv_offset >= base2k` and `tsk` at
        //    the base2k of `res` while it keeps the two-step noise, else a tensor.
        let shape = one_pass_shape(
            &res,
            &tensor,
            &tsk,
            lhs.size(),
            diagonal.size(),
            lhs.k(),
            diagonal.k(),
            cnv_offset,
        );
        Self {
            babies,
            lts,
            diagonal,
            tsk,
            res,
            tensor,
            cnv_offset,
            shape,
        }
    }
}

impl<BE, P, L> GiantProducts<BE> for CiphertextProducts<'_, BE, P, L>
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEBytesOf<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + GGLWEProductReference<BE>
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes,
    P: DiagonalProd<BE>,
    L: BabyStepOperand<BE>,
{
    fn size(&self) -> usize {
        self.shape.dft_out_size(self.res.size())
    }

    /// The tensor route normalizes each product, the low bits of the offset included.
    fn cnv_offset_lo(&self) -> i64 {
        if self.shape.one_pass {
            cnv_offset_to_limb_offset(self.cnv_offset, self.res.base2k.as_usize()).1
        } else {
            0
        }
    }

    /// The one-pass route leaves the low bits of the offset below `res_k` to the final normalization.
    fn switch_k(&self, res_k: usize) -> usize {
        res_k + self.cnv_offset_lo() as usize
    }

    fn batch(&self, _module: &Module<BE>) -> usize {
        1
    }

    fn tmp_bytes(&self, module: &Module<BE>, _n: usize) -> usize {
        ciphertext_products_tmp_bytes(
            module,
            &self.res,
            self.babies[0].size(),
            self.diagonal.size(),
            self.tensor.k,
            &self.tsk,
        )
    }

    fn products(
        &self,
        module: &Module<BE>,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        steps: core::ops::Range<usize>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let (n, cols) = (self.res.n.as_usize(), self.res.rank.as_usize() + 1);
        let (cnv_offset_hi, _) = cnv_offset_to_limb_offset(self.cnv_offset, self.res.base2k.as_usize());
        for (prod, g) in prods.iter_mut().zip(steps) {
            // 1. The pairs of giant step `g` over every term: a baby rotation and a diagonal, prepared or a GLWE view
            //    (`DiagonalProd::operand`).
            let pairs = self.babies.iter().zip(self.lts).flat_map(|(lhs, lt)| {
                lt.giant_steps[g]
                    .diagonals
                    .iter()
                    .map(move |d| (lhs.baby_step(d.baby), d.plaintext.operand()))
            });
            // 2. The tensor route's result, and the tensor columns in DFT: `diag[i] = sum a[i] * b[i]` and `cross`, in row
            //    order of `i < j`, `sum a[i] * b[j] + a[j] * b[i]`.
            let (mut tmp, scratch) = scratch.borrow().take_glwe_scratch(&self.res);
            let (mut diag, scratch) = scratch.take_vec_znx_dft_scratch(n, cols, self.shape.dft_size);
            let (mut cross, mut scratch) = scratch.take_vec_znx_dft_scratch(n, cols * (cols - 1) / 2, self.shape.dft_size);
            // 3. Baby-step kernel, `GLWETensoringImpl::glwe_tensor_apply_prepared_sum_dft`. Reference, per pair:
            //    `cnv_prepare_right` of a GLWE view, `cnv_apply_dft` (first pair) or `cnv_apply_dft_add` into `diag`,
            //    `cnv_pairwise_apply_dft` into `cross` (then `vec_znx_dft_add_assign` after the first pair); once:
            //    `vec_znx_dft_sub_assign` of `diag` from `cross`.
            BE::glwe_tensor_apply_prepared_sum_dft(
                module,
                cnv_offset_hi,
                &mut diag,
                &mut cross,
                pairs,
                self.diagonal.size(),
                self.diagonal.k(),
                &mut scratch,
            );
            // 4. Relinearization into `prod`, left in DFT (`glwe_relinearize_columns`).
            //    One-pass: `vec_znx_idft_normalize` of `d2 = diag[1]` alone, `gadget_dft_apply`, then
            //    `GGLWEProductDigitsStridedImpl::gglwe_product_fused` of `tsk` with `d0 = diag[0]` and `d1 = cross[0]`
            //    added in DFT, its low `cnv_offset` bits left to the final normalization.
            //    Tensor: `vec_znx_idft_normalize` of every column into a tensor, `GLWETensoringImpl::glwe_tensor_relinearize`
            //    into `tmp`, then `vec_znx_dft_apply` into `prod`.
            glwe_relinearize_columns(
                module,
                self.cnv_offset,
                &mut tmp,
                self.res.k,
                &self.tensor,
                &mut diag,
                &mut cross,
                &self.shape,
                &Resolved(&self.tsk),
                &mut scratch,
                Some(prod),
            );
        }
    }
}

/// The tensor key looked up once, the source of every product.
struct Resolved<'k, 'a, BE: Backend + 'a>(&'k GLWETensorKeyPreparedBackendRef<'a, BE>);

impl<'a, BE: Backend + 'a> GetTensorKey<BE> for Resolved<'_, 'a, BE> {
    fn lookup_tensor_key(&self, _k: TorusPrecision) -> crate::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        Ok(GLWETensorKeyPrepared(GGLWEPreparedToBackendRef::to_backend_ref(&self.0)))
    }
}

/// Bytes the product of a giant step of [`CiphertextProducts`] takes beside it, into a result of the layout `res`
/// from babies of `a_size` limbs and diagonals of `b_size` limbs through a tensor of width `tensor_k`: the result, the
/// tensor columns in DFT and the larger of the baby-step kernel and the relinearization of either route.
pub(super) fn ciphertext_products_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    a_size: usize,
    b_size: usize,
    tensor_k: TorusPrecision,
    tsk: &K,
) -> usize
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: ModuleN + GLWEBytesOf<BE> + VecZnxDftBytesOf + VecZnxIdftNormalizeTmpBytes + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let (n, cols, base2k) = (res.n().as_usize(), res.rank().as_usize() + 1, res.base2k().as_usize());
    let tensor = GLWELayout {
        k: tensor_k,
        ..res.glwe_layout()
    };
    // The widest columns, at any offset.
    let dft_size = normalize_input_limb_bound_with_offset(a_size + b_size, tensor.size(), base2k, base2k, base2k as i64 - 1);
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(res))
        + BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, dft_size))
        + BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols * (cols - 1) / 2, dft_size))
        + BE::glwe_tensor_apply_prepared_sum_dft_tmp_bytes(module, &tensor, a_size, b_size).max(
            glwe_relinearize_columns_tmp_bytes(module, res, &tensor, dft_size, tsk, true, true),
        )
}
