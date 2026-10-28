//! The products of a linear transformation whose diagonals are of the rank of its source.
//!
//! A giant step's product is the dot product of its baby rotations by its diagonals, summed over the terms of a sum
//! and relinearized once, left in DFT before the normalization that `glwe_mul_prepared_relinearize_sum` would end
//! with, so the giant-step loop rotates and sums it with one final normalization.

use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxDftAddAssign, VecZnxDftApply, VecZnxGadgetDecompose,
        VecZnxGadgetDecomposeTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    GLWEPreparedTensoring, ScratchArenaTakeCore,
    api::{GLWEBytesOf, GLWENormalize},
    gadget::cnv_offset_to_limb_offset,
    layouts::{
        Base2K, DiagonalProd, GGLWEInfos, GLWEInfos, GLWELayout, GetTensorKey, LWEInfos, LinearTransformation, MulRightOperand,
        TorusPrecision,
        prepared::{GLWETensorKeyPreparedBackendRef, LinearTransformationBabySteps},
    },
    oep::{GGLWEProductDigitsStridedImpl, GLWETensoringImpl},
    reference::{
        keyswitching::GGLWEProductReference,
        linear_transformation::prepared_giants::GiantProducts,
        operations::{
            GLWETensoringReference, MulLeft, MulRest, MulRight, NoOperand, OnePassShape, glwe_mul_relinearize_one_pass,
            one_pass_shape,
        },
    },
};

/// The products of the transforms `lts` from the baby caches `babies`, into a result of the layout `res`: giant step
/// `g`'s is the dot product of the baby rotations of every term by its diagonals of giant step `g`.
pub(super) struct CiphertextProducts<'a, BE: Backend, P, T> {
    babies: &'a [&'a LinearTransformationBabySteps<BE>],
    lts: &'a [&'a LinearTransformation<P>],
    tsk_source: &'a T,
    tsk: GLWETensorKeyPreparedBackendRef<'a, BE>,
    res: GLWELayout,
    cnv_offset: usize,
    shape: OnePassShape,
}

impl<'a, BE, P, T> CiphertextProducts<'a, BE, P, T>
where
    BE: Backend,
    P: DiagonalProd<BE>,
    T: GetTensorKey<BE>,
{
    /// Looks the tensor key up at the larger of the sources' and the diagonals' widths.
    pub(super) fn new(
        babies: &'a [&'a LinearTransformationBabySteps<BE>],
        lts: &'a [&'a LinearTransformation<P>],
        tsk_source: &'a T,
        res: GLWELayout,
        cnv_offset: usize,
    ) -> Self {
        let (lhs, diagonal) = (babies[0], lts[0].first_diagonal_plaintext().expect("checked non-empty"));
        let tensor_infos = GLWELayout {
            k: lhs.k().max(diagonal.k()),
            ..res
        };
        let tsk = tsk_source.get_tensor_key(tensor_infos.k).unwrap_or_else(|e| panic!("{e}"));
        let shape = one_pass_shape(
            &res,
            &tensor_infos,
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
            tsk_source,
            tsk,
            res,
            cnv_offset,
            shape,
        }
    }

    fn diagonal(&self) -> &P {
        self.lts[0].first_diagonal_plaintext().expect("checked non-empty")
    }
}

impl<BE, P, T> GiantProducts<BE> for CiphertextProducts<'_, BE, P, T>
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEPreparedTensoring<BE>
        + GLWEBytesOf<BE>
        + ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + GLWENormalize<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftAddAssign<BE>
        + GGLWEProductReference<BE>
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes,
    P: DiagonalProd<BE>,
    T: GetTensorKey<BE>,
{
    fn base2k(&self) -> Base2K {
        self.res.base2k
    }

    fn size(&self) -> usize {
        self.shape.dft_out_size(self.res.size())
    }

    /// The tensor path normalizes each product, the low bits of the offset included.
    fn cnv_offset_lo(&self) -> i64 {
        if self.shape.one_pass {
            cnv_offset_to_limb_offset(self.cnv_offset, self.res.base2k.as_usize()).1
        } else {
            0
        }
    }

    fn batch(&self, _module: &Module<BE>) -> usize {
        1
    }

    fn tmp_bytes(&self, module: &Module<BE>, _n: usize) -> usize {
        let tensor_k = self.babies[0].k().max(self.diagonal().k());
        ciphertext_products_tmp_bytes(
            module,
            &self.res,
            self.babies[0].size(),
            self.diagonal().size(),
            tensor_k,
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
        for (prod, g) in prods.iter_mut().zip(steps) {
            self.product(module, prod, g, scratch);
        }
    }

    fn product(&self, module: &Module<BE>, prod: &mut VecZnxDftBackendMut<'_, BE>, g: usize, scratch: &mut ScratchArena<'_, BE>) {
        let mut pairs = self.babies.iter().zip(self.lts).flat_map(|(lhs, lt)| {
            lt.giant_steps[g]
                .diagonals
                .iter()
                .map(move |d| (lhs.baby_step(d.baby), d.plaintext.mul_right()))
        });
        let summed = self.babies.len() > 1 || self.lts[0].giant_steps[g].diagonals.len() > 1;
        let (left, right) = pairs.next().expect("checked non-empty");
        let (lhs, diagonal) = (self.babies[0], self.diagonal());
        let right = match right {
            MulRightOperand::Prepared(prep) => MulRight::Prepared {
                prep,
                size: diagonal.size(),
                k: diagonal.k(),
            },
            MulRightOperand::Ciphertext(ct) => MulRight::Operand(ct),
        };
        let (mut tmp, mut scratch) = scratch.borrow().take_glwe_scratch(&self.res);
        glwe_mul_relinearize_one_pass(
            module,
            self.cnv_offset,
            &mut tmp,
            self.res.k,
            MulLeft::<NoOperand<BE>, _>::Prepared {
                prep: left,
                size: lhs.size(),
                k: lhs.k(),
            },
            right,
            MulRest::Pairs {
                pairs: &mut pairs,
                summed,
            },
            self.tsk_source,
            &mut scratch,
            Some(prod),
        );
    }
}

/// Bytes the product of a giant step of [`CiphertextProducts`] takes beside it, into a result of the layout `res`
/// from babies of `a_size` limbs and diagonals of `b_size` limbs through a tensor of width `tensor_k`: the result the
/// tensor path normalizes, then the dot product of the pairs, left in DFT by the fused product.
pub(super) fn ciphertext_products_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    a_size: usize,
    b_size: usize,
    tensor_k: TorusPrecision,
    tsk: &K,
) -> usize
where
    BE: GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWEPreparedTensoring<BE> + GLWEBytesOf<BE>,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let size = res.size().max(tsk.size());
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(res))
        + module.glwe_mul_prepared_relinearize_sum_tmp_bytes(res, a_size, b_size, tensor_k, tsk)
        + BE::gglwe_product_fused_tmp_bytes(module, size, size, a_size + b_size, tsk)
}
