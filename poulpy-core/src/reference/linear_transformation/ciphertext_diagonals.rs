//! Linear transformations whose diagonals are ciphertexts prepared as right operands.
//!
//! A giant step's product is the one-pass relinearized sum of its baby rotations by its diagonals, left in DFT
//! before the normalization that `glwe_mul_prepared_relinearize_sum` would end with, so the giant-step loop of the
//! plaintext transformation rotates and sums it with one final normalization.

#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxAutomorphismAssign, VecZnxBigAddAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxDftAutomorphism, VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero,
        VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize,
        VecZnxIdftNormalizeTmpBytes, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, CnvPVecRToBackendRef, GaloisElement, Module, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, ScratchArenaTakeCore,
    api::{GLWEBytesOf, GLWENormalize},
    gadget::cnv_offset_to_limb_offset,
    layouts::{
        Base2K, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GetAutomorphismKey, GetTensorKey, LWEInfos,
        LinearTransformation, LinearTransformationPlan, ModuleCoreAlloc, TorusPrecision, assert_ct_linear_transformation_into,
        prepared::{GLWETensorKeyPreparedBackendRef, LinearTransformationBabySteps},
    },
    oep::{GGLWEProductDigitsStridedImpl, GLWETensoringImpl, LinearTransformationImpl},
    reference::{
        keyswitching::{GGLWEProductReference, GLWEKeyswitchInternal},
        linear_transformation::{
            eval::giant_routes_tmp_bytes,
            prepared_giants::{GiantProducts, giant_tree_tmp_bytes, glwe_eval_giant_products},
        },
        operations::{
            GLWETensoringReference, MulLeft, MulRest, MulRight, NoOperand, OnePassShape, glwe_mul_relinearize_one_pass,
            one_pass_shape, one_pass_tmp_bytes,
        },
    },
};

/// The products of a transform of ciphertext diagonals into a result of the layout `res`: giant step `g`'s is the
/// one-pass relinearized sum of the baby rotations of `lhs` by the diagonals of giant step `g` of `rhs`.
struct CiphertextProducts<'a, BE: Backend, D, T> {
    lhs: &'a LinearTransformationBabySteps<BE>,
    lhs_k: TorusPrecision,
    rhs: &'a LinearTransformation<D>,
    tsk_source: &'a T,
    tsk: GLWETensorKeyPreparedBackendRef<'a, BE>,
    res: GLWELayout,
    cnv_offset: usize,
    shape: OnePassShape,
}

impl<BE, D, T> GiantProducts<BE> for CiphertextProducts<'_, BE, D, T>
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
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
    D: CnvPVecRToBackendRef<BE> + GLWEInfos,
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
        let diagonal = self.rhs.first_diagonal_plaintext().expect("checked non-empty");
        let tensor_k = self.lhs_k.max(diagonal.k());
        ciphertext_products_tmp_bytes(module, &self.res, self.lhs.size(), diagonal.size(), tensor_k, &self.tsk)
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
        let diagonals = &self.rhs.giant_steps[g].diagonals;
        let lefts: Vec<_> = diagonals.iter().map(|d| self.lhs.baby_step(d.baby)).collect();
        let rights: Vec<&D> = diagonals.iter().map(|d| &d.plaintext).collect();
        let (mut tmp, mut scratch) = scratch.borrow().take_glwe_scratch(&self.res);
        glwe_mul_relinearize_one_pass(
            module,
            self.cnv_offset,
            &mut tmp,
            self.res.k,
            MulLeft::<NoOperand<BE>, _>::Prepared {
                prep: lefts[0],
                size: self.lhs.size(),
                k: self.lhs_k,
            },
            MulRight::<NoOperand<BE>, _>::Prepared {
                prep: rights[0],
                size: rights[0].size(),
                k: rights[0].k(),
            },
            MulRest::Prepared(&lefts[1..], &rights[1..]),
            self.tsk_source,
            &mut scratch,
            Some(prod),
        );
    }
}

/// Bytes the product of a giant step of [`CiphertextProducts`] takes beside it, into a result of the layout `res`
/// from babies of `a_size` limbs and diagonals of `b_size` limbs through a tensor of width `tensor_k`: the result the
/// tensor path normalizes, then the one-pass product of the prepared pairs into DFT.
fn ciphertext_products_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    a_size: usize,
    b_size: usize,
    tensor_k: TorusPrecision,
    tsk: &K,
) -> usize
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEBytesOf<BE>
        + CnvPVecBytesOf
        + Convolution<BE>
        + GLWENormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + GGLWEProductReference<BE>
        + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(res))
        + one_pass_tmp_bytes(module, res, a_size, b_size, tensor_k, tsk, true, true, true)
}

/// Reference impl: scratch bytes for [`glwe_eval_ct_linear_transformation_into_reference`], `res` the destination,
/// `a` the source, `b` a diagonal, `key` the widest giant key and `tsk` the tensor key at the larger of the source's
/// and the diagonals' widths. The products are sized at the larger of the tensor key and `res`.
pub fn glwe_eval_ct_linear_transformation_tmp_bytes_reference<BE, R, A, B, K, T>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    b: &B,
    plan: &LinearTransformationPlan,
    key: &K,
    tsk: &T,
) -> usize
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEBytesOf<BE>
        + ModuleN
        + CnvPVecBytesOf
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GLWENormalize<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + GGLWEProductReference<BE>
        + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    let res_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: res.rank(),
    };
    let products_bytes =
        ciphertext_products_tmp_bytes(module, &res_infos, a.max_size(), b.max_size(), a.max_k().max(b.max_k()), tsk);
    let prod_size = tsk.size().max(res.max_size());
    let cols = res.rank().as_usize() + 1;
    giant_routes_tmp_bytes(module, cols, prod_size, products_bytes, 1, res, key).max(giant_tree_tmp_bytes(
        module,
        res,
        prod_size,
        products_bytes,
        plan,
        key,
    ))
}

/// Reference impl: `res = M(a)` for a transform `rhs` whose diagonals are rank-1 ciphertexts prepared as right
/// operands, from the baby cache `lhs` of a source of width `lhs_k`.
///
/// Runs [`assert_ct_linear_transformation_into`], looks the tensor key up at the larger of `lhs_k` and the diagonals'
/// width, then the giant-step loop of the plaintext transformation over the one-pass products of each giant step.
pub fn glwe_eval_ct_linear_transformation_into_reference<BE, R, D, T, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    lhs_k: TorusPrecision,
    rhs: &LinearTransformation<D>,
    tsk: &T,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl + LinearTransformationImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEBytesOf<BE>
        + GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + ModuleN
        + GGLWEProductReference<BE>
        + VecZnxAutomorphismAssign<BE>
        + GLWEKeyswitchInternal<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftAlloc<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEMulPlain<BE>
        + GaloisElement
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxDftAutomorphismAddWithPlanTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    D: CnvPVecRToBackendRef<BE> + GLWEInfos,
    T: GetTensorKey<BE>,
    H: GetAutomorphismKey<BE>,
{
    assert_ct_linear_transformation_into(res, lhs, lhs_k, rhs);
    let diagonal = rhs.first_diagonal_plaintext().expect("checked non-empty");
    let res_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: res.rank(),
    };
    let tensor_infos = GLWELayout {
        k: lhs_k.max(diagonal.k()),
        ..res_infos
    };
    let tsk_ref = tsk.get_tensor_key(tensor_infos.k).unwrap_or_else(|e| panic!("{e}"));
    let shape = one_pass_shape(
        &res_infos,
        &tensor_infos,
        &tsk_ref,
        lhs.size(),
        diagonal.size(),
        lhs_k,
        diagonal.k(),
        cnv_offset,
    );
    let products = CiphertextProducts {
        lhs,
        lhs_k,
        rhs,
        tsk_source: tsk,
        tsk: tsk_ref,
        res: res_infos,
        cnv_offset,
        shape,
    };
    glwe_eval_giant_products(module, res, rhs, &products, keys, scratch);
    res.set_canonical(true);
}
