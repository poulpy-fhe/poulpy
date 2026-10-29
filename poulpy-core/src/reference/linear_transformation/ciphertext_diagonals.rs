//! The products of a linear transformation whose diagonals are of the rank of its source.
//!
//! A giant step's product is the dot product of its baby rotations by its diagonals, summed over the terms of a sum
//! and relinearized once, left in DFT before the normalization that `glwe_mul_prepared_relinearize_sum` would end
//! with, so the giant-step loop rotates and sums it with one final normalization. Where the one-pass product does not
//! apply, each pair is tensored, the tensors summed, normalized and relinearized, then transformed.

use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftApply, VecZnxGadgetDecompose,
        VecZnxGadgetDecomposeTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{Backend, Module, PrepareHint, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    GLWEPreparedTensoring, GLWETensoring, ScratchArenaTakeCore,
    api::{GLWEAdd, GLWEBytesOf, GLWENormalize},
    gadget::cnv_offset_to_limb_offset,
    glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        DiagonalProd, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendRef, GetTensorKey, LWEInfos, LinearTransformation,
        MulRightOperand, TorusPrecision,
        prepared::{
            GGLWEPreparedToBackendRef, GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, LinearTransformationBabySteps,
        },
    },
    oep::{GGLWEProductDigitsStridedImpl, GLWETensoringImpl},
    reference::{
        keyswitching::GGLWEProductReference,
        linear_transformation::prepared_giants::GiantProducts,
        operations::{GLWETensoringReference, OnePassShape, one_pass_shape},
    },
};

/// The products of the transforms `lts` from the baby caches `babies`, into a result of the layout `res`: giant step
/// `g`'s is the dot product of the baby rotations of every term by its diagonals of giant step `g`.
pub(super) struct CiphertextProducts<'a, BE: Backend, P> {
    babies: &'a [&'a LinearTransformationBabySteps<BE>],
    lts: &'a [&'a LinearTransformation<P>],
    diagonal: &'a P,
    tsk: GLWETensorKeyPreparedBackendRef<'a, BE>,
    res: GLWELayout,
    tensor: GLWELayout,
    cnv_offset: usize,
    /// The one-pass product's shape, or `None` for the tensor path.
    one_pass: Option<OnePassShape>,
}

impl<'a, BE, P> CiphertextProducts<'a, BE, P>
where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    pub(super) fn new<T: GetTensorKey<BE>>(
        babies: &'a [&'a LinearTransformationBabySteps<BE>],
        lts: &'a [&'a LinearTransformation<P>],
        tsk_source: &'a T,
        res: GLWELayout,
        cnv_offset: usize,
    ) -> Self {
        let (lhs, diagonal) = (babies[0], lts[0].first_diagonal_plaintext().expect("checked non-empty"));
        let cols = res.rank.as_usize() + 1;
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
                    assert!(
                        p.mul_right_operand().is_some(),
                        "a diagonal of the rank of res gives no ciphertext operand"
                    );
                }
            }
        }
        let tensor = GLWELayout {
            k: lhs.k().max(diagonal.k()),
            ..res
        };
        let tsk = tsk_source.get_tensor_key(tensor.k).unwrap_or_else(|e| panic!("{e}"));
        let one_pass =
            (res.rank.as_usize() == 1 && cnv_offset >= res.base2k.as_usize() && tsk.base2k() == res.base2k).then(|| {
                one_pass_shape(
                    &res,
                    &tensor,
                    &tsk,
                    lhs.size(),
                    diagonal.size(),
                    lhs.k(),
                    diagonal.k(),
                    cnv_offset,
                )
            });
        Self {
            babies,
            lts,
            diagonal,
            tsk,
            res,
            tensor,
            cnv_offset,
            one_pass,
        }
    }

    fn summed(&self, g: usize) -> bool {
        self.babies.len() > 1 || self.lts[0].giant_steps[g].diagonals.len() > 1
    }
}

impl<BE, P> GiantProducts<BE> for CiphertextProducts<'_, BE, P>
where
    BE: GLWETensoringImpl + GGLWEProductDigitsStridedImpl,
    Module<BE>: GLWETensoringReference<BE>
        + GLWEPreparedTensoring<BE>
        + GLWETensoring<BE>
        + GLWEBytesOf<BE>
        + GLWEAdd<BE>
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
{
    fn size(&self) -> usize {
        self.one_pass
            .as_ref()
            .map_or(self.res.size(), |shape| shape.dft_out_size(self.res.size()))
    }

    /// The tensor paths normalize each product, the low bits of the offset included.
    fn cnv_offset_lo(&self) -> i64 {
        match &self.one_pass {
            Some(shape) if shape.one_pass => cnv_offset_to_limb_offset(self.cnv_offset, self.res.base2k.as_usize()).1,
            _ => 0,
        }
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
        let (lhs, diagonal) = (self.babies[0], self.diagonal);
        let (a_size, b_size) = (lhs.size(), diagonal.size());
        for (prod, g) in prods.iter_mut().zip(steps) {
            let pairs = self.babies.iter().zip(self.lts).flat_map(|(lhs, lt)| {
                lt.giant_steps[g].diagonals.iter().map(move |d| {
                    let right = d.plaintext.mul_right_operand().expect("checked ciphertext diagonals");
                    (lhs.baby_step(d.baby), right)
                })
            });
            let Some(shape) = &self.one_pass else {
                let (n, cols, base2k) = (self.res.n.as_usize(), self.res.rank.as_usize() + 1, self.res.base2k);
                let (mut tmp, scratch) = scratch.borrow().take_glwe_scratch(&self.res);
                let (mut acc, scratch) = scratch.take_glwe_tensor_scratch(&self.tensor);
                let (mut term, scratch) = scratch.take_glwe_tensor_scratch(&self.tensor);
                let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right_scratch(n, cols, b_size, PrepareHint::Reuse);
                for (i, (a, b)) in pairs.enumerate() {
                    let out = if i == 0 { &mut acc } else { &mut term };
                    match b {
                        MulRightOperand::Prepared(b) => {
                            module.glwe_tensor_apply_prepared(self.cnv_offset, out, a, a_size, b, b_size, base2k, &mut scratch)
                        }
                        MulRightOperand::Ciphertext(b) => {
                            glwe_prepare_right(module, &mut b_prep, b, diagonal.k().as_usize(), &mut scratch);
                            module.glwe_tensor_apply_prepared(
                                self.cnv_offset,
                                out,
                                a,
                                a_size,
                                &b_prep,
                                b_size,
                                base2k,
                                &mut scratch,
                            )
                        }
                    }
                    if i > 0 {
                        module.glwe_add_assign(&mut acc, &term);
                    }
                }
                // A tensor view reports canonical, so it must hold canonical digits before the relinearization's DFT.
                if self.summed(g) {
                    module.glwe_normalize_assign(&mut acc, &mut scratch);
                }
                module.glwe_tensor_relinearize(&mut tmp, &acc, &Resolved(&self.tsk), &mut scratch);
                let tmp = tmp.to_backend_ref();
                for col in 0..cols {
                    module.vec_znx_dft_apply(1, 0, prod, col, &tmp.data, col);
                }
                continue;
            };
            BE::glwe_mul_prepared_relinearize_sum_dft(
                module,
                self.cnv_offset,
                prod,
                &self.res,
                shape,
                pairs,
                a_size,
                lhs.k(),
                b_size,
                diagonal.k(),
                self.summed(g),
                &Resolved(&self.tsk),
                scratch,
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
/// from babies of `a_size` limbs and diagonals of `b_size` limbs through a tensor of width `tensor_k`: the larger of
/// the backend's one-pass dot product left in DFT and the tensor path's result and sum of two tensors with a right
/// operand prepared in the call.
pub(super) fn ciphertext_products_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    a_size: usize,
    b_size: usize,
    tensor_k: TorusPrecision,
    tsk: &K,
) -> usize
where
    BE: GLWETensoringImpl,
    Module<BE>:
        GLWEPreparedTensoring<BE> + GLWETensoring<BE> + GLWEBytesOf<BE> + GLWENormalize<BE> + Convolution<BE> + CnvPVecBytesOf,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let one_pass = BE::glwe_mul_prepared_relinearize_sum_dft_tmp_bytes(module, res, a_size, b_size, tensor_k, tsk);
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    let tensor = GLWELayout {
        k: tensor_k,
        ..res.glwe_layout()
    };
    let right = GLWELayout {
        k: TorusPrecision((b_size * res.base2k().as_usize()) as u32),
        ..tensor
    };
    let tensor_path = BE::scratch_aligned(module.glwe_bytes_of_from_infos(res))
        + 2 * BE::scratch_aligned(module.glwe_tensor_bytes_of_from_infos(&tensor))
        + BE::scratch_aligned(module.bytes_of_cnv_pvec_right(n, cols, b_size, PrepareHint::Reuse))
        + glwe_prepare_right_tmp_bytes(module, b_size, &right)
            .max(module.glwe_tensor_apply_prepared_tmp_bytes(&tensor, a_size, b_size, res.base2k()))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_tensor_relinearize_tmp_bytes(res, &tensor, tsk));
    one_pass.max(tensor_path)
}
