use poulpy_ckks::{CKKSResult as Result, api::CKKSLinearTransformationOps};
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{baby_caches, lt_streamed_sum_tmp_bytes, lt_tmp_bytes, shared_giants};
use crate::layouts::{
    BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef,
    matvec_operand::output_label,
    validate::{KeyLayouts, assert_scratch, ensure_rotation, matvec_pt_uses, validate_blas_matvec_pt},
};

/// Reference compositions of [`crate::api::BlasMatVecOps`]. Each runs its validator,
/// [`crate::layouts::validate_blas_matvec_pt`], before any write, so its errors leave `res` untouched, then asserts its
/// scratch and returns poulpy-ckks's budget errors before any write. It runs ciphertext after ciphertext on one
/// scratch, and the first error stops it.
pub trait BlasMatVecReference<BE: Backend> {
    /// The baby steps of one input ciphertext and one summed linear transformation from ciphertexts of `a`'s layout into
    /// one of `res`'s, by `w`'s diagonals: resident, or streamed for an unprepared operand.
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
}

impl<BE: Backend> BlasMatVecReference<BE> for Module<BE> {
    fn blas_matvec_pt_tmp_bytes_reference<R, W, A, K>(&self, res: &R, _w: &W, a: &A, keys: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos,
    {
        let (res, a) = (res.ct_layout(), a.ct_layout());
        if W::STREAMED {
            lt_streamed_sum_tmp_bytes(self, &res, &a, keys)
        } else {
            lt_tmp_bytes(self, &res, &a, keys, true)
        }
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
        let bytes = used.max_bytes(|key| self.blas_matvec_pt_tmp_bytes_reference(&*res, w, a, key));
        assert_scratch(op, bytes, scratch);
        let (cts, maps) = (a.cts(), w.blocks());
        let inputs = cts.len();
        // Every map shares one schedule, so the first one's baby steps serve them all.
        let mut caches = baby_caches(self, &[&maps[0]], &cts[0], inputs);
        let outputs = vec![res.cts_mut().iter_mut().collect()];
        shared_giants(
            self,
            &mut caches,
            outputs,
            |_, i| &cts[i],
            |_, o, i| &maps[o * inputs + i],
            output_label(w),
            keys,
            scratch,
        )
    }
}
