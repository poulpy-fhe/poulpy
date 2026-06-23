use anyhow::{Result, ensure};
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

/// Default (backend-generic) implementation of the CKKS bootstrapping
/// primitives.
///
/// The only genuinely new primitive bootstrapping needs is the modulus raise
/// ([`Self::ckks_mod_up_into_default`]); the rest of the pipeline
/// (CoeffsToSlots, EvalMod, SlotsToCoeffs) is composed by the caller from the
/// existing op traits. See [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps)
/// for the documented semantics.
#[doc(hidden)]
pub trait CKKSBootstrappingOpsDefault<BE: Backend> {
    fn ckks_mod_up_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_mod_up_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        // `dst` is the (freshly-allocated) widened target: ModUp raises the modulus
        // into its full allocated capacity, so the widening width is `dst.max_k()`.
        // `dst.k()` is meta-derived and is `0` on a fresh ciphertext, which would
        // spuriously fail the "must widen" check below.
        let k_large: usize = dst.max_k().as_usize();
        let k_small: usize = src.k().as_usize();

        ensure!(
            k_large >= k_small,
            "ckks_mod_up: dst.k ({k_large}) < src.k ({k_small}); ModUp must widen, not shrink, the modulus"
        );

        // MSB-align `src` into the wider `dst`: the value occupies the top
        // `k_small` bits and the freshly-introduced low-order limbs are zero.
        self.glwe_copy(dst, src);

        // Shift the digits down to their natural integer magnitude. This is the
        // modulus raise: the raised-from modulus `q = 2^k_small` becomes an
        // explicit, un-reduced multiple `I(X)·q` living in the `[0, 2^k_large)`
        // window, which EvalMod subsequently removes. (A zero shift — when the
        // input already fills the storage — is a no-op.)
        self.glwe_rsh(k_large - k_small, dst, scratch);

        // `log_delta` is unchanged (the integer `Δ·m` keeps its scale); the
        // remaining headroom now spans the full raised modulus, so the torus
        // width `k` becomes `k_large` and `log_budget = k_large - log_delta`.
        dst.set_meta(CKKSMeta {
            log_delta: src.log_delta(),
            log_sparsity: src.log_sparsity(),
        });
        dst.set_k(k_large.into());

        Ok(())
    }
}
