use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{BootstrappingKeys, ReplicatedBootstrapping, ReplicatedConversion},
};

/// Conversions between CKKS and a replicated slot layout, sharing keys.
///
/// # Metadata
///
/// `ckks_to_replicated_into` reads a full-layout ciphertext holding `k`
/// values with period `k` and writes them on the data slots of `conv`'s
/// layout, at scale `conv.log_delta`:
///
/// ```text
/// log_budget_out = src.log_budget − log_delta_diag
/// log_delta_out  = conv.log_delta
/// ```
///
/// `ckks_from_replicated_into` reads a ciphertext in `conv`'s layout, with
/// `R | k`, and writes its `k` values with period `k` on a full layout,
/// averaging the replicas:
///
/// ```text
/// log_budget_out = src.log_budget − log_delta_diag
/// log_delta_out  = src.log_delta + extra_bits + log2(N/(2k))
/// ```
///
/// `keys` holds the automorphism keys of
/// [`ReplicatedConversion::rotations`].
pub trait CKKSReplicatedOps<BE: Backend> {
    fn ckks_replicated_conversion_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K, conv: &ReplicatedConversion<BE>) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    fn ckks_to_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    fn ckks_from_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;
}

/// Bootstrapping of a replicated slot layout of `k` values, `R | k`, in
/// `SlotsToCoeffs`-first order.
///
/// 1. The input, at the scale `Δ_b` or at the leveled scale, is brought to
///    `Δ_b` at width [`ReplicatedBootstrapPlan::k_in`](crate::layouts::ReplicatedBootstrapPlan::k_in).
/// 2. The normalized bottom trace averages the `N/(2kR)` copies of the
///    layout's period with shaped automorphisms.
/// 3. `SlotsToCoeffs` gathers the data slots onto the coefficients of the
///    subring of degree `2k`, `extra_bits` under the scale, and rounds plainly
///    to the message ratio `q_b/Δ_b` of the EvalMod plan.
/// 4. ModUp, with sparse-secret encapsulation when `keys` carries it, lifted
///    by `mod_up_extra_bits`.
/// 5. The normalized top trace onto period `k` removes the lifted multiples
///    off the subring, and `CoeffsToSlots` writes the real and imaginary
///    parts of the coefficients onto the data slots, side by side when
///    packed, each rounded with shaping.
/// 6. EvalMod, then the parts are recombined.
/// 7. The result is relabelled at `Δ_b` and rounded with shaping to the
///    leveled scale.
///
/// `dst.k()` is the output width; ModUp raises to
/// [`ReplicatedBootstrapPlan::k_boot`](crate::layouts::ReplicatedBootstrapPlan::k_boot)
/// of it. The keys cover [`ReplicatedBootstrapping::galois_elements`].
pub trait CKKSReplicatedBootstrapOps<BE: Backend> {
    fn ckks_replicated_bootstrap_tmp_bytes<C, I, F, A, T, S>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos;

    fn ckks_replicated_bootstrap_into<Dst, Src, F, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>;
}
