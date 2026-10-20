use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::ReplicatedConversion};

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
