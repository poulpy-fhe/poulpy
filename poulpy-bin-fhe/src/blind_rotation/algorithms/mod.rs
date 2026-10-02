mod cggi;

pub use cggi::*;

use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef, ModuleCoreAlloc};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::blind_rotation::{BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, LookupTable};

/// Marker trait for blind-rotation algorithm variants.
///
/// Implementors act as phantom types that bind a specific algorithm identity
/// to key and execution types.  This prevents accidental cross-algorithm key
/// usage at the type level.  Currently the only implementation is [`CGGI`].
pub trait BlindRotationAlgo: Sync {
    /// Allocates a zero-filled [`BlindRotationKey`] from a dimension descriptor.
    fn alloc_key<M, A>(module: &M, infos: &A) -> BlindRotationKey<M::OwnedBuf, Self, M::ZnxWord>
    where
        M: ModuleCoreAlloc + ModuleN,
        A: BlindRotationKeyInfos,
        Self: Sized;
}

/// Trait for executing the blind rotation algorithm.
///
/// Dispatched by `Module<BE>` through [`crate::oep::BlindRotationExecuteImpl`].  Callers should prefer the convenience
/// method on [`BlindRotationKeyPrepared::execute`] rather than calling these
/// methods directly.
pub use crate::api::BlindRotationExecute;

impl<BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    /// Performs blind rotation using `self` as the bootstrapping key.
    ///
    /// Convenience wrapper around [`BlindRotationExecute::blind_rotation_execute`].
    pub fn execute<R, L, M>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: BlindRotationExecute<BRA, BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.blind_rotation_execute(res, lwe, lut, self, scratch);
    }
}

impl<BE: Backend, BRA: BlindRotationAlgo> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`BlindRotationKeyPrepared::execute`].
    ///
    /// See [`BlindRotationExecute::blind_rotation_execute_tmp_bytes`].
    pub fn execute_tmp_bytes<A, B, M>(
        module: &M,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &A,
        brk_infos: &B,
    ) -> usize
    where
        A: GLWEInfos,
        B: BlindRotationKeyInfos,
        M: BlindRotationExecute<BRA, BE>,
    {
        module.blind_rotation_execute_tmp_bytes(block_size, extension_factor, glwe_infos, brk_infos)
    }
}

/// Compatibility entry point for the callable canonical host-staging composition.
///
/// This function bypasses backend selection. Generic blind rotation calls
/// [`crate::api::BlindRotationModSwitch`] so a backend can replace the staging
/// boundary. See [`crate::reference::blind_rotation::mod_switch_2n_ref`] for
/// normalized signed-limb inputs and rounding semantics.
pub use crate::reference::blind_rotation::mod_switch_2n_ref as mod_switch_2n;
