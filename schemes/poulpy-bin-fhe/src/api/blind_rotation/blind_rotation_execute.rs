use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos, BlindRotationKeyPrepared, LookupTable};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Public execution and scratch queries dispatched through the selected backend.
pub trait BlindRotationExecute<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_execute`][Self::blind_rotation_execute].
    ///
    /// `block_size` is the number of LWE coefficients processed per GGSW
    /// product (1 for standard CGGI, > 1 for block-binary).
    /// `extension_factor` is the number of LUT polynomials (must be a power
    /// of two; 1 for the classical single-polynomial LUT).
    fn blind_rotation_execute_tmp_bytes<G, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: BlindRotationKeyInfos;

    /// Evaluates the lookup table `lut` at the index encrypted in `lwe`,
    /// writing the result GLWE ciphertext into `res`.
    ///
    /// After a successful call, decrypting `res` and reading coefficient 0
    /// yields `lut(dec(lwe))` (up to rounding noise from the decomposition).
    ///
    /// # Panics
    ///
    /// Panics if dimension mismatches are detected between `res`,
    /// `lwe`, `lut`, and `brk`.
    fn blind_rotation_execute<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
        brk: &BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWEToBackendRef<BE> + LWEInfos;
}
