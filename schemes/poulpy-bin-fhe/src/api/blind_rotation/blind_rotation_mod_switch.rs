use crate::blind_rotation::LookUpTableRotationDirection;
use poulpy_core::layouts::{LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::Backend;

/// Explicit host staging boundary for blind-rotation exponents.
/// The result contains the switched body followed by the switched mask.
pub trait BlindRotationModSwitch<BE: Backend<ZnxWord = i64>> {
    /// Switches the LWE body and mask to rotation indices modulo `modulus`.
    /// `res` must hold the body followed by every mask coefficient.
    fn blind_rotation_mod_switch<L: LWEToBackendRef<BE> + LWEInfos>(
        &self,
        modulus: usize,
        res: &mut [i64],
        lwe: &L,
        direction: LookUpTableRotationDirection,
    );
}
