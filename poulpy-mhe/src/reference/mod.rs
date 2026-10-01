//! Portable multiparty implementations composed from `poulpy-core` and
//! `poulpy-hal` operations, the default dispatch target of the
//! `impl_mhe_*_reference!` opt-ins. A backend replacing one family may keep
//! calling these for the others.
pub mod evaluation_key;
pub mod ggsw;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub mod refresh;
pub mod sharing;
pub mod tensor_key;
pub mod threshold;
pub use evaluation_key::*;
pub use ggsw::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use refresh::*;
pub use sharing::*;
pub use tensor_key::*;
pub use threshold::*;

/// Validates a wide flood before adding its balanced digits to canonical coefficients.
fn assert_flood<BE: poulpy_hal::layouts::Backend, E: poulpy_core::SmudgingInfos>(
    base2k: usize,
    k: usize,
    flood: &E,
) -> poulpy_core::SmudgingNoise {
    use poulpy_hal::layouts::ZnxWord;

    let noise = flood.smudging_infos();
    assert!(
        noise.k > 0 && noise.k <= k,
        "invalid share: flood precision outside the ciphertext precision"
    );
    assert!(
        base2k > 0 && base2k <= BE::ZnxWord::BITS - 2,
        "invalid share: flood noise exceeds coefficient headroom"
    );
    noise.assert_valid_for(base2k, k);
    noise
}
