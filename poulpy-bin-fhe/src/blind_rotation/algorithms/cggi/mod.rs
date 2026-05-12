mod algorithm;
mod key;
mod key_compressed;
mod key_prepared;

use std::marker::PhantomData;

use poulpy_core::{Distribution, layouts::ModuleCoreAlloc};
use poulpy_hal::api::ModuleN;

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos};

/// Algorithm marker for the
/// Chillotti-Gama-Georgieva-Izabachène (CGGI / TFHE) blind rotation.
///
/// `CGGI` is a zero-sized phantom type that selects the CGGI execution path
/// throughout the type system.  It implements [`BlindRotationAlgo`] and is
/// used as the `BRA` type parameter of `BlindRotationKey`,
/// `BlindRotationKeyPrepared`, and `BlindRotationExecute`.
///
/// Three concrete execution paths are dispatched at runtime based on the key
/// distribution stored in the prepared key:
///
/// - **`BinaryFixed` / `BinaryProb`** with `block_size == 1`: Classic CGGI —
///   one GGSW external product per LWE coefficient.
/// - **`BinaryBlock`** with `block_size > 1`: Block-CGGI — `block_size` LWE
///   coefficients are processed together with a shared DFT, reducing total
///   DFT evaluations.
/// - **`BinaryBlock`** with `extension_factor > 1`: Extended block-CGGI — the
///   lookup table spans multiple polynomials, increasing the effective domain
///   of the evaluated function.
#[derive(Clone)]
pub struct CGGI {}

impl BlindRotationAlgo for CGGI {
    fn alloc_key<M, A>(module: &M, infos: &A) -> BlindRotationKey<M::OwnedBuf, Self>
    where
        M: ModuleCoreAlloc + ModuleN,
        A: BlindRotationKeyInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(module.n(), infos.n_glwe().as_usize());
        }
        BlindRotationKey {
            keys: (0..infos.n_lwe().as_usize())
                .map(|_| module.ggsw_alloc_from_infos(infos))
                .collect(),
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}
