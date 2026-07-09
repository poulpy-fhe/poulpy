use crate::CKKSResult as Result;
use crate::default::sub::CKKSSubDefault;

use poulpy_core::{GLWENormalize, GLWEShift, GLWESub, layouts::GLWE};
use poulpy_hal::{
    api::{
        VecZnxLshSubBackend, VecZnxLshSubCoeffToCoeffBackend, VecZnxLshTmpBytes, VecZnxRshSubBackend,
        VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    default::plaintext::CKKSPlaintextDefault,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, UnnormalizedCKKSCiphertext, ciphertext::UnnormalizedCKKSCiphertextRefMut},
    oep::carry_verb::ckks_carry_verb_oep,
};

ckks_carry_verb_oep! {
    verb: sub,
    doc_verb: "subtraction",
    impl_trait: CKKSSubImpl,
    default_trait: CKKSSubDefault,
    glwe_bound: GLWESub,
    pt_vec_bounds: [VecZnxLshSubBackend, VecZnxRshSubBackend],
    pt_const_bounds: [VecZnxLshSubCoeffToCoeffBackend, VecZnxRshSubCoeffIntoBackend],
}

#[macro_export]
macro_rules! impl_ckks_sub_defaults {
    ($be:ty) => {
        impl $crate::default::sub::CKKSSubDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_sub_defaults;
