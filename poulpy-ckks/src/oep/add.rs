use crate::CKKSResult as Result;
use crate::default::add::CKKSAddDefault;

use poulpy_core::{GLWEAdd, GLWENormalize, GLWEShift, layouts::GLWE};
use poulpy_hal::{
    api::{
        VecZnxLshAddCoeffToCoeffBackend, VecZnxLshAddIntoBackend, VecZnxLshTmpBytes, VecZnxRshAddCoeffIntoBackend,
        VecZnxRshAddIntoBackend, VecZnxRshTmpBytes,
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
    verb: add,
    doc_verb: "addition",
    impl_trait: CKKSAddImpl,
    default_trait: CKKSAddDefault,
    glwe_bound: GLWEAdd,
    pt_vec_bounds: [VecZnxLshAddIntoBackend, VecZnxRshAddIntoBackend],
    pt_const_bounds: [VecZnxLshAddCoeffToCoeffBackend, VecZnxRshAddCoeffIntoBackend],
}

#[macro_export]
macro_rules! impl_ckks_add_defaults {
    ($be:ty) => {
        impl $crate::default::add::CKKSAddDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_add_defaults;
