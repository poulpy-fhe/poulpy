use crate::CKKSResult as Result;
use crate::reference::sub::CKKSSubReference;

use poulpy_core::{GLWENormalize, GLWEShift, GLWESub, layouts::GLWE};
use poulpy_hal::{
    api::{VecZnxLshSub, VecZnxLshTmpBytes, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, UnnormalizedCKKSCiphertext, ciphertext::UnnormalizedCKKSCiphertextRefMut},
    oep::carry_verb::ckks_carry_verb_oep,
    reference::plaintext::CKKSPlaintextReference,
};

ckks_carry_verb_oep! {
    verb: sub,
    doc_verb: "subtraction",
    impl_trait: CKKSSubImpl,
    default_trait: CKKSSubReference,
    glwe_bound: GLWESub,
    pt_vec_bounds: [VecZnxLshSub, VecZnxRshSub],
}

#[macro_export]
macro_rules! impl_ckks_sub_reference {
    ($be:ty) => {
        impl $crate::reference::sub::CKKSSubReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_sub_reference;
