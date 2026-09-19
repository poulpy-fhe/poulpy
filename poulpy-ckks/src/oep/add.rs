use crate::CKKSResult as Result;
use crate::reference::add::CKKSAddReference;

use poulpy_core::{GLWEAdd, GLWENormalize, GLWEShift, layouts::GLWE};
use poulpy_hal::{
    api::{VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, UnnormalizedCKKSCiphertext, ciphertext::UnnormalizedCKKSCiphertextRefMut},
    oep::carry_verb::ckks_carry_verb_oep,
    reference::plaintext::CKKSPlaintextReference,
};

ckks_carry_verb_oep! {
    verb: add,
    doc_verb: "addition",
    impl_trait: CKKSAddImpl,
    default_trait: CKKSAddReference,
    glwe_bound: GLWEAdd,
    pt_vec_bounds: [VecZnxLshAdd, VecZnxRshAdd],
}

#[macro_export]
macro_rules! impl_ckks_add_reference {
    ($be:ty) => {
        impl $crate::reference::add::CKKSAddReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_add_reference;
