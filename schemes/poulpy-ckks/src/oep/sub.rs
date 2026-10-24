use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, oep::carry_verb::ckks_carry_verb_oep};

ckks_carry_verb_oep! {
    verb: sub,
    doc_verb: "subtraction",
    impl_trait: CKKSSubImpl,
    reference_trait: CKKSSubReference,
    dollar: $,
}

pub use impl_ckks_sub_reference;
