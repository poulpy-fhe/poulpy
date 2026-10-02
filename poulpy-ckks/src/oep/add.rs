use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, oep::carry_verb::ckks_carry_verb_oep};

ckks_carry_verb_oep! {
    verb: add,
    doc_verb: "addition",
    impl_trait: CKKSAddImpl,
    reference_trait: CKKSAddReference,
    dollar: $,
}

pub use impl_ckks_add_reference;
