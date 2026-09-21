use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary,
    reference::{CKKSPlaintextReference, carry_verb::ckks_carry_verb_reference},
};

ckks_carry_verb_reference! {
    verb: add,
    doc_verb: "addition",
    trait_name: CKKSAddReference,
    glwe_bound: GLWEAdd,
    glwe_into: glwe_add_into,
    glwe_assign: glwe_add_assign,
    glwe_lsh_verb: glwe_lsh_add,
    pt_vec_bounds: [VecZnxLshAdd, VecZnxRshAdd],
}

impl<BE: Backend> CKKSAddReference<BE> for poulpy_hal::layouts::Module<BE> {}
