use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWENormalize, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, Ring, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, api::CKKSAddOps, checked_log_budget_sub, ckks_offset_binary,
    reference::carry_verb::ckks_carry_verb_reference,
};

ckks_carry_verb_reference! {
    verb: add,
    doc_verb: "addition",
    trait_name: CKKSAddReference,
    ops: CKKSAddOps,
    glwe_bound: GLWEAdd,
    glwe_into: glwe_add_into,
    glwe_assign: glwe_add_assign,
    glwe_lsh_verb: glwe_lsh_add,
    pt_bounds: [VecZnxLshAdd, VecZnxRshAdd],
}

impl<BE: Backend, Rg: Ring> CKKSAddReference<BE, Rg> for poulpy_hal::layouts::Module<BE> {}
