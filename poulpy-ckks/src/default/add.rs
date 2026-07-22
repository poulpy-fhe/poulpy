use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{
        VecZnxLshAddCoeffToCoeffBackend, VecZnxLshAddIntoBackend, VecZnxLshTmpBytes, VecZnxRshAddCoeffIntoBackend,
        VecZnxRshAddIntoBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary,
    default::{CKKSPlaintextDefault, carry_verb::ckks_carry_verb_default},
    layouts::CKKSModuleAlloc,
};

ckks_carry_verb_default! {
    verb: add,
    doc_verb: "addition",
    trait_name: CKKSAddDefault,
    glwe_bound: GLWEAdd,
    glwe_into: glwe_add_into,
    glwe_assign: glwe_add_assign,
    glwe_lsh_verb: glwe_lsh_add,
    pt_vec_bounds: [VecZnxLshAddIntoBackend, VecZnxRshAddIntoBackend],
    pt_const_bounds: [VecZnxLshAddCoeffToCoeffBackend, VecZnxRshAddCoeffIntoBackend],
}
