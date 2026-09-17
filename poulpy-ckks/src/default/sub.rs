use crate::CKKSResult as Result;
use poulpy_core::{
    GLWENormalize, GLWEShift, GLWESub,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshSub, VecZnxLshTmpBytes, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary,
    default::{CKKSPlaintextDefault, carry_verb::ckks_carry_verb_default},
    layouts::CKKSModuleAlloc,
};

ckks_carry_verb_default! {
    verb: sub,
    doc_verb: "subtraction",
    trait_name: CKKSSubDefault,
    glwe_bound: GLWESub,
    glwe_into: glwe_sub,
    glwe_assign: glwe_sub_assign,
    glwe_lsh_verb: glwe_lsh_sub,
    pt_vec_bounds: [VecZnxLshSub, VecZnxRshSub],
}
