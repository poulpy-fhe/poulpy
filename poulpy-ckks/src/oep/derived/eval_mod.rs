use crate::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, api::CKKSEvalModOps, layouts::eval_mod::EvalMod};
use poulpy_core::layouts::{BSGSMeta, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, SetBSGSMeta};
use poulpy_hal::{
    execution::{TaskExecutor, worker_scratch_bytes},
    layouts::{Backend, Module, ScratchArena},
};

/// Serial maximum or two aligned worker arenas for the selected single-input operations.
pub fn ckks_eval_mod_pair_tmp_bytes<BE, R1, R2, C1, C2, P, F, T>(
    module: &Module<BE>,
    left_out: &R1,
    right_out: &R2,
    left_in: &C1,
    right_in: &C2,
    params: &EvalMod<F, P>,
    tsk: &T,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE>,
    R1: CKKSCtBounds,
    R2: CKKSCtBounds,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
    P: CKKSCtBounds,
    T: poulpy_core::layouts::GGLWEInfos,
{
    let bytes = module
        .ckks_eval_mod_tmp_bytes(left_out, left_in, params, tsk)
        .max(module.ckks_eval_mod_tmp_bytes(right_out, right_in, params, tsk));
    if <BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        2 * worker_scratch_bytes::<BE>(bytes)
    } else {
        bytes
    }
}

/// Evaluates two independent inputs through the selected single-input operations.
/// Parallel execution splits the supplied workspace into two equally sized aligned arenas.
#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_mod_pair<BE, R1, R2, C1, C2, P, F, H>(
    module: &Module<BE>,
    left_out: &mut R1,
    right_out: &mut R2,
    left_in: &C1,
    right_in: &C2,
    params: &EvalMod<F, P>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    C1: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    C2: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta + Sync,
    F: Sync,
    H: GetTensorKey<BE> + Sync,
{
    if !<BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        module.ckks_eval_mod(left_out, left_in, params, tsk, &mut scratch.borrow())?;
        return module.ckks_eval_mod(right_out, right_in, params, tsk, &mut scratch.borrow());
    }
    let bytes = (scratch.available() / 2) / BE::SCRATCH_ALIGN * BE::SCRATCH_ALIGN;
    let (arenas, _) = scratch.borrow().split(2, bytes);
    let mut arenas = arenas.into_iter();
    let mut left_scratch = arenas.next().unwrap();
    let mut right_scratch = arenas.next().unwrap();
    let (left, right) = <BE::TaskExecutor as TaskExecutor>::join(
        || module.ckks_eval_mod(left_out, left_in, params, tsk, &mut left_scratch),
        || module.ckks_eval_mod(right_out, right_in, params, tsk, &mut right_scratch),
    );
    left?;
    right
}
