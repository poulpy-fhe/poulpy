//! Deterministic multi-branch PaCo evaluation.
//!
//! A branch rotates the exhausted ciphertext by one coefficient-class stride,
//! evaluates seqPaCo, and rotates the recovered class back. Sequential and
//! parallel drivers share that implementation. The parallel driver uses a
//! caller-bounded number of workers and a distinct backend [`Module`] handle
//! per worker; this avoids assuming that one backend handle is reentrant.
//!
//! Memory model: per-branch **outputs** are deliberately heap-allocated (owned
//! ciphertexts) rather than scratch-carved — they cross worker-thread
//! boundaries into the ordered merge and their count (`κ`) is
//! schedule-dependent. Stage-internal working memory, by contrast, comes from
//! each worker's own scratch arena (`PaCoWorker`), sized per worker.

use crate::layouts::CKKSCiphertextOwned;
use crate::layouts::CKKSPlaintextOwned;
use crate::{CKKSError, CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef;
use std::{sync::mpsc::sync_channel, thread};

use anyhow::Context;
use poulpy_core::{
    GLWEAutomorphism, GLWEKeyswitch, GLWELinearTransformations, GLWERotate,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, CyclotomicOrder, Module, Normalized, ScratchArena, ScratchOwned},
};

use super::{
    bootstrap::paco_bootstrap_branch_validated_into,
    ops::PaCoSlotOps,
    preflight::{branch_schedule, checked_branch_shift, direct_tmp_bytes_validated, preflight, validate_encapsulation_key},
};
use crate::layouts::paco::{context::PaCoContext, keyset::PaCoKeys};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSRotateOps, CKKSSubOps, PaCoScalar,
    },
    layouts::{CKKSModuleAlloc, PaCoWorker},
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl},
};

#[derive(Clone, Copy)]
struct BranchExecution {
    shift: i64,
    output_meta: (usize, usize),
}

#[derive(Clone, Copy)]
struct ValidatedSchedule {
    kappa: usize,
    stride: usize,
    output_meta: (usize, usize),
    required_scratch: usize,
}

/// Evaluates one shifted branch into `output`.
fn run_branch_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    branch: BranchExecution,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE>,
    F: PaCoScalar,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    if branch.shift == 0 {
        return paco_bootstrap_branch_validated_into::<BE, F, K, _>(
            module,
            output,
            input,
            context,
            keys,
            branch.output_meta,
            scratch,
        );
    }

    let mut shifted = module.ckks_ciphertext_alloc_from_infos(input);
    module.glwe_rotate(-branch.shift, &mut shifted, input);
    shifted.set_meta_checked(input.meta())?;
    paco_bootstrap_branch_validated_into::<BE, F, K, _>(module, output, &shifted, context, keys, branch.output_meta, scratch)?;
    module.glwe_rotate_assign(branch.shift, output, scratch);
    Ok(())
}

fn set_recombined_sparsity<BE: Backend + CKKSPaCoCoeffEncodingImpl<BE>, F: PaCoScalar>(
    output: &mut CKKSCiphertextOwned<BE>,
    context: &PaCoContext<BE, F>,
    kappa: usize,
) -> Result<()> {
    let active = kappa
        .checked_mul(context.plan().c())
        .context("PaCo kappa*C overflows usize")?;
    let gap = context.plan().n() / active;
    ckks_ensure!(gap.is_power_of_two(), "PaCo recombination gap {gap} is not a power of two");
    output
        .set_meta_checked(CKKSMeta {
            log_delta: output.log_delta(),
            log_sparsity: gap.trailing_zeros() as usize,
            slots: output.slots(),
        })
        .map_err(Into::into)
}

/// Sequential direct-mode driver shared by the public operation delegate.
pub(crate) fn paco_bootstrap_direct_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE>,
    F: PaCoScalar,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    let (kappa, stride) = branch_schedule(context, input.log_sparsity())?;
    let (output_meta, required_scratch) = preflight(module, output, input, context, keys, false, scratch)?;
    let schedule = ValidatedSchedule {
        kappa,
        stride,
        output_meta,
        required_scratch,
    };
    paco_bootstrap_direct_validated_into::<BE, F, K, _>(module, output, input, context, keys, schedule, scratch)
}

/// Runs the sequential branch schedule after the public-call preflight has
/// validated the common ciphertext, context, and key invariants once.
fn paco_bootstrap_direct_validated_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    schedule: ValidatedSchedule,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE>,
    F: PaCoScalar,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    ckks_ensure!(
        scratch.available() >= schedule.required_scratch,
        "PaCo direct branch needs {} scratch bytes, but only {} are available",
        schedule.required_scratch,
        scratch.available(),
    );
    run_branch_into::<BE, F, K, _>(
        module,
        output,
        input,
        context,
        keys,
        BranchExecution {
            shift: 0,
            output_meta: schedule.output_meta,
        },
        scratch,
    )?;
    if schedule.kappa == 1 {
        return set_recombined_sparsity(output, context, schedule.kappa);
    }
    for branch in 1..schedule.kappa {
        // Keep branch temporaries fresh: reusing a backing buffer across
        // branches would let later automorphisms observe stale limbs from a
        // previous branch.
        let mut branch_output = module.ckks_ciphertext_alloc(context.base2k(), output.k());
        run_branch_into::<BE, F, K, _>(
            module,
            &mut branch_output,
            input,
            context,
            keys,
            BranchExecution {
                shift: checked_branch_shift(branch, schedule.stride)?,
                output_meta: schedule.output_meta,
            },
            scratch,
        )?;
        module.ckks_add_assign(output, &branch_output, scratch)?;
    }
    set_recombined_sparsity(output, context, schedule.kappa)
}

/// Performs the optional dense-to-PaCo switch once, preserving the exhausted
/// ciphertext metadata for the direct branch driver.
fn encapsulate_input<BE, F, K, Src>(
    module: &Module<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    Module<BE>: CKKSModuleAlloc<BE> + GLWEKeyswitch<BE>,
    K: PaCoKeys<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    let switching_key = validate_encapsulation_key(input, context, keys)?;

    let mut structured = module.ckks_ciphertext_alloc(context.base2k(), input.k());
    module.glwe_keyswitch(&mut structured, input, &switching_key.to_backend_ref(), scratch);
    structured.set_meta_checked(input.meta())?;
    Ok(structured)
}

/// Sequential encapsulated driver shared by the public operation delegate.
pub(crate) fn paco_bootstrap_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE>,
    F: PaCoScalar,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    let (kappa, stride) = branch_schedule(context, input.log_sparsity())?;
    let (output_meta, required_scratch) = preflight(module, output, input, context, keys, true, scratch)?;
    let schedule = ValidatedSchedule {
        kappa,
        stride,
        output_meta,
        required_scratch,
    };
    let structured = encapsulate_input(module, input, context, keys, scratch)?;
    paco_bootstrap_direct_validated_into::<BE, F, K, _>(module, output, &structured, context, keys, schedule, scratch)
}

/// Parallel direct-mode driver with reusable caller-owned workers and ordered
/// recombination.
#[allow(
    clippy::too_many_arguments,
    reason = "the internal driver mirrors the explicit caller-allocated public parallel operation"
)]
pub(crate) fn paco_bootstrap_parallel_direct_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    workers: &mut [PaCoWorker<BE>],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE> + Sync,
    F: PaCoScalar,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    PaCoContext<BE, F>: Sync,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + Send + Sync,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + Sync,
{
    let (kappa, stride) = branch_schedule(context, input.log_sparsity())?;
    let (output_meta, required_scratch) = preflight(module, output, input, context, keys, false, scratch)?;
    let schedule = ValidatedSchedule {
        kappa,
        stride,
        output_meta,
        required_scratch,
    };
    paco_bootstrap_parallel_direct_validated_into::<BE, F, K, _>(module, output, input, context, keys, schedule, workers, scratch)
}

/// Runs the bounded branch pool after the common runtime preflight.
#[allow(
    clippy::too_many_arguments,
    reason = "validated schedule state is kept separate from the caller-owned execution resources"
)]
fn paco_bootstrap_parallel_direct_validated_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    schedule: ValidatedSchedule,
    workers: &mut [PaCoWorker<BE>],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE> + Sync,
    F: PaCoScalar,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    PaCoContext<BE, F>: Sync,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + Send + Sync,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + Sync,
{
    // The caller thread evaluates branch zero while the remaining execution
    // contexts each receive their own Module and scratch arena.
    let background_workers = workers.len().min(schedule.kappa.saturating_sub(1));
    if background_workers == 0 {
        return paco_bootstrap_direct_validated_into::<BE, F, K, _>(module, output, input, context, keys, schedule, scratch);
    }

    // Validate every borrowed execution context before any branch can mutate
    // the output or caller scratch. Scratch is queried on the worker's own
    // module because backend handles may have configuration-specific bounds.
    for (worker, worker_context) in workers[..background_workers].iter_mut().enumerate() {
        let (worker_module, worker_scratch) = worker_context.parts_mut();
        ckks_ensure!(
            worker_module.n() == context.plan().n(),
            "PaCo worker {} module degree {} does not match plan degree {}",
            worker + 1,
            worker_module.n(),
            context.plan().n(),
        );
        ckks_ensure!(
            worker_module.cyclotomic_order() == module.cyclotomic_order(),
            "PaCo worker {} cyclotomic order {} does not match caller order {}",
            worker + 1,
            worker_module.cyclotomic_order(),
            module.cyclotomic_order(),
        );
        let worker_required = direct_tmp_bytes_validated(worker_module, output, context, keys)
            .with_context(|| format!("cannot size PaCo worker {} scratch", worker + 1))?;
        let worker_available = worker_scratch.borrow().available();
        ckks_ensure!(
            worker_available >= worker_required,
            "PaCo worker {} needs {} scratch bytes, but only {} are available",
            worker + 1,
            worker_required,
            worker_available,
        );
    }

    // Per-worker one-slot channels keep peak branch-result storage
    // O(workers.len()), while round-robin receives preserve exact sequential
    // branch order. All predictable worker errors were rejected above before
    // the caller-owned output is touched.
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(background_workers);
        let mut receivers = Vec::with_capacity(background_workers);
        let mut spawn_error = None;
        for (worker, worker_context) in workers[..background_workers].iter_mut().enumerate() {
            let (worker_module, worker_scratch) = worker_context.parts_mut();
            let (sender, receiver) = sync_channel(1);
            let spawn = thread::Builder::new()
                .name(format!("paco-{}", worker + 1))
                .spawn_scoped(scope, move || {
                    for branch in ((worker + 1)..schedule.kappa).step_by(background_workers) {
                        let result = (|| -> Result<(usize, CKKSCiphertextOwned<BE>)> {
                            let mut branch_output = worker_module
                                .ckks_ciphertext_alloc(context.base2k(), TorusPrecision(schedule.output_meta.1 as u32));
                            run_branch_into::<BE, F, K, _>(
                                worker_module,
                                &mut branch_output,
                                input,
                                context,
                                keys,
                                BranchExecution {
                                    shift: checked_branch_shift(branch, schedule.stride)?,
                                    output_meta: schedule.output_meta,
                                },
                                &mut worker_scratch.borrow(),
                            )
                            .with_context(|| format!("PaCo worker {} failed branch {branch}", worker + 1))?;
                            Ok((branch, branch_output))
                        })();
                        let failed = result.is_err();
                        if sender.send(result).is_err() || failed {
                            return;
                        }
                    }
                });
            match spawn {
                Ok(handle) => {
                    handles.push((worker + 1, handle));
                    receivers.push(receiver);
                }
                Err(error) => {
                    spawn_error = Some(CKKSError::Internal(::anyhow::anyhow!(
                        "cannot spawn PaCo worker {}: {error}",
                        worker + 1
                    )));
                    break;
                }
            }
        }

        // This runs concurrently with all successfully spawned workers and
        // never shares the caller's Module handle with another thread.
        let branch_zero = if spawn_error.is_none() {
            run_branch_into::<BE, F, K, _>(
                module,
                output,
                input,
                context,
                keys,
                BranchExecution {
                    shift: 0,
                    output_meta: schedule.output_meta,
                },
                scratch,
            )
            .context("PaCo caller context failed branch 0")
        } else {
            Ok(())
        };

        let mut evaluation_error = spawn_error.or_else(|| branch_zero.err().map(CKKSError::from));
        if evaluation_error.is_none() {
            for expected in 1..schedule.kappa {
                let worker = (expected - 1) % background_workers;
                match receivers[worker].recv() {
                    Ok(Ok((branch, branch_output))) if branch == expected => {
                        if let Err(error) = module
                            .ckks_add_assign(output, &branch_output, scratch)
                            .with_context(|| format!("cannot recombine PaCo branch {expected}"))
                        {
                            evaluation_error = Some(CKKSError::from(error));
                            break;
                        }
                    }
                    Ok(Ok((branch, _))) => {
                        evaluation_error = Some(CKKSError::Internal(::anyhow::anyhow!(
                            "PaCo worker returned branch {branch}, expected deterministic branch {expected}"
                        )));
                        break;
                    }
                    Ok(Err(error)) => {
                        evaluation_error = Some(error);
                        break;
                    }
                    Err(error) => {
                        evaluation_error = Some(CKKSError::Internal(::anyhow::anyhow!(
                            "PaCo worker for branch {expected} disconnected: {error}"
                        )));
                        break;
                    }
                }
            }
        }

        // Dropping receivers unblocks any worker whose bounded send was
        // waiting after another branch failed.
        drop(receivers);
        for (worker, handle) in handles {
            if handle.join().is_err() && evaluation_error.is_none() {
                evaluation_error = Some(CKKSError::Internal(::anyhow::anyhow!("PaCo worker {worker} thread panicked")));
            }
        }

        if let Some(error) = evaluation_error {
            return Err(error);
        }
        Ok(())
    })?;
    set_recombined_sparsity(output, context, schedule.kappa)
}

/// Parallel encapsulated driver. Encapsulation is deliberately performed once
/// on the caller's module before read-only branch sharing begins.
#[allow(
    clippy::too_many_arguments,
    reason = "the internal driver mirrors the explicit caller-allocated public parallel operation"
)]
pub(crate) fn paco_bootstrap_parallel_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    workers: &mut [PaCoWorker<BE>],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    K: PaCoKeys<BE> + Sync,
    F: PaCoScalar,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    PaCoContext<BE, F>: Sync,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + Send + Sync,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized>,
    Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
{
    let (kappa, stride) = branch_schedule(context, input.log_sparsity())?;
    let (output_meta, required_scratch) = preflight(module, output, input, context, keys, true, scratch)?;
    let schedule = ValidatedSchedule {
        kappa,
        stride,
        output_meta,
        required_scratch,
    };
    let structured = encapsulate_input(module, input, context, keys, scratch)?;
    paco_bootstrap_parallel_direct_validated_into::<BE, F, K, _>(
        module,
        output,
        &structured,
        context,
        keys,
        schedule,
        workers,
        scratch,
    )
}
