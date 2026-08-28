//! Preflight for the PaCo drivers: runtime validation and scratch sizing.
//!
//! Everything here inspects layouts, keys, and plans without mutating a ciphertext: conservative per-branch scratch bounds, the public tmp-bytes entry point, the branch-schedule arithmetic, and the encapsulation-key checks.
//! The concurrency-bearing branch drivers live in [`parallel`](super::parallel); keeping them free of validation logic keeps that code independently reviewable.

use crate::layouts::CKKSCiphertextOwned;
use crate::{CKKSResult as Result, ckks_ensure};

use anyhow::Context;
use poulpy_core::{
    GLWEAutomorphism, GLWEBytesOf, GLWEKeyswitch, GLWELinearTransformations, GLWERotate,
    layouts::{
        Degree, GGLWEKeyUse, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWELayout,
        GLWERelinearizationKeyLayoutHelper, GLWESwitchingKeyDegrees, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
        WithEffectiveDsize,
    },
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena};

use super::{
    bootstrap::{BranchKeyStage, branch_key_schedule, validate_runtime},
    ops::PaCoSlotOps,
};
use crate::SlotsKind;
use crate::layouts::paco::{
    context::PaCoContext,
    keyset::{PaCoKeys, validate_gadget_backend_view},
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSRotateOps, CKKSSubOps, PaCoScalar,
    },
    layouts::CKKSModuleAlloc,
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl},
};

/// Ciphertext layout used to conservatively size every phase of a branch.
///
/// `k` tracks the widest intermediate while `max_size` preserves the caller's
/// actual destination capacity. A completed output may have a much smaller
/// effective width after compaction, so sizing directly from its current `k`
/// would understate the scratch needed when that buffer is reused.
#[derive(Clone, Copy)]
struct BranchScratchLayout {
    glwe_layout: GLWELayout,
    max_size: usize,
    meta: CKKSMeta,
}

impl BranchScratchLayout {
    fn checked_at(&self, k: TorusPrecision, stage: &str) -> Result<Self> {
        let capacity = self
            .max_size
            .checked_mul(self.base2k().as_usize())
            .with_context(|| format!("{stage} scratch capacity overflows usize"))?;
        ckks_ensure!(
            k.as_usize() <= capacity,
            "{stage} precision {k} exceeds the branch scratch capacity {capacity}",
        );
        ckks_ensure!(
            self.meta.log_delta <= k.as_usize(),
            "{stage} precision {k} is smaller than the branch scale {}",
            self.meta.log_delta,
        );
        let mut layout = *self;
        layout.glwe_layout.k = k;
        Ok(layout)
    }

    fn expect_current(&self, k: TorusPrecision, stage: &str) -> Result<Self> {
        ckks_ensure!(
            self.k() == k,
            "PaCo scratch schedule mismatch before {stage}: current precision {}, stage expects {k}",
            self.k(),
        );
        self.checked_at(k, stage)
    }

    fn advance(&mut self, source_k: TorusPrecision, destination_k: TorusPrecision, stage: &str) -> Result<(Self, Self)> {
        let source = self.expect_current(source_k, stage)?;
        let mut destination = self.checked_at(destination_k, stage)?;
        destination.meta.slots = SlotsKind::Complex;
        *self = destination;
        Ok((source, destination))
    }
}

impl LWEInfos for BranchScratchLayout {
    fn n(&self) -> Degree {
        self.glwe_layout.n()
    }

    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.glwe_layout.base2k()
    }

    fn max_size(&self) -> usize {
        self.max_size
    }

    fn k(&self) -> TorusPrecision {
        self.glwe_layout.k()
    }
}

impl GLWEInfos for BranchScratchLayout {
    fn rank(&self) -> Rank {
        self.glwe_layout.rank()
    }
}

impl CKKSInfos for BranchScratchLayout {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

fn automorphism_layout_for<'a, H, L>(keys: &'a H, element: i64, k: TorusPrecision) -> Result<GGLWEKeyUse<'a, L>>
where
    L: poulpy_core::layouts::GGLWEInfos,
    H: GLWEAutomorphismKeyLayoutHelper<L>,
{
    let (key, effective_dsize) = keys
        .get_automorphism_key_layout_for(element, k)
        .with_context(|| format!("PaCo rotation-key layout is missing Galois element {element} at precision {k}"))?;
    Ok(key.with_dsize(effective_dsize))
}

/// Computes a conservative scratch bound for one direct branch after the
/// common runtime layouts have been validated.
pub(super) fn direct_tmp_bytes_validated<BE, F, K>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    context: &PaCoContext<BE, F>,
    keys: &K,
) -> Result<usize>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
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
{
    let plan = context.plan();
    let canonical = &keys.bootstrapping_keys()[0];
    let schedule = branch_key_schedule(module, context, output.k().as_usize())?;
    let working_k = schedule.working_k();
    let mut branch_layout = BranchScratchLayout {
        glwe_layout: GLWELayout {
            n: canonical.n(),
            base2k: canonical.base2k(),
            k: working_k,
            rank: canonical.rank(),
        },
        // Every stage has the working accumulator's physical capacity while
        // its active precision evolves through the schedule below.
        max_size: working_k.div_ceil(canonical.base2k()) as usize,
        meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: plan.log_delta_bsk(),
            slots: SlotsKind::Complex,
        },
    };
    let beta_k = plan
        .log_delta_bsk()
        .checked_add(plan.log_beta_budget())
        .context("PaCo coefficient-encoding torus width overflows usize")?;
    let degree = u32::try_from(plan.n()).context("PaCo degree does not fit the layout type")?;
    let beta_k = u32::try_from(beta_k).context("PaCo coefficient-encoding width does not fit the layout type")?;
    let beta_layout = CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            n: Degree(degree),
            base2k: context.base2k(),
            k: TorusPrecision(beta_k),
            rank: Rank(1),
        },
        meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: plan.log_delta_bsk(),
            slots: SlotsKind::Complex,
        },
    };

    let bsk = canonical;
    let mut required = module
        .ckks_mul_pt_vec_tmp_bytes(&branch_layout, bsk, &beta_layout)
        .max(module.ckks_add_tmp_bytes())
        .max(module.ckks_sub_tmp_bytes())
        .max(module.ckks_copy_tmp_bytes())
        .max(module.glwe_rotate_tmp_bytes());

    for stage in schedule.stages() {
        match stage {
            BranchKeyStage::Trace { element, k } => {
                let layout = branch_layout.expect_current(*k, "trace")?;
                let key = automorphism_layout_for::<_, K::AutomorphismKey>(keys.rotation_keys(), *element, *k)?;
                required = required.max(module.glwe_automorphism_tmp_bytes(&layout, &layout, &key));
            }
            BranchKeyStage::LinearTransformation {
                factor,
                source_k,
                destination_k,
            } => {
                let (source, destination) = branch_layout.advance(*source_k, *destination_k, "linear transformation")?;
                let eval = module.ckks_eval_linear_transformation_streamed_into_tmp_bytes(
                    &destination,
                    &source,
                    factor,
                    keys.rotation_keys(),
                );
                let stage_required = module
                    .glwe_bytes_of_from_infos(&destination)
                    .checked_add(eval)
                    .context("PaCo linear-transformation scratch size overflows usize")?;
                required = required.max(stage_required);
            }
            BranchKeyStage::PsiPair {
                pair,
                source_k,
                destination_k,
            } => {
                let (source, destination) = branch_layout.advance(*source_k, *destination_k, "paired psi tail")?;
                let conjugation = automorphism_layout_for::<_, K::AutomorphismKey>(keys.rotation_keys(), -1, *source_k)?;
                required = required.max(module.ckks_conjugate_tmp_bytes(&source, &conjugation));
                for factor in pair.iter() {
                    let eval = module.ckks_eval_linear_transformation_streamed_into_tmp_bytes(
                        &destination,
                        &source,
                        factor,
                        keys.rotation_keys(),
                    );
                    let stage_required = module
                        .glwe_bytes_of_from_infos(&destination)
                        .checked_add(eval)
                        .context("PaCo paired-psi linear-transformation scratch size overflows usize")?;
                    required = required.max(stage_required);
                }
            }
            BranchKeyStage::PsiMask {
                mu,
                element,
                source_k,
                destination_k,
            } => {
                let (source, destination) = branch_layout.advance(*source_k, *destination_k, "masked psi tail")?;
                let key = automorphism_layout_for::<_, K::AutomorphismKey>(keys.rotation_keys(), *element, *source_k)?;
                required = required
                    .max(module.ckks_conjugate_tmp_bytes(&source, &key))
                    .max(module.ckks_mul_pt_vec_tmp_bytes(&destination, &source, *mu));
            }
            BranchKeyStage::Product {
                element,
                source_k,
                destination_k,
            } => {
                let (source, destination) = branch_layout.advance(*source_k, *destination_k, "product fold")?;
                let rotation = automorphism_layout_for::<_, K::AutomorphismKey>(keys.rotation_keys(), *element, *source_k)?;
                let (relinearization, effective_dsize) = keys
                    .relinearization_keys()
                    .get_relinearization_key_layout_for(*source_k)
                    .with_context(|| format!("PaCo relinearization-key layout is missing precision {source_k}"))?;
                let relinearization = relinearization.with_dsize(effective_dsize);
                required = required
                    .max(module.ckks_rotate_tmp_bytes(&source, &rotation))
                    .max(module.ckks_mul_tmp_bytes(&destination, &source, &source, &relinearization));
            }
            BranchKeyStage::Conjugation { element, k } => {
                let layout = branch_layout.expect_current(*k, "post-product conjugation")?;
                let key = automorphism_layout_for::<_, K::AutomorphismKey>(keys.rotation_keys(), *element, *k)?;
                required = required.max(module.ckks_conjugate_tmp_bytes(&layout, &key));
            }
        }
    }
    ckks_ensure!(
        branch_layout.k() == schedule.final_k(),
        "PaCo scratch schedule ended at {}, expected {}",
        branch_layout.k(),
        schedule.final_k(),
    );
    required = required.max(BE::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(module, plan)?);
    Ok(required)
}

/// Scratch bytes for one public PaCo call. With `encapsulated`, includes the
/// one-time dense-to-structured key switch in addition to a direct branch.
pub(crate) fn paco_bootstrap_tmp_bytes<BE, F, K, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    encapsulated: bool,
) -> Result<usize>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
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
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    validate_runtime(module, context, output, input, keys)?;
    let direct = direct_tmp_bytes_validated(module, output, context, keys)?;
    if !encapsulated {
        return Ok(direct);
    }
    let switching_key = validate_encapsulation_key(input, context, keys)?;
    let structured = encapsulated_input_layout(input, context);
    Ok(direct.max(module.glwe_keyswitch_tmp_bytes(&structured, input, switching_key)))
}

/// Performs layout validation and rejects undersized scratch before an output
/// ciphertext or encapsulation temporary is mutated.
pub(super) fn preflight<BE, F, K, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    encapsulated: bool,
    scratch: &ScratchArena<'_, BE>,
) -> Result<((usize, usize), usize)>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
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
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let output_meta = validate_runtime(module, context, output, input, keys)?;
    let direct = direct_tmp_bytes_validated(module, output, context, keys)?;
    let required = if encapsulated {
        let switching_key = validate_encapsulation_key(input, context, keys)?;
        let structured = encapsulated_input_layout(input, context);
        direct.max(module.glwe_keyswitch_tmp_bytes(&structured, input, switching_key))
    } else {
        direct
    };
    ckks_ensure!(
        scratch.available() >= required,
        "PaCo bootstrap needs {required} scratch bytes, but only {} are available",
        scratch.available(),
    );
    Ok((output_meta, direct))
}

/// Branch schedule for an input at `log_sparsity`: the number of branches
/// `kappa` and the coefficient stride between them.
///
/// A ciphertext at `log_sparsity = s` carries `M(X^(2^s))`, so its live
/// coefficients are the `N/2^s` multiples of `2^s`. One branch refreshes `C`
/// coefficient classes at gap `N/C` from position 0, and branch `b` is
/// evaluated on the input pre-rotated by `-b*stride`, so `kappa` branches
/// cover the multiples of `stride = N/(kappa*C)`. Taking
/// `kappa = N/(C*2^s)` makes that stride exactly `2^s`: every live
/// coefficient is refreshed once, and no branch spends work on a coefficient
/// the sparsity guarantees is zero.
pub(super) fn branch_schedule<BE: Backend + CKKSPaCoCoeffEncodingImpl<BE>, F: PaCoScalar>(
    context: &PaCoContext<BE, F>,
    log_sparsity: usize,
) -> Result<(usize, usize)> {
    let (n, c) = (context.plan().n(), context.plan().c());
    let stride = 1usize
        .checked_shl(log_sparsity as u32)
        .context("PaCo input sparsity 2^log_sparsity overflows usize")?;
    let live = c.checked_mul(stride).context("PaCo C*2^log_sparsity overflows usize")?;
    ckks_ensure!(
        live <= n,
        "PaCo input at log_sparsity={log_sparsity} has {} live coefficients, fewer than the plan's C={c} classes",
        n / stride,
    );
    ckks_ensure!(
        n.is_multiple_of(live),
        "PaCo C*2^log_sparsity={live} must divide the ring degree {n}"
    );
    let kappa = n / live;
    ckks_ensure!(
        kappa.is_power_of_two(),
        "PaCo branch count N/(C*2^log_sparsity)={kappa} is not a power of two"
    );
    Ok((kappa, stride))
}

pub(super) fn checked_branch_shift(branch: usize, stride: usize) -> Result<i64> {
    let shift = branch.checked_mul(stride).context("PaCo branch shift overflows usize")?;
    i64::try_from(shift)
        .context("PaCo branch shift does not fit i64")
        .map_err(Into::into)
}

/// Resolves and validates the dense-to-PaCo switching key an encapsulated
/// call requires, without touching the input ciphertext.
pub(super) fn validate_encapsulation_key<'a, BE, F, K, Src>(
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &'a K,
) -> Result<&'a K::SwitchingKey>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    K: PaCoKeys<BE>,
    Src: CKKSCtBounds,
{
    let switching_key = keys
        .encapsulation_key()
        .context("PaCo encapsulated bootstrap requires a dense-to-PaCo switching key")?;
    ckks_ensure!(
        input.n().as_usize() == context.plan().n(),
        "PaCo encapsulation input has incompatible degree {}",
        input.n()
    );
    ckks_ensure!(input.rank().as_usize() == 1, "PaCo encapsulation input must have rank 1");
    let structured_size = input.k().as_usize().div_ceil(context.base2k().as_usize());
    let switching_key_view = GGLWEPreparedToBackendRef::to_backend_ref(switching_key);
    validate_gadget_backend_view(
        "PaCo encapsulation key",
        switching_key,
        &switching_key_view,
        context.plan().n(),
        context.base2k(),
        structured_size,
    )?;
    ckks_ensure!(
        switching_key.input_degree().as_usize() == context.plan().n()
            && switching_key.output_degree().as_usize() == context.plan().n(),
        "PaCo encapsulation-key input/output degrees must both equal {}",
        context.plan().n(),
    );
    ckks_ensure!(
        switching_key.k().as_usize() >= input.k().as_usize(),
        "PaCo encapsulation-key width {} is smaller than input width {}",
        switching_key.k(),
        input.k(),
    );
    Ok(switching_key)
}

/// Layout produced by dense-to-PaCo encapsulation. The key switch accepts an
/// input in a different limb radix, but its result must use the context radix
/// consumed by the compiled plaintexts and bootstrapping keys.
fn encapsulated_input_layout<BE: Backend + CKKSPaCoCoeffEncodingImpl<BE>, F: PaCoScalar, Src: CKKSCtBounds>(
    input: &Src,
    context: &PaCoContext<BE, F>,
) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: input.n(),
            base2k: context.base2k(),
            k: input.k(),
            rank: input.rank(),
        },
        meta: input.meta(),
    }
}
