//! Preflight for the PaCo drivers: runtime validation and scratch sizing.
//!
//! Everything here inspects layouts, keys, and plans without mutating a ciphertext: conservative per-branch scratch bounds, the public tmp-bytes entry point, the branch-schedule arithmetic, and the encapsulation-key checks.
//! The concurrency-bearing branch drivers live in [`parallel`](super::parallel); keeping them free of validation logic keeps that code independently reviewable.

use crate::{CKKSResult as Result, ckks_ensure};

use anyhow::Context;
use poulpy_core::{
    GLWEAutomorphism, GLWEKeyswitch, GLWELinearTransformations, GLWERotate,
    layouts::{
        Degree, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GLWELayout, GLWESwitchingKeyDegrees,
        GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::{bootstrap::validate_runtime, parallel::PaCoBootstrapModule};
use crate::layouts::paco::{
    context::PaCoContext,
    keyset::{PaCoKeys, validate_gadget_backend_view},
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSMulOps, CKKSRotateOps, CKKSSubOps, PaCoScalar},
    layouts::CKKSCiphertext,
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

/// Computes a conservative scratch bound for one direct branch after the
/// common runtime layouts have been validated.
pub(super) fn direct_tmp_bytes_validated<BE, F, K>(
    module: &Module<BE>,
    output: &CKKSCiphertext<BE::OwnedBuf>,
    context: &PaCoContext<BE, F>,
    keys: &K,
) -> Result<usize>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    Module<BE>: PaCoBootstrapModule<BE>,
    K: PaCoKeys<BE>,
{
    let plan = context.plan();
    let canonical = &keys.bootstrapping_keys()[0];
    // The branch evaluates at working level `output.k() + circuit_depth` (see
    // `branch_working_k`), so size the op scratch from that, not the seed width.
    let working_k = super::bootstrap::branch_working_k(plan, output.k().as_usize())?;
    let branch_layout = BranchScratchLayout {
        glwe_layout: GLWELayout {
            n: canonical.n(),
            base2k: canonical.base2k(),
            k: TorusPrecision(working_k as u32),
            rank: canonical.rank(),
        },
        max_size: working_k.div_ceil(canonical.base2k().as_usize()),
        meta: canonical.meta(),
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
        },
    };

    // Streamed linear transformations prepare a diagonal on demand.  Size
    // that preparation from the widest compiled factor rather than using the
    // output ciphertext as an implicit plaintext proxy: custom factor budgets
    // may legitimately be wider than the final ciphertext.
    let factor_k = plan
        .c2s()
        .log_delta()
        .checked_add(plan.c2s().log_budget())
        .context("PaCo CoeffsToSlots factor width overflows usize")?
        .max(
            plan.stc()
                .log_delta()
                .checked_add(plan.stc().log_budget())
                .context("PaCo SlotsToCoeffs factor width overflows usize")?,
        );
    let factor_k = u32::try_from(factor_k).context("PaCo factor width does not fit the layout type")?;
    let factor_layout = CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            n: Degree(degree),
            base2k: context.base2k(),
            k: TorusPrecision(factor_k),
            rank: Rank(1),
        },
        meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: plan.c2s().log_delta().max(plan.stc().log_delta()),
        },
    };

    let bsk = canonical;
    let mut required = module
        .ckks_mul_pt_vec_tmp_bytes(&branch_layout, bsk, &beta_layout)
        .max(module.ckks_mul_tmp_bytes(&branch_layout, &branch_layout, &branch_layout, keys.tensor_key()))
        .max(module.ckks_add_tmp_bytes())
        .max(module.ckks_sub_tmp_bytes())
        .max(module.ckks_copy_tmp_bytes())
        .max(module.glwe_rotate_tmp_bytes());

    for &element in context.galois_elements() {
        let key = keys
            .rotation_keys()
            .get_automorphism_key(element)
            .with_context(|| format!("PaCo rotation-key map is missing Galois element {element}"))?;
        required = required
            .max(module.glwe_automorphism_tmp_bytes(&branch_layout, &branch_layout, key))
            .max(module.ckks_rotate_tmp_bytes(&branch_layout, key))
            .max(module.ckks_conjugate_tmp_bytes(&branch_layout, key))
            .max(module.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(
                &branch_layout,
                &branch_layout,
                &factor_layout,
                key,
            ));
    }
    required = required.max(BE::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(module, plan)?);
    Ok(required)
}

/// Scratch bytes for one public PaCo call. With `encapsulated`, includes the
/// one-time dense-to-structured key switch in addition to a direct branch.
pub(crate) fn paco_bootstrap_tmp_bytes<BE, F, K, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertext<BE::OwnedBuf>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    encapsulated: bool,
) -> Result<usize>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    Module<BE>: PaCoBootstrapModule<BE> + GLWEKeyswitch<BE>,
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
    output: &CKKSCiphertext<BE::OwnedBuf>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    encapsulated: bool,
    scratch: &ScratchArena<'_, BE>,
) -> Result<((usize, usize), usize)>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    Module<BE>: PaCoBootstrapModule<BE> + GLWEKeyswitch<BE>,
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

/// Validates `kappa` and returns the coefficient stride `N/(kappa*C)`.
pub(super) fn branch_stride<BE: Backend + CKKSPaCoCoeffEncodingImpl<BE>, F: PaCoScalar>(
    context: &PaCoContext<BE, F>,
    kappa: usize,
) -> Result<usize> {
    ckks_ensure!(
        kappa > 0 && kappa.is_power_of_two(),
        "PaCo kappa must be a non-zero power of two, got {kappa}"
    );
    let active = kappa
        .checked_mul(context.plan().c())
        .context("PaCo kappa*C overflows usize")?;
    ckks_ensure!(
        active <= context.plan().n(),
        "PaCo kappa*C={active} exceeds ring degree {}",
        context.plan().n(),
    );
    ckks_ensure!(
        context.plan().n().is_multiple_of(active),
        "PaCo kappa*C must divide the ring degree"
    );
    Ok(context.plan().n() / active)
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
