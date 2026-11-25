use std::{error::Error, fmt};

use poulpy_core::layouts::Degree;
use poulpy_hal::api::ModuleN;

/// Typed error returned by every fallible CKKS **operation** (the `api` op
/// traits and the layers backing them).
///
/// Callers branch on [`CKKSError::Composition`] for recoverable semantic
/// conditions — e.g. [`CKKSCompositionError::InsufficientHomomorphicCapacity`]
/// means "bootstrap now"; the layout/plan mismatch variants mean caller misuse.
/// [`CKKSError::Internal`] carries an [`anyhow::Error`] context chain for
/// invariant or backend failures and is not intended for matching.
///
/// Every operation refuses a ciphertext or linear-transformation diagonal above its module's degree before any write:
/// with [`CKKSCompositionError::EncryptionDegreeMismatch`] for encryption and decryption, with
/// [`CKKSCompositionError::InvalidOperand`] otherwise. A plaintext, a constant included, is checked to embed into its
/// ciphertext's degree instead. The `# Errors` section of an operation lists its other refusals.
///
/// Setup-time constructors (plan compilation, key-set assembly) still return
/// `anyhow::Result`; typed variants they raise internally are recovered by the
/// downcasting [`From<anyhow::Error>`] bridge when they cross an op boundary.
#[derive(Debug)]
#[non_exhaustive]
pub enum CKKSError {
    /// Semantic composition/alignment failure; match on the inner
    /// [`CKKSCompositionError`] to branch on the condition.
    Composition(CKKSCompositionError),
    /// Internal invariant or backend failure, with its context chain.
    Internal(anyhow::Error),
}

impl CKKSError {
    /// The semantic composition error, if this is one.
    pub fn composition(&self) -> Option<&CKKSCompositionError> {
        match self {
            Self::Composition(c) => Some(c),
            Self::Internal(_) => None,
        }
    }
}

impl fmt::Display for CKKSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Composition(c) => fmt::Display::fmt(c, f),
            // `{:#}` renders anyhow's full context chain on one line.
            Self::Internal(e) => write!(f, "{e:#}"),
        }
    }
}

impl Error for CKKSError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Composition(c) => c.source(),
            Self::Internal(e) => Some(e.as_ref()),
        }
    }
}

impl From<CKKSCompositionError> for CKKSError {
    fn from(e: CKKSCompositionError) -> Self {
        Self::Composition(e)
    }
}

/// Bridge from `anyhow`-based internals: a [`CKKSCompositionError`] anywhere in
/// the chain is recovered as [`CKKSError::Composition`] (so typed conditions
/// raised inside setup/compile helpers stay matchable); anything else becomes
/// [`CKKSError::Internal`] with its context preserved.
impl From<anyhow::Error> for CKKSError {
    fn from(e: anyhow::Error) -> Self {
        let composition = e.chain().find_map(|cause| {
            cause
                .downcast_ref::<CKKSCompositionError>()
                .or_else(|| cause.downcast_ref::<CKKSError>()?.composition())
        });
        match composition {
            Some(c) => Self::Composition(c.clone()),
            None => Self::Internal(e),
        }
    }
}

/// Result type of every fallible CKKS operation.
pub type CKKSResult<T> = core::result::Result<T, CKKSError>;

/// `bail!` for [`CKKSResult`] contexts: formats into [`CKKSError::Internal`].
macro_rules! ckks_bail {
    ($($arg:tt)*) => {
        return Err($crate::CKKSError::Internal(::anyhow::anyhow!($($arg)*)))
    };
}

/// `ensure!` for [`CKKSResult`] contexts: formats into [`CKKSError::Internal`].
macro_rules! ckks_ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::CKKSError::Internal(::anyhow::anyhow!($($arg)*)));
        }
    };
}

/// `ensure!` for caller-fixable operand conditions: `InvalidOperand` for `$op`, the reason formatted like `format!`.
macro_rules! ckks_ensure_operand {
    ($op:expr, $cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::CKKSError::Composition($crate::CKKSCompositionError::InvalidOperand {
                op: $op,
                reason: format!($($arg)*),
            }));
        }
    };
}

pub(crate) use {ckks_bail, ckks_ensure, ckks_ensure_operand};

type Result<T> = CKKSResult<T>;

/// CKKS composition and alignment errors returned by high-level operations.
///
/// These errors describe semantic failures such as insufficient precision,
/// incompatible plaintext/ciphertext layouts, or metadata that cannot fit in
/// the requested output storage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CKKSCompositionError {
    /// Shrinking a ciphertext buffer would drop required semantic bits.
    LimbReallocationShrinksBelowMetadata {
        max_k: usize,
        log_delta: usize,
        base2k: usize,
        requested_limbs: usize,
    },
    /// An operation requires more `log_budget` than is still available.
    InsufficientHomomorphicCapacity {
        op: &'static str,
        available_log_budget: usize,
        required_bits: usize,
    },
    /// A plaintext and ciphertext use different limb radices.
    PlaintextBase2KMismatch {
        op: &'static str,
        ct_base2k: usize,
        pt_base2k: usize,
    },
    /// A plaintext operation received a plaintext of a degree its `rule` excludes.
    PlaintextDegreeMismatch {
        op: &'static str,
        ct_n: usize,
        pt_n: usize,
        rule: &'static str,
    },
    /// Encryption or decryption received a ciphertext and secret key of different degrees, or above the module's.
    EncryptionDegreeMismatch {
        op: &'static str,
        module_n: usize,
        ct_n: usize,
        sk_n: usize,
    },
    /// No rotation/conjugation key: the key source has none, or refused the request, as `source` says. A key the
    /// source holds but whose layout is too narrow is [`Self::AutomorphismKeyTooNarrow`].
    MissingAutomorphismKey {
        op: &'static str,
        /// The slot rotation of the missing key, that a conjugation composes with (0 alone), or its Galois element
        /// at the operand's order where `op` looks keys up by element: linear transformations, replicated
        /// conversions, bootstrapping and folds.
        rotation: i64,
        k: u32,
        source: poulpy_core::CoreError,
    },
    /// A rotation/conjugation key the source holds whose layout does not decompose an input of the precision `k` the
    /// operation uses it at: its `max_input_k` is below `k`.
    AutomorphismKeyTooNarrow {
        op: &'static str,
        rotation: i64,
        k: u32,
        max_input_k: u32,
        source: poulpy_core::CoreError,
    },
    /// No relinearization key at the precision the operation works at: the key source has none, or refused the
    /// request, as `source` says. A key the source holds but whose layout is too narrow is
    /// [`Self::RelinearizationKeyTooNarrow`].
    MissingRelinearizationKey {
        op: &'static str,
        k: u32,
        source: poulpy_core::CoreError,
    },
    /// A relinearization key the source holds whose layout does not decompose an input of the precision `k` the
    /// operation uses it at, or the product it relinearizes: its `max_input_k` is below `k`.
    RelinearizationKeyTooNarrow {
        op: &'static str,
        k: u32,
        max_input_k: u32,
        source: poulpy_core::CoreError,
    },
    /// A bootstrapping preset, built or re-derived by `op`, does not validate at its security level (see
    /// [`presets::bootstrapping`](crate::presets::bootstrapping#security)): `source` is the
    /// [`SecurityBoundExceeded`](poulpy_core::CoreError::SecurityBoundExceeded) of the raised ciphertext or a key,
    /// with its modulus and bound, or the [`SecurityUnsupported`](poulpy_core::CoreError::SecurityUnsupported) of a
    /// secret the table has no bound for.
    SecurityLevelNotMet {
        op: &'static str,
        source: poulpy_core::CoreError,
    },
    /// A plaintext cannot be aligned into the requested destination precision.
    PlaintextAlignmentImpossible {
        op: &'static str,
        ct_log_budget: usize,
        pt_log_delta: usize,
        pt_k: usize,
    },
    /// A multiplication would consume more semantic precision than available.
    MultiplicationPrecisionUnderflow {
        op: &'static str,
        lhs_log_budget: usize,
        rhs_log_budget: usize,
        lhs_log_delta: usize,
        rhs_log_delta: usize,
    },
    /// A plan literal (DFT, EvalMod, bootstrapping) violates a shape or
    /// numeric invariant required to compile or evaluate it.
    InvalidPlan { plan: &'static str, reason: String },
    /// An operand does not fit the operation: its shape, dimension, count, shift,
    /// batch, context or width, as opposed to a plan literal ([`Self::InvalidPlan`]).
    InvalidOperand { op: &'static str, reason: String },
    /// A prepared right multiplication operand was built under a different
    /// ring degree, limb radix, or rank than the destination it is applied to.
    PreparedOperandLayoutMismatch {
        op: &'static str,
        dst_n: usize,
        dst_base2k: usize,
        dst_rank: usize,
        prep_n: usize,
        prep_base2k: usize,
        prep_rank: usize,
    },
    /// A homomorphic-DFT evaluation received a prepared matrix whose
    /// kind/format/sparsity does not match the entry point.
    DftMatrixMismatch {
        op: &'static str,
        expected: &'static str,
        got: String,
    },
}

impl fmt::Display for CKKSCompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimbReallocationShrinksBelowMetadata {
                max_k,
                log_delta,
                base2k,
                requested_limbs,
            } => write!(
                f,
                "cannot reallocate to {requested_limbs} limbs: requested capacity is {} bits but ciphertext metadata requires a larger buffer (max_k={max_k}, log_delta={log_delta}, base2k={base2k})",
                requested_limbs * base2k,
            ),
            Self::InsufficientHomomorphicCapacity {
                op,
                available_log_budget,
                required_bits,
            } => write!(
                f,
                "{op} cannot consume {required_bits} bits of log_budget: only {available_log_budget} bits remain"
            ),
            Self::PlaintextBase2KMismatch {
                op,
                ct_base2k,
                pt_base2k,
            } => write!(
                f,
                "{op} requires matching base2k values, got ciphertext base2k={ct_base2k} and plaintext base2k={pt_base2k}"
            ),
            Self::PlaintextDegreeMismatch { op, ct_n, pt_n, rule } => {
                write!(
                    f,
                    "{op} cannot read a plaintext of degree {pt_n} against a ciphertext of degree {ct_n}: the degree must be {rule}"
                )
            }
            Self::EncryptionDegreeMismatch {
                op,
                module_n,
                ct_n,
                sk_n,
            } => write!(
                f,
                "{op} requires equal ciphertext and secret key degrees of at most module degree {module_n}, got ciphertext degree {ct_n} and secret key degree {sk_n}"
            ),
            Self::MissingAutomorphismKey { op, rotation, k, source } => {
                write!(
                    f,
                    "{op} requires an automorphism key for rotation {rotation} at precision k={k}: {source}"
                )
            }
            Self::MissingRelinearizationKey { op, k, source } => {
                write!(f, "{op} requires a relinearization key at precision k={k}: {source}")
            }
            Self::SecurityLevelNotMet { op, source } => {
                write!(f, "{op} does not meet the preset's security level: {source}")
            }
            Self::AutomorphismKeyTooNarrow {
                op,
                rotation,
                k,
                max_input_k,
                ..
            } => write!(
                f,
                "{op} uses the automorphism key for rotation {rotation} at precision k={k}, past its layout's max_input_k={max_input_k}"
            ),
            Self::RelinearizationKeyTooNarrow { op, k, max_input_k, .. } => write!(
                f,
                "{op} uses the relinearization key at precision k={k}, past its layout's max_input_k={max_input_k}"
            ),
            Self::PlaintextAlignmentImpossible {
                op,
                ct_log_budget,
                pt_log_delta,
                pt_k,
            } => write!(
                f,
                "{op} cannot align plaintext with ciphertext: ct.log_budget + pt.log_delta = {} but plaintext precision is {pt_k} bits (ct.log_budget={ct_log_budget}, pt.log_delta={pt_log_delta})",
                ct_log_budget + pt_log_delta
            ),
            Self::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_budget,
                rhs_log_budget,
                lhs_log_delta,
                rhs_log_delta,
            } => write!(
                f,
                "{op} cannot compose inputs: min(log_budget)={} is smaller than min(log_delta)={} (lhs: log_budget={lhs_log_budget}, log_delta={lhs_log_delta}; rhs: log_budget={rhs_log_budget}, log_delta={rhs_log_delta})",
                lhs_log_budget.min(rhs_log_budget),
                lhs_log_delta.min(rhs_log_delta)
            ),
            Self::InvalidPlan { plan, reason } => {
                write!(f, "invalid {plan}: {reason}")
            }
            Self::InvalidOperand { op, reason } => {
                write!(f, "{op}: invalid operand: {reason}")
            }
            Self::PreparedOperandLayoutMismatch {
                op,
                dst_n,
                dst_base2k,
                dst_rank,
                prep_n,
                prep_base2k,
                prep_rank,
            } => write!(
                f,
                "{op} received a prepared operand from a different layout: destination (n={dst_n}, base2k={dst_base2k}, rank={dst_rank}) vs prepared (n={prep_n}, base2k={prep_base2k}, rank={prep_rank})"
            ),
            Self::DftMatrixMismatch { op, expected, got } => {
                write!(f, "{op} requires a {expected} DFT matrix, got {got}")
            }
        }
    }
}

impl CKKSCompositionError {
    /// The refusal of `op` for a failed lookup of the rotation key of `rotation` at precision `k`:
    /// [`Self::AutomorphismKeyTooNarrow`] when the source answered with a key too narrow for `k`,
    /// [`Self::MissingAutomorphismKey`] otherwise.
    pub fn automorphism_key_lookup(op: &'static str, rotation: i64, k: u32, source: poulpy_core::CoreError) -> Self {
        match too_narrow(&source) {
            Some(max_input_k) => Self::AutomorphismKeyTooNarrow {
                op,
                rotation,
                k,
                max_input_k,
                source,
            },
            None => Self::MissingAutomorphismKey { op, rotation, k, source },
        }
    }

    /// The refusal of `op` for a failed lookup of the relinearization key at precision `k`:
    /// [`Self::RelinearizationKeyTooNarrow`] when the source answered with a key too narrow for `k`,
    /// [`Self::MissingRelinearizationKey`] otherwise.
    pub fn tensor_key_lookup(op: &'static str, k: u32, source: poulpy_core::CoreError) -> Self {
        match too_narrow(&source) {
            Some(max_input_k) => Self::RelinearizationKeyTooNarrow {
                op,
                k,
                max_input_k,
                source,
            },
            None => Self::MissingRelinearizationKey { op, k, source },
        }
    }
}

/// The `max_input_k` of a key a lookup found too narrow.
fn too_narrow(source: &poulpy_core::CoreError) -> Option<u32> {
    match source {
        poulpy_core::CoreError::GGLWEKeyTooNarrow { max_input_k, .. } => Some(*max_input_k),
        _ => None,
    }
}

impl Error for CKKSCompositionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MissingAutomorphismKey { source, .. }
            | Self::AutomorphismKeyTooNarrow { source, .. }
            | Self::MissingRelinearizationKey { source, .. }
            | Self::RelinearizationKeyTooNarrow { source, .. }
            | Self::SecurityLevelNotMet { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// `InvalidOperand` for `op` carrying a validator's error as its reason.
pub(crate) fn invalid_operand(op: &'static str, error: impl fmt::Display) -> CKKSError {
    CKKSCompositionError::InvalidOperand {
        op,
        reason: error.to_string(),
    }
    .into()
}

pub(crate) fn checked_log_budget_sub(op: &'static str, available_log_budget: usize, required_bits: usize) -> Result<usize> {
    available_log_budget.checked_sub(required_bits).ok_or_else(|| {
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op,
            available_log_budget,
            required_bits,
        }
        .into()
    })
}

pub(crate) fn ensure_base2k_match(op: &'static str, ct_base2k: usize, pt_base2k: usize) -> Result<()> {
    if ct_base2k != pt_base2k {
        return Err(CKKSCompositionError::PlaintextBase2KMismatch {
            op,
            ct_base2k,
            pt_base2k,
        }
        .into());
    }
    Ok(())
}

pub(crate) fn ensure_encryption_degrees(op: &'static str, module_n: usize, ct_n: usize, sk_n: usize) -> Result<()> {
    if ct_n != sk_n || ct_n > module_n {
        return Err(CKKSCompositionError::EncryptionDegreeMismatch {
            op,
            module_n,
            ct_n,
            sk_n,
        }
        .into());
    }
    Ok(())
}

/// `InvalidOperand` for `op` when one of the operand `degrees` exceeds the degree of `module`.
pub(crate) fn ensure_operand_degrees<M: ModuleN + ?Sized>(
    op: &'static str,
    module: &M,
    degrees: impl IntoIterator<Item = Degree>,
) -> Result<()> {
    if degrees.into_iter().any(|n| n.as_usize() > module.n()) {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "operand degree exceeds the module degree".into(),
        }
        .into());
    }
    Ok(())
}

/// A plaintext operand embeds into the ciphertext ring when its degree is the
/// ciphertext's, or a power-of-two divisor of it.
pub(crate) fn ensure_plaintext_degree_embeds(op: &'static str, ct_n: usize, pt_n: usize) -> Result<()> {
    if pt_n != ct_n && !(pt_n.is_power_of_two() && ct_n.is_multiple_of(pt_n)) {
        return Err(CKKSCompositionError::PlaintextDegreeMismatch {
            op,
            ct_n,
            pt_n,
            rule: DIVISOR,
        }
        .into());
    }
    Ok(())
}

/// The degree rule of the plaintext operands of the coefficient-domain operations and of the prepared dot products.
pub(crate) const DIVISOR: &str = "the ciphertext's or a power-of-two divisor of it";
/// The degree rule of a decryption's plaintext.
pub(crate) const SPARSE_DIVISOR: &str = "the ciphertext's, or a divisor of it the ciphertext's sparsity allows";

pub(crate) fn ensure_plaintext_alignment(op: &'static str, ct_log_budget: usize, pt_log_delta: usize, pt_k: usize) -> Result<()> {
    let available = ct_log_budget + pt_log_delta;
    if available < pt_k {
        return Err(CKKSCompositionError::PlaintextAlignmentImpossible {
            op,
            ct_log_budget,
            pt_log_delta,
            pt_k,
        }
        .into());
    }
    Ok(())
}

/// The operand checks of a plaintext addition or subtraction into a ciphertext of radix `ct_base2k`,
/// degree `ct_n` and budget `ct_log_budget`, in their order.
pub(crate) fn ensure_plaintext_operand<P: crate::CKKSInfos>(
    op: &'static str,
    ct_base2k: usize,
    ct_n: usize,
    ct_log_budget: usize,
    pt: &P,
) -> Result<()> {
    ensure_base2k_match(op, ct_base2k, pt.base2k().as_usize())?;
    ensure_plaintext_degree_embeds(op, ct_n, pt.n().as_usize())?;
    ensure_plaintext_alignment(op, ct_log_budget, pt.log_delta(), pt.log_delta() + pt.log_budget())
}

pub(crate) fn checked_mul_ct_log_budget(
    op: &'static str,
    lhs_log_budget: usize,
    rhs_log_budget: usize,
    lhs_log_delta: usize,
    rhs_log_delta: usize,
) -> Result<usize> {
    // Bivariate-torus CKKS multiplication returns the already-rescaled product:
    // the output scale is `min(log_delta)` and the discarded scale/headroom is
    // therefore `max(log_delta)`, not `lhs_log_delta + rhs_log_delta` as it
    // would be for an unreduced fixed-point product. See the `CKKSMulOps`
    // metadata docs and the bivariate Torus analysis cited in the README
    // ("Revisiting Key Decomposition Techniques for FHE", ePrint 2023/771).
    lhs_log_budget
        .min(rhs_log_budget)
        .checked_sub(lhs_log_delta.max(rhs_log_delta))
        .ok_or_else(|| {
            CKKSCompositionError::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_budget,
                rhs_log_budget,
                lhs_log_delta,
                rhs_log_delta,
            }
            .into()
        })
}

pub(crate) fn checked_mul_pt_log_budget(
    op: &'static str,
    lhs_log_budget: usize,
    rhs_log_budget: usize,
    lhs_log_delta: usize,
    rhs_log_delta: usize,
) -> Result<usize> {
    lhs_log_budget.checked_sub(rhs_log_delta).ok_or_else(|| {
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op,
            lhs_log_budget,
            rhs_log_budget,
            lhs_log_delta,
            rhs_log_delta,
        }
        .into()
    })
}
