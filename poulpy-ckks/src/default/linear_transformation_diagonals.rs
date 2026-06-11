//! Production helper: build a [`LinearTransformation<CKKSPlaintext>`] from a
//! raw [`ComplexDiagonals`] map.
//!
//! Performs BSGS pre-rotation + giant-step bucketing inside the
//! scheme-agnostic [`ComplexDiagonals::build_transform`] and only does the
//! per-diagonal host encode + upload here. Callers supply *raw* matrix
//! diagonals — `diag_i[j] = M[j][(j+i) mod cols]` — with no manual
//! `rot(diag, -n1·j)` pre-rotation needed.

use std::fmt::Debug;

use poulpy_core::{
    ModuleTransfer,
    layouts::{Base2K, DiagonalArithmetic, LinearTransformationStrategy},
};
use poulpy_hal::{
    api::{ModuleNew, NegacyclicFFT},
    layouts::{Backend, HostBytesBackend, Module, TransferFrom},
};
use rand_distr::num_traits::{Float, FloatConst, NumCast};

use crate::{
    CKKSInfos, CKKSMeta,
    api::LinearTransformation,
    encoding::reim::Encoder,
    layouts::{CKKSModuleAlloc, CKKSPlaintextVecHostCodec, CKKSScalar, ComplexDiagonals, plaintext::CKKSPlaintext},
};

/// Encodes the complex linear transformation represented by `diagonals` into a
/// [`LinearTransformation<CKKSPlaintext>`] ready for the
/// [`LinearTransformationOps`](crate::api::LinearTransformationOps) pipeline.
///
/// The plaintext encoding goes through the host (CKKS's `encode_reim` is a
/// host-side codec) and is uploaded to the backend; this is the standard path
/// for backends other than `HostBytesBackend`.
///
/// `strategy` selects the BSGS schedule (`Auto` / `Bsgs { giant_step }` /
/// `Direct`). The resulting transform reports `lt.galois_elements(cyclotomic_order)` so the
/// caller knows which Galois keys to generate.
///
/// `transpose` selects the orientation of the matrix-vector product applied to
/// the encrypted input `a`:
/// - `false`: encode `B` as-is — engine produces `B·a` (matrix on the left).
/// - `true`:  encode `Bᵀ` via [`ComplexDiagonals::transpose`] — engine produces
///   `Bᵀ·a = a·B` (matrix on the right of the row vector `a`).
#[allow(clippy::too_many_arguments)]
pub fn ckks_encode_linear_transformation_from_diagonals<BE, F, E>(
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    base2k: Base2K,
    meta: CKKSMeta,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    transpose: bool,
) -> LinearTransformation<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
    F: DiagonalArithmetic + CKKSScalar + Float + FloatConst + Debug + NumCast,
    E: NegacyclicFFT<F>,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    let owned_transposed;
    let cd: &ComplexDiagonals<F> = if transpose {
        let mut t = diagonals.clone();
        t.transpose();
        owned_transposed = t;
        &owned_transposed
    } else {
        diagonals
    };
    cd.build_transform(strategy, |pre_re, pre_im| {
        let mut host_pt = host_module.ckks_pt_vec_alloc(base2k, meta);
        encoder
            .encode_reim(&mut host_pt, pre_re, pre_im)
            .expect("ckks_encode_linear_transformation_from_diagonals: encode_reim slot length mismatch");
        CKKSPlaintext::from_inner(module.upload_glwe_plaintext(&host_pt.inner), host_pt.meta())
    })
}
