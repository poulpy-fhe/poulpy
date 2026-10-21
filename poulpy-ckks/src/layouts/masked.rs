//! Polynomial constants of a replicated slot layout.

use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, GLWEToBackendRef, GetDegree, IntPolyInfos,
    LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Data, Module, Ring, ScratchArena, ScratchOwned, ZnxWord},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSModuleInfos, CoeffsMeta, ReplicatedLayout, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, ComplexPolynomial, Polynomial, SplitStrategy},
    reference::linear_transformation_diagonals::period_degree,
};

use super::{
    CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextOwned,
    eval_mod::{encode_bsgs_with, encode_complex_bsgs_with, encode_scalar_constants},
};

/// Constants of a polynomial for a [`SlotLayout::Replicated`] ciphertext:
/// the scalar constants, which carry the coefficient metadata, and each
/// constant `c_j` as the masked plaintext `c_j·π_S`, of degree `2R` in
/// `Y = X^(N/(2R))` and quantized with shaped rounding at the same scale, so
/// that its quantization error is suppressed on the data slots.
pub struct MaskedCoeffs<P> {
    pub scalars: P,
    pub masked: Vec<P>,
}

impl<P> MaskedCoeffs<P> {
    /// Rebuilds the constants by mapping every plaintext.
    pub fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> MaskedCoeffs<Q> {
        MaskedCoeffs {
            scalars: f(&self.scalars),
            masked: self.masked.iter().map(f).collect(),
        }
    }
}

/// Constants that a replicated slot layout reads as masked plaintexts.
pub trait CKKSMaskedConstants<BE: Backend> {
    type Masked: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos;

    /// The masked plaintext `c_idx·π_S` of constant `idx`, when encoded.
    fn masked_constant(&self, idx: usize) -> Option<&Self::Masked>;
}

impl<BE: Backend, D: Data, W: ZnxWord, R: Ring> CKKSMaskedConstants<BE> for CKKSPlaintext<D, W, R>
where
    Self: GLWEToBackendRef<BE>,
{
    type Masked = Self;

    fn masked_constant(&self, _idx: usize) -> Option<&Self> {
        None
    }
}

impl<BE: Backend, P> CKKSMaskedConstants<BE> for MaskedCoeffs<P>
where
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
{
    type Masked = P;

    fn masked_constant(&self, idx: usize) -> Option<&P> {
        self.masked.get(idx)
    }
}

impl<BE: Backend, P: GLWEToBackendRef<BE>> GLWEToBackendRef<BE> for MaskedCoeffs<P> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        self.scalars.to_backend_ref()
    }
}

impl<P: LWEInfos> LWEInfos for MaskedCoeffs<P> {
    fn base2k(&self) -> Base2K {
        self.scalars.base2k()
    }

    fn max_size(&self) -> usize {
        self.scalars.max_size()
    }

    fn n(&self) -> Degree {
        self.scalars.n()
    }

    fn k(&self) -> TorusPrecision {
        self.scalars.k()
    }
}

impl<P: GLWEInfos> GLWEInfos for MaskedCoeffs<P> {
    fn rank(&self) -> Rank {
        self.scalars.rank()
    }
}

impl<P: IntPolyInfos> IntPolyInfos for MaskedCoeffs<P> {
    fn encoded_k(&self) -> TorusPrecision {
        self.scalars.encoded_k()
    }
}

impl<P: GLWEPlaintextInfos> GLWEPlaintextInfos for MaskedCoeffs<P> {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        self.scalars.plaintext_meta()
    }
}

impl<P: CKKSInfos> CKKSInfos for MaskedCoeffs<P> {}

impl<P: BSGSMeta> BSGSMeta for MaskedCoeffs<P> {
    fn bsgs_log_budget(&self) -> usize {
        self.scalars.bsgs_log_budget()
    }

    fn bsgs_log_delta(&self) -> usize {
        self.scalars.bsgs_log_delta()
    }
}

/// Encodes `values` as scalar constants and as masked plaintexts of `layout`.
pub(crate) fn encode_masked_constants<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    layout: ReplicatedLayout,
    values: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<MaskedCoeffs<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    let scalars = encode_scalar_constants(module, base2k, coeff_meta, values, scratch)?;
    let degree = period_degree::<BE, _>(module, 1 << layout.log_r());
    let masked = values
        .iter()
        .map(|&value| {
            let mut pt = module.ckks_plaintext_alloc(degree, base2k, coeff_meta.k);
            pt.set_meta(GLWEPlaintextMeta {
                slots: SlotsKind::Real,
                slot_layout: SlotLayout::Replicated(layout),
                ..coeff_meta.meta
            });
            module.ckks_encode_masked_constant_into(&mut pt, value, F::zero(), scratch)?;
            Ok(pt)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(MaskedCoeffs { scalars, masked })
}

/// [`encode_masked_constants`] with the phase `phases[j]` on the second half
/// of the `2^log_k` values of `layout`: constant `j` is `c_j` on the first
/// half and `c_j·phases[j]` on the second, a plaintext of `2^log_k·R` slots,
/// or the compact `c_j·π_S` when its phase is one.
pub(crate) fn encode_masked_constants_phased<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    layout: ReplicatedLayout,
    values: &[F],
    phases: &[(F, F)],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<MaskedCoeffs<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    let mut constants = encode_masked_constants(module, base2k, coeff_meta, layout, values, scratch)?;
    let half = 1usize << (layout.log_k() - 1);
    let slots = (2 * half) << layout.log_r();
    let mut host = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
    for ((masked, &value), &(re, im)) in constants.masked.iter_mut().zip(values).zip(phases) {
        if re == F::one() && im == F::zero() {
            continue;
        }
        let mut pt = module.ckks_pt_vec_alloc(slots, base2k, coeff_meta.k);
        pt.set_meta(GLWEPlaintextMeta {
            slots: SlotsKind::Complex,
            slot_layout: SlotLayout::Replicated(layout),
            ..coeff_meta.meta
        });
        let re: Vec<F> = (0..2 * half).map(|j| if j < half { value } else { value * re }).collect();
        let im: Vec<F> = (0..2 * half).map(|j| if j < half { F::zero() } else { value * im }).collect();
        module.ckks_encode_reim_into(&mut pt, &re, &im, &mut host.borrow())?;
        *masked = pt;
    }
    Ok(constants)
}

/// BSGS-encodes `polynomial` for a ciphertext in `layout`, every constant
/// carrying its masked plaintext.
pub fn encode_bsgs_replicated<BE, F>(
    polynomial: &Polynomial<F>,
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    strategy: SplitStrategy,
    layout: ReplicatedLayout,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<BSGSPolynomial<MaskedCoeffs<CKKSPlaintextOwned<BE>>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    let mut coeff_meta = coeff_meta;
    coeff_meta.meta.slots = SlotsKind::Real;
    encode_bsgs_with(polynomial, strategy, false, &mut |values: &[F]| {
        encode_masked_constants(module, base2k, coeff_meta, layout, values, scratch)
    })
}

/// [`encode_bsgs_replicated`] for a complex polynomial.
pub fn encode_complex_bsgs_replicated<BE, F>(
    polynomial: &ComplexPolynomial<F>,
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    strategy: SplitStrategy,
    layout: ReplicatedLayout,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<ComplexBSGSPolynomial<MaskedCoeffs<CKKSPlaintextOwned<BE>>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    let mut coeff_meta = coeff_meta;
    coeff_meta.meta.slots = SlotsKind::Real;
    encode_complex_bsgs_with(polynomial, strategy, &mut |values: &[F]| {
        encode_masked_constants(module, base2k, coeff_meta, layout, values, scratch)
    })
}
