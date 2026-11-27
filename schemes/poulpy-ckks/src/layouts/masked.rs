//! Polynomial constants, each on its own plaintext.

use anyhow::Result;
use poulpy_core::layouts::{Base2K, GLWEPlaintextInfos, GLWEPlaintextMeta, GetDegree, LWEInfos};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, Ring, ScratchArena};

use crate::{
    CKKSModuleInfos, CoeffsMeta, ReplicatedLayout, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    encoding::linear_transformation::period_degree,
    shaping::masked_constant,
};

use super::{CKKSPlaintextAlloc, CKKSPlaintextOwned, CKKSScalar, plaintext::quantize_into_plaintext};

/// The degree of a constant of `slot_layout`: 1 on the full layout, the degree of the period of `π_S` on a replicated
/// one.
pub(crate) fn constant_degree<M: CKKSModuleInfos + GetDegree, Rg: Ring>(module: &M, slot_layout: SlotLayout) -> usize {
    match slot_layout {
        SlotLayout::Replicated(layout) => period_degree::<_, Rg>(module, 1 << layout.log_r()).as_usize(),
        SlotLayout::Full => 1,
    }
}

/// Each of `values` on its own plaintext of `module`, in the slot layout of `coeff_meta`, at the degree of the period
/// of `π_S`: 1 on the full layout, written by `encode`.
fn encode_constants<B, F, Rg: Ring>(
    module: &Module<B>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    values: &[F],
    mut encode: impl FnMut(&mut CKKSPlaintextOwned<B, Rg>, F) -> Result<()>,
) -> Result<Vec<CKKSPlaintextOwned<B, Rg>>>
where
    B: Backend,
    Module<B>: CKKSPlaintextAlloc<B> + GetDegree + CKKSModuleInfos,
    F: Copy,
{
    // A period past the ring is capped at it, which `encode` refuses.
    let degree = constant_degree::<_, Rg>(module, coeff_meta.meta.slot_layout);
    let meta = GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..coeff_meta.meta
    };
    values
        .iter()
        .map(|&value| {
            let mut pt = module.ckks_plaintext_alloc(degree.into(), base2k, coeff_meta.k);
            pt.set_meta_checked(meta)?;
            encode(&mut pt, value)?;
            Ok(pt)
        })
        .collect()
}

/// Encodes `values` as constants of the slot layout of `coeff_meta`.
pub(crate) fn encode_masked_constants<BE, F, Rg: Ring>(
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    values: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSPlaintextOwned<BE, Rg>>>
where
    BE: Backend,
    Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingOps<BE, F, Rg> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    encode_constants(module, base2k, coeff_meta, values, |pt, value| {
        Ok(module.ckks_encode_masked_constant_host_into(pt, value, F::zero(), scratch)?)
    })
}

/// [`encode_masked_constants`] on the host module, through the host codec.
pub(crate) fn encode_host_constants<F: CKKSScalar, Rg: Ring>(
    op: &'static str,
    module: &Module<HostBytesBackend>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    values: &[F],
) -> Result<Vec<CKKSPlaintextOwned<HostBytesBackend, Rg>>> {
    encode_constants(module, base2k, coeff_meta, values, |pt, value| {
        let n = pt.n().as_usize();
        let coeff = masked_constant::<F, Rg>(op, pt.slot_layout(), n, value, F::zero())?;
        quantize_into_plaintext(op, pt, &(0..n).map(coeff).collect::<Vec<_>>())
    })
}

/// Each of `values` as a constant of the slot layout of `coeff_meta`, a plaintext of the host module labelled `Real`:
/// of degree 1 on the full layout, `c·π_S` at the degree of the period of `π_S` on a replicated one, quantized with
/// shaping.
///
/// # Errors
///
/// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a zero `coeff_meta.k`, before allocating
/// anything, and on a replicated layout for a period past the degree of `module`, a shaping the ring refuses or a
/// `coeff_meta` of 126 bits of scale and budget or more, which leaves no shaping pass within `i128`;
/// [`LimbReallocationShrinksBelowMetadata`](crate::CKKSCompositionError::LimbReallocationShrinksBelowMetadata) for a
/// scale above `coeff_meta.k`; and an error for a value not representable at the scale of `coeff_meta`.
pub fn ckks_encode_host_constants<F: CKKSScalar, Rg: Ring>(
    module: &Module<HostBytesBackend>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    values: &[F],
) -> Result<Vec<CKKSPlaintextOwned<HostBytesBackend, Rg>>> {
    const OP: &str = "ckks_encode_host_constants";
    coeff_meta.ensure_width(OP)?;
    encode_host_constants(OP, module, base2k, coeff_meta, values)
}

/// [`encode_masked_constants`] of `coeff_meta` in `layout` with the phase `phases[j]` on the second half of the
/// `2^log_k` values of `layout`: constant `j` is `c_j` on the first half and `c_j·phases[j]` on the second, a plaintext
/// of `2^log_k·R` slots, or `c_j·π_S` when its phase is one.
pub(crate) fn encode_masked_constants_phased<BE, F, Rg: Ring>(
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    layout: ReplicatedLayout,
    values: &[F],
    phases: &[(F, F)],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSPlaintextOwned<BE, Rg>>>
where
    BE: Backend,
    Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingOps<BE, F, Rg> + GetDegree + CKKSModuleInfos,
    F: CKKSEncodingScalar,
{
    let half = 1usize << (layout.log_k() - 1);
    let slots = (2 * half) << layout.log_r();
    values
        .iter()
        .zip(phases)
        .map(|(&value, &(re, im))| {
            if re == F::one() && im == F::zero() {
                let masked = encode_masked_constants(module, base2k, coeff_meta, &[value], scratch)?;
                return Ok(masked.into_iter().next().expect("one value, one constant"));
            }
            let mut pt = module.ckks_plaintext_alloc_from_slots(slots, base2k, coeff_meta.k);
            pt.set_meta(GLWEPlaintextMeta {
                slots: SlotsKind::Complex,
                ..coeff_meta.meta
            });
            let re: Vec<F> = (0..2 * half).map(|j| if j < half { value } else { value * re }).collect();
            let im: Vec<F> = (0..2 * half).map(|j| if j < half { F::zero() } else { value * im }).collect();
            module.ckks_encode_slots_host_into(&mut pt, &re, &im, &mut scratch.borrow())?;
            Ok(pt)
        })
        .collect()
}
