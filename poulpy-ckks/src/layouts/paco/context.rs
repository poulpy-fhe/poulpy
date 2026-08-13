//! Compiled PaCo evaluation material.
//!
//! [`PaCoContext`] owns only input-independent plaintext transformations. The
//! encrypted bootstrapping ciphertexts and prepared gadget keys live together
//! in [`PaCoKeys`](super::keyset::PaCoKeys), matching the key ownership model of
//! standard CKKS bootstrapping. The public, caller-allocated entry points are
//! exposed by [`CKKSPaCoOps`](crate::api::CKKSPaCoOps); this module contains the
//! shared implementation used by the sequential and parallel drivers.
//!
//! Two factor fusions already present in the implementation are deliberately
//! preserved: the final partial CoeffsToSlots factor includes the PaCo
//! psi/mu map, and the first SlotsToCoeffs factor includes eta routing and
//! pair packing. They are part of the existing numerical contract and are not
//! changed by the production-readiness refactor.
//!
//! The psi/mu fusion takes whichever form the schedule makes cheapest: merged
//! with butterfly layers it is the conjugation-augmented `(A, B)` pair (one
//! level for the factor, the pairing, and the mask together); with ψ
//! scheduled alone it is the unfused fast tail — one fused conj-rotate
//! keyswitch and one μ-mask multiplication ([`PaCoPsiTailMaterial`]).

use crate::layouts::CKKSPlaintextOwned;
use std::marker::PhantomData;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena};

use super::plan::{PaCoDFTPlan, PaCoPlan};
use crate::SlotsKind;
use crate::default::paco::{
    lt::{PaCoPsiTail, paco_psi_c2s_factors, paco_stc_factors},
    ops::conj_rotate_galois_element,
};
use crate::{
    CKKSMeta,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, LinearTransformation, PaCoScalar},
    layouts::{CKKSModuleAlloc, CKKSScalar},
};

/// Compiled, backend-resident plaintext material for one validated PaCo plan.
///
/// The context is independent of secret and evaluation keys and may be reused
/// for every ciphertext with the same plan, ring degree, and radix. Its fields
/// are private so a plan cannot be paired with transformations compiled for a
/// different layout. `F` records the scalar precision used to generate and
/// encode the factors; the operation API requires the same `F` when it builds
/// input-dependent coefficient encodings, preventing mixed-precision contexts.
///
/// The context carries no FFT engine: factor and input-dependent coefficient
/// encoding are entirely the backend's
/// ([`CKKSPaCoOps::ckks_paco_coeff_encodings`](crate::api::CKKSPaCoOps::ckks_paco_coeff_encodings)).
/// Any precomputed encoding material is the module's own state
/// ([`CKKSEncodingOps`](crate::api::CKKSEncodingOps)); the context stores
/// none.
pub struct PaCoContext<BE: Backend, F> {
    plan: PaCoPlan,
    base2k: Base2K,
    coeffs_to_slots: Vec<LinearTransformation<CKKSPlaintextOwned<BE>>>,
    psi_tail: PaCoPsiTailMaterial<BE>,
    slots_to_coeffs: Vec<LinearTransformation<CKKSPlaintextOwned<BE>>>,
    galois_elements: Vec<i64>,
    scalar: PhantomData<F>,
}

/// Encoded evaluation material for the ψ/μ tail (the compiled form of
/// [`PaCoPsiTail`]).
pub(crate) enum PaCoPsiTailMaterial<BE: Backend> {
    /// The conjugation-augmented pair `(A, B)`: one plain conjugation
    /// keyswitch, then `A·w + B·conj(w)` at one level.
    Pair([LinearTransformation<CKKSPlaintextOwned<BE>>; 2]),
    /// The operation-lean unfused tail (ψ scheduled alone): one fused
    /// conj-rotate keyswitch (`galois_element`), one addition, and one
    /// multiplication by the share-scaled μ mask plaintext.
    Mask {
        mu: CKKSPlaintextOwned<BE>,
        galois_element: i64,
    },
}

/// Rejects generated factors that the host CKKS codec would overflow or
/// quantize entirely to zero. The fallible encoder below performs the same
/// representability check coefficient-by-coefficient; this preflight adds the
/// non-zero-transform invariant needed to catch scaling underflow.
fn validate_factor_encoding<F: CKKSScalar>(diagonals: &crate::layouts::ComplexDiagonals<F>, dft: &PaCoDFTPlan) -> Result<()> {
    let width = dft
        .log_delta()
        .checked_add(dft.log_budget())
        .context("PaCo factor torus width overflows usize")?;
    let scale = F::from(dft.log_delta())
        .context("PaCo factor scale exponent is not representable by the selected scalar")?
        .exp2();
    ensure!(scale.is_finite(), "PaCo factor scale 2^{} is not finite", dft.log_delta());

    let mut nonzero = false;
    for map in [&diagonals.re, &diagonals.im] {
        for index in map.indexes() {
            let values = map
                .get(index)
                .context("PaCo generated-factor index is missing its diagonal")?;
            for &value in values {
                ensure!(value.is_finite(), "PaCo generated factor contains a non-finite coefficient");
                let quantized = (value * scale).round();
                ensure!(
                    quantized.is_finite(),
                    "PaCo generated factor overflows at scale 2^{}",
                    dft.log_delta()
                );
                let representable = if width <= 63 {
                    quantized.to_i64().is_some()
                } else {
                    quantized.to_i128().is_some()
                };
                ensure!(
                    representable,
                    "PaCo generated factor coefficient is not representable at scale 2^{}",
                    dft.log_delta(),
                );
                nonzero |= quantized != <F as DiagonalArithmetic>::zero();
            }
        }
    }
    ensure!(nonzero, "PaCo generated factor quantizes entirely to zero");
    Ok(())
}

/// Checks the worst-case magnitude used by the input-dependent beta
/// plaintexts before accepting a compiled context.
///
/// Every beta slot lies on the unit circle. The encoder's normalized inverse
/// FFT therefore keeps each real coefficient in `[-1, 1]`, so
/// `2^log_delta_bsk` conservatively bounds every quantized coefficient. The
/// host codec stores those coefficients in either `i64` or `i128`, selected
/// from the semantic plaintext width.
fn validate_beta_encoding<F: PaCoScalar>(plan: &PaCoPlan) -> Result<()> {
    let width = plan
        .log_delta_bsk()
        .checked_add(plan.log_beta_budget())
        .context("PaCo beta plaintext width overflows usize")?;
    let scale = F::from(plan.log_delta_bsk())
        .context("PaCo beta scale exponent is not representable by the selected scalar")?
        .exp2();
    ensure!(
        scale.is_finite(),
        "PaCo beta scale 2^{} is not finite for the selected scalar",
        plan.log_delta_bsk(),
    );
    let representable = if width <= 63 {
        scale.round().to_i64().is_some()
    } else {
        scale.round().to_i128().is_some()
    };
    ensure!(
        representable,
        "PaCo beta coefficients are not representable at scale 2^{}",
        plan.log_delta_bsk(),
    );
    Ok(())
}

impl<BE: Backend, F> PaCoContext<BE, F> {
    /// Compiles the PaCo linear transformations for `plan`.
    ///
    /// `module` must have ring degree `plan.n()`, and `base2k` must be
    /// non-zero. Each factor is encoded at the scale and plaintext budget
    /// carried by its [`PaCoDFTPlan`]. Compilation performs host-side matrix
    /// generation once, then stages each diagonal through `scratch` for
    /// backend-native slot encoding.
    ///
    /// Returns an error for a dimension-only or numerically unrepresentable
    /// plan, incompatible module degree/cyclotomic order, invalid radix, or a
    /// factor/plaintext encoding failure.
    pub fn compile(module: &Module<BE>, base2k: Base2K, plan: PaCoPlan, scratch: &mut ScratchArena<'_, BE>) -> Result<Self>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        F: PaCoScalar,
    {
        plan.check_evaluation().context("invalid PaCo evaluation plan")?;
        ensure!(
            module.n() == plan.n(),
            "PaCo backend module degree {} does not match plan degree {}",
            module.n(),
            plan.n()
        );
        let expected_order = plan
            .n()
            .checked_mul(2)
            .context("PaCo cyclotomic order overflows usize")
            .and_then(|order| i64::try_from(order).context("PaCo cyclotomic order does not fit i64"))?;
        ensure!(
            module.cyclotomic_order() == expected_order,
            "PaCo backend cyclotomic order {} does not match plan degree {} (expected {expected_order})",
            module.cyclotomic_order(),
            plan.n(),
        );
        ensure!(
            (1..=63).contains(&base2k.as_usize()),
            "PaCo base2k must be in [1, 63], got {}",
            base2k,
        );
        ensure!(
            plan.log_q() < F::MANTISSA_BITS,
            "PaCo log_q={} requires exact integer support beyond the {}-bit mantissa of the selected scalar",
            plan.log_q(),
            F::MANTISSA_BITS,
        );
        validate_beta_encoding::<F>(&plan)?;

        let full_slots = plan.half_n();
        let encode_factor = |diagonals: &crate::layouts::ComplexDiagonals<F>,
                             dft: &PaCoDFTPlan,
                             giant_step: usize,
                             scratch: &mut ScratchArena<'_, BE>|
         -> Result<LinearTransformation<CKKSPlaintextOwned<BE>>> {
            validate_factor_encoding(diagonals, dft)?;
            let slots = diagonals.slots();
            ensure!(
                slots > 0 && full_slots.is_multiple_of(slots) && (full_slots / slots).is_power_of_two(),
                "PaCo factor slot count {slots} is incompatible with full slot count {full_slots}",
            );
            let k = dft
                .log_delta()
                .checked_add(dft.log_budget())
                .context("PaCo factor torus width overflows usize")?;
            crate::default::ckks_encode_linear_transformation_from_diagonals(
                module,
                base2k,
                crate::CoeffsMeta {
                    k: k.into(),
                    meta: CKKSMeta {
                        log_sparsity: (full_slots / slots).trailing_zeros() as usize,
                        log_delta: dft.log_delta(),
                        slots: SlotsKind::Complex,
                    },
                },
                diagonals,
                poulpy_core::layouts::LinearTransformationStrategy::Bsgs { giant_step },
                false,
                scratch,
            )
            .context("cannot encode PaCo linear-transformation factor")
        };

        let encode_chain = |factors: &[crate::layouts::ComplexDiagonals<F>],
                            dft: &PaCoDFTPlan,
                            giant_steps: &[usize],
                            scratch: &mut ScratchArena<'_, BE>|
         -> Result<Vec<LinearTransformation<CKKSPlaintextOwned<BE>>>> {
            ensure!(
                factors.len() == giant_steps.len(),
                "PaCo generated {} factors for a {}-entry BSGS schedule",
                factors.len(),
                giant_steps.len(),
            );
            factors
                .iter()
                .zip(giant_steps)
                .map(|(factor, &giant_step)| encode_factor(factor, dft, giant_step, scratch))
                .collect()
        };

        let packing_slots = plan.c().checked_mul(2).context("PaCo packing slot count overflows usize")?;

        let (coeffs_to_slots_factors, psi_factors) = paco_psi_c2s_factors::<F>(&plan);
        let coeffs_to_slots = encode_chain(
            &coeffs_to_slots_factors,
            plan.c2s(),
            &plan.c2s().giant_steps()[..coeffs_to_slots_factors.len()],
            scratch,
        )?;
        let psi_tail = match &psi_factors {
            PaCoPsiTail::Pair(pair) => {
                let psi_giant_step = *plan
                    .c2s()
                    .giant_steps()
                    .last()
                    .context("validated PaCo CoeffsToSlots schedule is empty")?;
                PaCoPsiTailMaterial::Pair([
                    encode_factor(&pair[0], plan.c2s(), psi_giant_step, scratch)?,
                    encode_factor(&pair[1], plan.c2s(), psi_giant_step, scratch)?,
                ])
            }
            PaCoPsiTail::Mask(tile) => {
                ensure!(
                    tile.len() == packing_slots,
                    "PaCo ψ mask tile has {} slots, expected {packing_slots}",
                    tile.len(),
                );
                let dft = plan.c2s();
                let scale = F::from(dft.log_delta())
                    .context("PaCo ψ mask scale exponent is not representable by the selected scalar")?
                    .exp2();
                ensure!(
                    tile.iter()
                        .any(|value| (value.re * scale).round() != <F as DiagonalArithmetic>::zero()),
                    "PaCo ψ mask quantizes entirely to zero at scale 2^{}",
                    dft.log_delta(),
                );
                let full = plan.half_n();
                let mut re = vec![<F as DiagonalArithmetic>::zero(); full];
                let mut im = vec![<F as DiagonalArithmetic>::zero(); full];
                for (index, (re_slot, im_slot)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                    let value = tile[index % packing_slots];
                    *re_slot = value.re;
                    *im_slot = value.im;
                }
                let k_pt = dft
                    .log_delta()
                    .checked_add(dft.log_budget())
                    .context("PaCo ψ mask torus width overflows usize")?;
                let mut pt = module.ckks_pt_vec_alloc(base2k, k_pt.into());
                pt.set_meta_checked(CKKSMeta {
                    log_sparsity: 0,
                    log_delta: dft.log_delta(),
                    slots: SlotsKind::Complex,
                })?;
                module
                    .ckks_encode_reim_into(&mut pt, &re, &im, scratch)
                    .context("cannot encode the PaCo ψ mask plaintext")?;
                PaCoPsiTailMaterial::Mask {
                    mu: pt,
                    galois_element: conj_rotate_galois_element(plan.c() as i64, module.cyclotomic_order()),
                }
            }
        };
        let slots_to_coeffs_factors = paco_stc_factors::<F>(&plan);
        let slots_to_coeffs = encode_chain(&slots_to_coeffs_factors, plan.stc(), &plan.stc().giant_steps(), scratch)?;
        let galois_elements = plan.galois_elements_from_factors(
            &coeffs_to_slots_factors,
            &psi_factors,
            &slots_to_coeffs_factors,
            module.cyclotomic_order(),
        );

        Ok(Self {
            plan,
            base2k,
            coeffs_to_slots,
            psi_tail,
            slots_to_coeffs,
            galois_elements,
            scalar: PhantomData,
        })
    }

    /// The validated plan used to compile this context.
    pub fn plan(&self) -> &PaCoPlan {
        &self.plan
    }

    /// Maximum final output level `k` a bootstrap can produce with `keys`: the
    /// bootstrapping-key (seed) width minus the budget the circuit consumes.
    ///
    /// Allocate the bootstrap output ciphertext at any `k` up to this value (e.g.
    /// `module.ckks_ciphertext_alloc(ctx.base2k(), ctx.max_output_k(&keys)?)`); a
    /// lower level runs the whole circuit at a narrower — and cheaper — working level.
    /// Errors if the plan consumes more budget than the key width provides.
    pub fn max_output_k<K>(&self, keys: &K) -> Result<TorusPrecision>
    where
        K: super::keyset::PaCoKeys<BE>,
    {
        let seed_k = keys.bootstrapping_keys()[0].k().as_usize();
        let k_out = seed_k
            .checked_sub(self.plan.consumed_bits())
            .context("PaCo bootstrapping-key width is smaller than the plan's budget consumption")?;
        Ok(TorusPrecision(k_out as u32))
    }

    /// Limb radix expected by ciphertexts, plaintexts, and evaluation keys.
    pub fn base2k(&self) -> Base2K {
        self.base2k
    }

    /// Galois elements required by every evaluation using this context.
    ///
    /// The set is derived while compiling the factor schedules, so runtime
    /// validation does not regenerate the same diagonal matrices merely to
    /// enumerate their rotations.
    pub fn galois_elements(&self) -> &[i64] {
        &self.galois_elements
    }

    pub(crate) fn coeffs_to_slots(&self) -> &[LinearTransformation<CKKSPlaintextOwned<BE>>] {
        &self.coeffs_to_slots
    }

    pub(crate) fn psi_tail(&self) -> &PaCoPsiTailMaterial<BE> {
        &self.psi_tail
    }

    pub(crate) fn slots_to_coeffs(&self) -> &[LinearTransformation<CKKSPlaintextOwned<BE>>] {
        &self.slots_to_coeffs
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_scale_is_rejected_during_context_preflight() {
        let plan = PaCoPlan::new(8, 4, 8, 28)
            .unwrap()
            .with_evaluation(
                1_024,
                16,
                PaCoDFTPlan::uniform(5, 3, 2, 30, 16).unwrap(),
                PaCoDFTPlan::uniform(3, 2, 2, 30, 16).unwrap(),
            )
            .unwrap();
        let error = validate_beta_encoding::<f64>(&plan).expect_err("a non-finite beta scale must be rejected");
        assert!(error.to_string().contains("beta scale"), "unexpected error: {error:#}");
    }
}
