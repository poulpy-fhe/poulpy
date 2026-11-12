//! Coefficient masks on the last CoeffsToSlots factor and the merged last SlotsToCoeffs factor.

use anyhow::{Result, ensure};
use poulpy_core::{
    GLWEDiagonalProd, LinearTransformationPrepared,
    layouts::{
        Base2K, Diagonals, GLWELayout, GLWEToBackendRef, IntPolyInfos, LWEInfos, LinearTransformation, LinearTransformationPlan,
        Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar,
        CKKSLinearTransformationOps,
    },
    layouts::{
        CKKSDFTFactors, CKKSPlaintextAlloc, CKKSPlaintextOwned, ComplexDiagonals, DFTMatrixPrepared, DFTPlan, Decode, DftScalar,
        Encode, Split, Standard,
    },
};

use super::{BootstrappingContextPrepared, BootstrappingPipeline};

/// The contexts a mask serves. A bypass adds back the coefficients the mask drops, and a LUT runs no EvalMod it could
/// clear them in.
const MASK_CONTEXT: &str = "takes an identity S2C-first context with EvalMod on the standard ring and without EvalRound+";

/// The last CoeffsToSlots factor times a slot-wise mask, as a prepared matrix of that one factor: its transform keeps the
/// coefficients the mask selects and zeroes the others, at no level. Built by
/// [`BootstrappingContextPrepared::coeffs_mask`] and evaluated through [`BootstrappingContextPrepared::masked`].
pub struct CoeffsMaskPrepared<BE: Backend>(pub(crate) DFTMatrixPrepared<BE, Encode, Split>);

/// The CoeffsToSlots factors of a context with a [`CoeffsMaskPrepared`] in place of the last one, from
/// [`BootstrappingContextPrepared::masked`].
pub struct MaskedFactors<'a, BE: Backend> {
    dft: &'a DFTMatrixPrepared<BE, Encode, Split>,
    last: &'a LinearTransformationPrepared<BE>,
}

// Unsplit like the context it masks: see `ckks_coeffs_to_slots_assign` for the scale.
impl<BE: Backend> CKKSDFTFactors<BE, Encode, Standard> for MaskedFactors<'_, BE> {
    type Diagonal = <DFTMatrixPrepared<BE, Encode, Split> as CKKSDFTFactors<BE, Encode, Split>>::Diagonal;

    fn dft_plan(&self) -> &DFTPlan {
        self.dft.dft_plan()
    }

    fn factors<'a>(&'a self) -> impl Iterator<Item = &'a LinearTransformation<Self::Diagonal>>
    where
        BE: 'a,
        Encode: 'a,
        Standard: 'a,
    {
        let head = self.dft.factor_operands().split_last().map_or(&[][..], |(_, head)| head);
        head.iter().chain(std::iter::once(self.last))
    }

    // `masked` holds a mask to the schedule of the factor it replaces.
    fn factor_plans(&self) -> &[LinearTransformationPlan] {
        self.dft.factor_plans()
    }
}

/// The last SlotsToCoeffs factor with one kernel polynomial per coset of the coefficients: it finishes SlotsToCoeffs
/// and multiplies each coset by its kernel in the same level. Built by
/// [`BootstrappingContextPrepared::merged_s2c_factor`].
pub struct MergedS2CFactorPrepared<BE: Backend>(pub(crate) LinearTransformationPrepared<BE>);

impl<BE: Backend> MergedS2CFactorPrepared<BE> {
    /// The factor, evaluated as a linear transformation: its rotations are those of the last SlotsToCoeffs factor.
    pub fn factor(&self) -> &LinearTransformationPrepared<BE> {
        &self.0
    }
}

/// The radix and the width of the diagonals of the factors `dft` evaluates, `InvalidOperand` for `op` when it evaluates
/// none, as a matrix whose only factor is on the raise.
fn evaluated_diagonals<BE: Backend, Dir, Fmt>(
    op: &'static str,
    dft: &DFTMatrixPrepared<BE, Dir, Fmt>,
) -> Result<(Base2K, TorusPrecision)> {
    dft.factor_operands()
        .iter()
        .find_map(|factor| factor.first_diagonal_plaintext())
        .map(|diagonal| (diagonal.base2k(), diagonal.encoded_k()))
        .ok_or_else(|| invalid_operand(op, "the matrix evaluates no factor"))
}

fn invalid_operand(op: &'static str, reason: &str) -> anyhow::Error {
    crate::CKKSCompositionError::InvalidOperand {
        op,
        reason: reason.to_string(),
    }
    .into()
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F>
where
    F: CKKSEncodingScalar,
{
    /// The CoeffsToSlots factors of this context with `mask`, from [`Self::coeffs_mask`], in place of the last one, for
    /// [`ckks_coeffs_to_slots_assign`](crate::api::CKKSDFTOps::ckks_coeffs_to_slots_assign).
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a context [`Self::coeffs_mask`] refuses, a `mask`
    /// of another factor or of another plan of CoeffsToSlots, or a context whose CoeffsToSlots evaluates no factor.
    pub fn masked<'a>(&'a self, mask: &'a CoeffsMaskPrepared<BE>) -> crate::CKKSResult<MaskedFactors<'a, BE>> {
        const OP: &str = "masked";
        if !self.takes_a_mask() {
            return Err(invalid_operand(OP, MASK_CONTEXT).into());
        }
        let Some(context_last) = self.coeffs_to_slots().factor_plans().last() else {
            return Err(invalid_operand(OP, "the matrix evaluates no factor").into());
        };
        let of_this_factor = |_: &&LinearTransformationPrepared<BE>| {
            mask.0.factor_plans().first() == Some(context_last) && mask.0.plan() == self.coeffs_to_slots().plan()
        };
        let Some(last) = mask.0.factor_operands().first().filter(of_this_factor) else {
            return Err(invalid_operand(OP, "the mask is not of the last CoeffsToSlots factor of the context").into());
        };
        Ok(MaskedFactors {
            dft: self.coeffs_to_slots(),
            last,
        })
    }

    /// Scratch of [`Self::coeffs_mask`].
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a context whose CoeffsToSlots evaluates no
    /// factor, or a plan the ring does not support.
    pub fn coeffs_mask_tmp_bytes(&self, module: &Module<BE>) -> Result<usize>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSLinearTransformationOps<BE>,
    {
        let dft = self.coeffs_to_slots();
        let (base2k, k) = evaluated_diagonals("coeffs_mask", dft)?;
        // A mask is no longer periodic: its diagonals fill the slots, however compact those of the factor are.
        let dense = GLWELayout {
            n: (2usize << dft.plan().log_slots()).into(),
            base2k,
            k,
            rank: Rank(0),
        };
        Ok(module
            .ckks_dft_matrix_new_tmp_bytes::<Encode, Split>(dft.plan())?
            .max(module.ckks_linear_transformation_rhs_prepare_tmp_bytes(&dense)))
    }

    /// The last CoeffsToSlots factor restricted to the coefficients `keep` selects, for [`Self::masked`]: CoeffsToSlots
    /// reads coefficient `j` of its input where `keep[j]` holds and drops it elsewhere, at no cost to the transform.
    /// Coefficients `p` and `N/2 + p` share a real slot: `keep` selects at most one of them, on either half, and the slot
    /// gives it back as coefficient `p` of the next SlotsToCoeffs. `scratch` covers [`Self::coeffs_mask_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand), for a context that is not an identity S2C-first one
    /// with EvalMod on the standard ring and without EvalRound+, whose bypass would add back what the mask drops, or that
    /// evaluates no CoeffsToSlots factor, a `keep` of another length than the degree `N` of the bootstrap, a `keep`
    /// selecting both coefficients of a slot, or a coefficient off the subring of the plaintexts, all before the arena
    /// check.
    pub fn coeffs_mask(
        &self,
        module: &Module<BE>,
        keep: &[bool],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CoeffsMaskPrepared<BE>>
    where
        Module<BE>:
            CKKSEncodingOps<BE, F> + CKKSDFTMatrixOps<BE, F> + CKKSDFTMatrixPreparedFactory<BE> + CKKSLinearTransformationOps<BE>,
        F: DftScalar,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        const OP: &str = "coeffs_mask";
        let refuse = |reason| Err(invalid_operand(OP, reason));
        if !self.takes_a_mask() {
            return refuse(MASK_CONTEXT);
        }
        let dft = self.coeffs_to_slots();
        let (base2k, _) = evaluated_diagonals(OP, dft)?;
        let plan = dft.plan();
        let slots = 1usize << plan.log_slots();
        let gap = 1usize << self.log_gap();
        let half = slots * gap;
        if keep.len() != 2 * half {
            return refuse("takes one entry per coefficient");
        }
        if (0..half).any(|p| keep[p] && keep[half + p]) {
            return refuse("a real slot carries one of the two coefficients it holds");
        }
        if keep.iter().enumerate().any(|(p, &k)| k && !p.is_multiple_of(gap)) {
            return refuse("a mask keeps coefficients of the subring of the transform");
        }
        let bytes = self.coeffs_mask_tmp_bytes(module)?;
        ensure!(
            scratch.available() >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {}",
            scratch.available()
        );
        // Slot `s` holds coefficients `g·bitrev(s)` and `N/2 + g·bitrev(s)`: `-i` moves the second to the real part kept.
        let bitrev = |s: usize| {
            s.reverse_bits()
                .checked_shr(usize::BITS - slots.trailing_zeros())
                .unwrap_or(0)
        };
        let (mut re, mut im) = (vec![F::zero(); slots], vec![F::zero(); slots]);
        for s in 0..slots {
            let p = gap * bitrev(s);
            if keep[p] {
                re[s] = F::one();
            }
            if keep[half + p] {
                im[s] = -F::one();
            }
        }
        let mut scale = ComplexDiagonals::new(Diagonals::new(slots), Diagonals::new(slots));
        scale.re.set(0, re);
        scale.im.set(0, im);
        let log_n = (2 * half).ilog2() as usize;
        let masked = crate::layouts::dft::ckks_scaled_last_factors::<BE, F>(module, base2k, plan, log_n, &[scale], scratch)?;
        Ok(CoeffsMaskPrepared(module.ckks_dft_matrix_prepare_alloc(&masked, scratch)?))
    }

    /// The cosets of the coefficients the last SlotsToCoeffs factor keeps apart: `2^depth` of that factor.
    pub fn merged_s2c_cosets(&self) -> usize {
        let plan = self.slots_to_coeffs().plan();
        1 << plan.schedule.steps.last().expect("a DFT plan has at least one factor").depth
    }

    /// Scratch of [`Self::merged_s2c_factor`], for a factor of at most `k` bits.
    ///
    /// # Errors
    ///
    /// A plan the ring does not support.
    pub fn merged_s2c_factor_tmp_bytes(&self, module: &Module<BE>, k: usize) -> Result<usize>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    {
        let dft = self.slots_to_coeffs();
        let (base2k, _) = evaluated_diagonals("merged_s2c_factor", dft)?;
        let n = self.bootstrap_degree();
        let dense = GLWELayout {
            n: n.into(),
            base2k,
            k: k.into(),
            rank: Rank(0),
        };
        Ok(module
            .ckks_dft_matrix_new_tmp_bytes::<Decode, Split>(dft.plan())?
            .max(module.ckks_linear_transformation_rhs_prepare_tmp_bytes(&dense))
            .max(<Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_encoding_host_tmp_bytes(
                module, n,
            )))
    }

    /// The last SlotsToCoeffs factor, at scale `2^log_delta`, with `kernels[r % kernels.len()]` the polynomial of the
    /// coefficients of index `g·k` for `k = r` modulo [`Self::merged_s2c_cosets`], `g` the gap of the plaintexts: on the
    /// state SlotsToCoeffs leaves before that factor, from [`DFTMatrix::head`](crate::layouts::DFTMatrix::head), it gives `sum_r c_r·kernels[r]`, with `c_r`
    /// the input's coefficients of coset `r`, at their own indexes. `scratch` covers
    /// [`Self::merged_s2c_factor_tmp_bytes`] at the width of the factor; an arena below it at width `log_delta` is refused
    /// on entry, and one below it at the width of the factor at its preparation.
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand), for a context that is not S2C-first, a kernel count
    /// that does not divide the cosets, a kernel of another length than the degree of the bootstrap, or a kernel off the
    /// subring of the plaintexts, all before the arena check.
    pub fn merged_s2c_factor(
        &self,
        module: &Module<BE>,
        kernels: &[Vec<F>],
        log_delta: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<MergedS2CFactorPrepared<BE>>
    where
        Module<BE>:
            CKKSEncodingOps<BE, F> + CKKSDFTMatrixOps<BE, F> + CKKSDFTMatrixPreparedFactory<BE> + CKKSLinearTransformationOps<BE>,
        F: DftScalar,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        const OP: &str = "merged_s2c_factor";
        let refuse = |reason| Err(invalid_operand(OP, reason));
        if self.pipeline() != BootstrappingPipeline::S2CFirst {
            return refuse("takes an S2C-first context");
        }
        let cosets = self.merged_s2c_cosets();
        if kernels.is_empty() || !cosets.is_multiple_of(kernels.len()) {
            return refuse("takes a kernel count dividing the cosets");
        }
        let n = self.bootstrap_degree();
        if kernels.iter().any(|kernel| kernel.len() != n) {
            return refuse("takes kernels of the degree of the bootstrap");
        }
        let gap = 1usize << self.log_gap();
        if kernels
            .iter()
            .flat_map(|kernel| kernel.iter().enumerate())
            .any(|(e, &c)| c != F::zero() && !e.is_multiple_of(gap))
        {
            return refuse("takes kernels of the subring of the transform");
        }
        let dft = self.slots_to_coeffs();
        let (base2k, _) = evaluated_diagonals(OP, dft)?;
        let bytes = self.merged_s2c_factor_tmp_bytes(module, log_delta)?;
        ensure!(
            scratch.available() >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {}",
            scratch.available()
        );
        // Kernel values, read through the host codec at 60 bits, past the mantissa of `f64`.
        const LOG_SCALE: usize = 60;
        let width = crate::layouts::dft::integer_bits(kernels.iter().flatten().fold(F::one(), |m, v| m.max(v.abs())));
        let mut pt =
            module.ckks_plaintext_alloc::<poulpy_hal::layouts::Standard>(n.into(), base2k, (LOG_SCALE + 2 + width).into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(LOG_SCALE),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: crate::SlotLayout::Full,
        });
        let kernel_values = kernels
            .iter()
            .map(|kernel| {
                module.ckks_encode_coeffs_host_into(&mut pt, kernel, scratch)?;
                let (mut re, mut im) = (vec![F::zero(); n / 2], vec![F::zero(); n / 2]);
                module.ckks_decode_slots_host_into(&pt, &mut re, &mut im, scratch)?;
                Ok((re, im))
            })
            .collect::<Result<Vec<_>>>()?;
        // Before the last factor, block `B` holds the coset `bitrev(B)`.
        let bits = cosets.trailing_zeros();
        let bitrev = |b: usize| b.reverse_bits().checked_shr(usize::BITS - bits).unwrap_or(0);
        let values: Vec<_> = (0..cosets)
            .map(|b| kernel_values[bitrev(b) % kernels.len()].clone())
            .collect();
        let merged = crate::layouts::dft::ckks_block_scaled_last_factor::<BE, F>(
            module,
            base2k,
            dft.plan(),
            n.ilog2() as usize,
            &values,
            log_delta,
            scratch,
        )?;
        let [factor]: [_; 1] = module
            .ckks_dft_matrix_prepare_alloc(&merged, scratch)?
            .inner
            .factors
            .try_into()
            .unwrap_or_else(|_| unreachable!("one factor"));
        Ok(MergedS2CFactorPrepared(factor))
    }
}
