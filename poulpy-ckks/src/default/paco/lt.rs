//! The PaCo homomorphic factor chains, on the crate's official DFT
//! generator.
//!
//! PaCo's hybrid encoding treats each `2C`-wide slot block as an element of
//! `ℂ[Z]/(Z^{2C} − i)`: the evaluation points satisfy `x_k^{2C} = i` exactly
//! (`x_k = ζ^{5^{bitrev(k)}}`, `ζ = e^{2πi/8C}`), uniformly for every block
//! and chunk.
//!
//! - [`paco_c2s_factors`] — the **partial CoeffToSlot** chain (seqPaCo
//!   line 7): the `2C`-point Encode (IFFT) factorization of
//!   [`gen_dft_matrices_blockwise`](crate::default::dft) over the `n = 2hC`
//!   tile, returning each block's coefficients to the slots in **natural
//!   order**. Its product is exactly the inverse of the cleartext packing
//!   [`pack_chunk`](super::key::pack_chunk) — the encoder's own per-block
//!   FFT — pinned per backend by the `paco_packing` suite gate.
//! - [`paco_stc_factors`] — the **SlotToCoeff′** chain (lines 14–16 fused):
//!   the `C/2`-point Decode factorization with the η mask and the
//!   `Tr_{2C→C/2}` pair-packing fold absorbed into it — the routing/fold as
//!   the composed [`paco_pack_factor`], η's `1/π` as generator scaling, its
//!   dyadic `q/4` as the caller's scale relabel.
//!
//! The factor matrices are `m × m` with
//! `m ∈ {n, 2C, C/2}`, generally below the full `N/2` slots — the evaluation
//! layer **sparse-encodes** them (gap-mapped sub-ring plaintexts), which makes
//! the encoded diagonals act on the full slot vector exactly like
//! [`mul_vec_tiled`] (the cleartext oracle primitive).
//!
//! Grouping is fully caller-controlled (levels ↔ diagonals): each chain is
//! generated from an evaluation-order factorization schedule over its
//! **unit list** — `[L_1, …, L_{log 2C}, ψ]` for the c2s chain,
//! `[pack, L_1, …, L_{log C − 1}]` for the stc chain — where the fused
//! pseudo-layers (ψ/μ pair, pack) are first-class schedulable units (see
//! [`PaCoDFTPlan`](super::plan::PaCoDFTPlan)). Both chains are returned in
//! **application order** (first element is applied to the ciphertext first);
//! the chain product is grouping-invariant.

use poulpy_core::layouts::DiagonalArithmetic;

use crate::{
    default::dft::{DftScalar, gen_dft_matrices_blockwise},
    layouts::{ComplexDiagonals, dft::DFTType},
};
use crate::{
    encoding::paco::cpx::Cpx,
    layouts::paco::plan::{PaCoPlan, PaCoSlotOrder},
};

/// The grouped partial-CoeffToSlot **butterfly** factors (seqPaCo line 7), in
/// application order: the `2C`-point Encode (IFFT) factorization acting
/// block-locally on each `2C`-block of the `n = 2hC` tile, merged per the
/// evaluation-order `schedule` (which must sum to `log 2C`).
///
/// Applied (tiled) after the `Tr_{N/2→n}` fold, it turns each block's
/// evaluations back into its `ℂ[Z]/(Z^{2C} − i)` coefficients, in natural
/// order: slot `v·2C + j` holds coefficient `j` of block `v`. This is the raw
/// chain (no ψ/μ); the pipeline uses [`paco_psi_c2s_factors`].
pub(crate) fn paco_c2s_factors<F: DftScalar>(p: &PaCoPlan, schedule: &[usize]) -> Vec<ComplexDiagonals<F>> {
    let log_block = p.log_c() + 1;
    let log_tile = p.slots().trailing_zeros() as usize;
    assert_eq!(
        schedule.iter().sum::<usize>(),
        log_block,
        "c2s butterfly schedule {schedule:?} must sum to log 2C = {log_block}"
    );
    let factors = gen_dft_matrices_blockwise(DFTType::Encode, log_block, log_tile, schedule, 1.0, true);
    match p.slot_order() {
        PaCoSlotOrder::Natural => factors,
        // Conjugate every factor by the tiled low-bit reversal `Π`: the chain
        // becomes `(Π F_L Π)…(Π F_1 Π)` = `Π · chain · Π`, whose output is the
        // `BitRevLow`-relabeled mid-pipeline; the trailing input-side `Π` is
        // absorbed into the cleartext packing gather
        // ([`pack_chunk_into_with`](crate::layouts::paco::secret)), not
        // materialized here.
        PaCoSlotOrder::BitRevLow => factors.iter().map(|f| conjugate_by_low_bitrev(f, p.log_c() - 1)).collect(),
    }
}

/// Multiplies every stored diagonal value of `cd` by the real constant `s`
/// (already at the working precision `F` — never route irrational constants
/// through `f64`, it would cap an `f128` chain at 53 bits).
fn scale_factor<F: DftScalar>(cd: &mut ComplexDiagonals<F>, s: F) {
    for part in [&mut cd.re, &mut cd.im] {
        for i in part.indexes() {
            let scaled: Vec<F> = part.get(i).expect("indexed").iter().map(|&x| x * s).collect();
            part.set(i, scaled);
        }
    }
}

/// The ψ/μ tail of the partial-CoeffToSlot chain (seqPaCo lines 8–10), in
/// the cheapest form the schedule admits. See [`paco_psi_c2s_factors`].
pub(crate) enum PaCoPsiTail<F> {
    /// ψ merged with butterfly layers: the conjugation-augmented pair
    /// `(A, B)`, evaluated as `A·w + B·conj(w)` at one level.
    Pair([ComplexDiagonals<F>; 2]),
    /// ψ standing alone (last schedule entry `1`): one `2C`-tile of μ-mask
    /// values (the scaling share on the lower half, zero above), evaluated as
    /// `μ ⊙ (w + conj(rot_C(w)))` — a single fused conj-rotate keyswitch
    /// ([`conj_rotate_galois_element`](super::ops::conj_rotate_galois_element))
    /// followed by one plaintext multiplication. Same map and level count as
    /// the degenerate pair (`M_L = I`), at half the multiplications and one
    /// fewer keyswitch — the operation-lean layout.
    Mask(Vec<Cpx<F>>),
}

/// The partial-CoeffToSlot chain with seqPaCo lines 8–10 (ψ-pairing and the
/// μ mask) fused into its last factor.
///
/// `μ ⊙ (x + conj(rot_C(x)))` is antilinear in the chain output `x = M_L·w`
/// (conjugation is ℝ-linear, not ℂ-linear), so it cannot ride the factor
/// product as one more diagonal matrix; when ψ shares its schedule group with
/// butterfly layers it folds as the standard conjugation-augmented pair
/// instead:
///
/// ```text
/// μ ⊙ (M_L·w + conj(rot_C(M_L·w))) = A·w + B·conj(w)
/// A = D_μ · M_L,   B = D_μ · R_C · conj(M_L)
/// ```
///
/// (`D_μ` = the μ diagonal, `R_C` = the rotation-by-C matrix, both on the 2C
/// tile; `conj(M_L)` entry-wise.) Returns `(rest, tail)`: the leading factors
/// and a [`PaCoPsiTail`] replacing the last one. For the pair, the evaluator
/// applies `rest`, forms `conj(w)` with one plain conjugation keyswitch,
/// evaluates `A` on `w` and `B` on `conj(w)` at the same level, and adds —
/// the μ mask's level is folded away at the cost of a second diagonal matrix.
///
/// When ψ stands alone (last schedule entry `1`, so `M_L = I`) the pair
/// degenerates to `A = D_μ, B = D_μ·R_C` and the two matrices buy nothing;
/// the tail is emitted as [`PaCoPsiTail::Mask`] instead and the evaluator
/// runs the unfused fast sequence — `w += conj(rot_C(w))` as **one** fused
/// conj-rotate keyswitch, then one μ multiplication — the same map at the
/// same level count with strictly fewer operations. Schedules therefore span
/// the full speed↔depth trade: `[…, 1]` for the operation-lean paper layout,
/// ψ merged deeper to spend diagonals for a level.
///
/// The schedule (`p.c2s().factorization_depth()`, evaluation order) covers
/// the unit list `[L_1, …, L_{log 2C}, ψ]`. `dft.scaling()` is distributed
/// uniformly, one share per factor: the pair counts once (`A` and `B` both
/// get the last share), and the mask carries its share in its values.
pub(crate) fn paco_psi_c2s_factors<F: DftScalar + DiagonalArithmetic>(
    p: &PaCoPlan,
) -> (Vec<ComplexDiagonals<F>>, PaCoPsiTail<F>) {
    let dft = p.c2s();
    let log_block = p.log_c() + 1;
    debug_assert!(dft.check("c2s", log_block + 1).is_ok());
    let depth = dft.factorization_depth();
    let n_factors = depth.len();
    let psi_group_layers = depth[n_factors - 1] - 1;

    // Butterfly schedule = the schedule with the ψ unit stripped off the last
    // group (dropping the group when ψ stands alone).
    let mut butterfly = depth[..n_factors - 1].to_vec();
    if psi_group_layers > 0 {
        butterfly.push(psi_group_layers);
    }
    let mut factors = if butterfly.is_empty() {
        Vec::new()
    } else {
        paco_c2s_factors::<F>(p, &butterfly)
    };

    // One uniform share of the user scaling per factor; the tail is one
    // factor (A and B are added, so both carry the same share; the mask holds
    // its share directly). The root is taken at the working precision F after
    // converting the plan's f64 scaling value. The combined chain scaling is
    // validated as dyadic, but an individual per-chain root need not be.
    let share = F::from(dft.scaling()).unwrap().powf(F::one() / F::from(n_factors).unwrap());
    factors.iter_mut().for_each(|f| scale_factor(f, share));

    let two_c = 2 * p.c();
    if psi_group_layers == 0 {
        let zero = <F as num_traits::Zero>::zero();
        let mask: Vec<Cpx<F>> = (0..two_c)
            .map(|i| Cpx::new(if i < p.c() { share } else { zero }, zero))
            .collect();
        return (factors, PaCoPsiTail::Mask(mask));
    }
    let last = factors.pop().expect("non-empty butterfly schedule");

    // D_μ: 1 on the lower half of every 2C block.
    let mut mu = ComplexDiagonals::new(
        poulpy_core::layouts::Diagonals::<F>::new(two_c),
        poulpy_core::layouts::Diagonals::<F>::new(two_c),
    );
    let mut mu_diag = vec![<F as num_traits::Zero>::zero(); two_c];
    mu_diag[..p.c()].fill(F::one());
    mu.re.set(0, mu_diag);

    // R_C: slot rotation by C (`out[j] = v[j+C]`).
    let mut rot_c = ComplexDiagonals::new(
        poulpy_core::layouts::Diagonals::<F>::new(two_c),
        poulpy_core::layouts::Diagonals::<F>::new(two_c),
    );
    rot_c.re.set(p.c() as i64, vec![F::one(); two_c]);

    let a = mu.compose(&last);
    let mut conj_last = last;
    conj_last.conjugate();
    let b = mu.compose(&rot_c.compose(&conj_last));
    (factors, PaCoPsiTail::Pair([a, b]))
}

/// The grouped decomposed-SlotToCoeff factors (seqPaCo lines 14–16 fused), in
/// application order.
///
/// The chain consumes the **extraction output directly** (the 2C-periodic
/// `2i·sin(2π·m̃_j/q)` vector, live on the lower half of each 2C block): the
/// paper's η mask and `Tr_{2C→C/2}` packing fold are absorbed so neither
/// costs a level or a keyswitch of its own —
///
/// - the η routing pattern and the fold are [`paco_pack_factor`] (four
///   diagonals on the 2C tile), composed into the first-applied Decode factor
///   ([`ComplexDiagonals::compose`]);
/// - η's transcendental scalar `1/π` rides the generator's `scaling`
///   (distributed across the factors);
/// - η's dyadic scalar `q/4` is **not** in the chain at all: the caller
///   relabels the output scale by `log q − 2` bits (pure metadata — see
///   [`CKKSPaCoOps::ckks_paco_bootstrap_direct_into`](crate::api::CKKSPaCoOps::ckks_paco_bootstrap_direct_into)).
///
/// The Decode part is the `C/2`-point factorization. Its (packed) input is
/// `C/2`-periodic, so the dense mod-`C/2` canonical factors are exact when
/// applied tiled. The chain must map slot `j` to the **encoder's**
/// coefficient pair `j` (so the output polynomial carries `m̃_j` at
/// coefficient `j·N/C`); the encoder-convention (`bit_reversed = false`)
/// factorization pairs slot `j` with coefficient `bitrev(j)` instead, so the
/// input-side bit-reversal is absorbed into the first-applied **butterfly**
/// factor (`bit_reverse_columns`).
///
/// The schedule (`p.stc().factorization_depth()`, evaluation order) covers the
/// unit list `[pack, L_1, …, L_{log C − 1}]`: the first group always
/// contains the pack unit — composed on the group's input side, or emitted
/// as a standalone first factor when it stands alone (first entry `1`). Both
/// `dft.scaling()` and η's structural `1/π` are distributed uniformly, one
/// share per factor. At `C = 2` the unit list is `[pack]` alone.
pub(crate) fn paco_stc_factors<F: DftScalar + DiagonalArithmetic>(p: &PaCoPlan) -> Vec<ComplexDiagonals<F>> {
    let dft = p.stc();
    debug_assert!(dft.check("stc", p.log_c()).is_ok());
    let depth = dft.factorization_depth();
    let n_factors = depth.len();
    let pack_group_layers = depth[0] - 1;

    // Butterfly schedule = the schedule with the pack unit stripped off the
    // first group (dropping the group when the pack stands alone).
    let mut butterfly = depth[1..].to_vec();
    if pack_group_layers > 0 {
        butterfly.insert(0, pack_group_layers);
    }
    let mut factors = if butterfly.is_empty() {
        Vec::new()
    } else {
        let log_half_c = p.log_c() - 1;
        let mut f = gen_dft_matrices_blockwise(DFTType::Decode, log_half_c, log_half_c, &butterfly, 1.0, false);
        // The raw encoder-convention factorization consumes its coefficient
        // input in bit-reversed order. Under `Natural` the mid-pipeline data
        // is in natural order, so the reversal is folded into the first
        // factor's input side (growing it toward the dense ceiling `C/2`);
        // under `BitRevLow` the data already arrives `P`-ordered — on the
        // `C/2`-periodic packed input, `P` *is* the full input bit-reversal —
        // and the chain is used raw, with no folded permutation.
        if p.slot_order() == PaCoSlotOrder::Natural {
            f[0] = bit_reverse_columns(&f[0]);
        }
        f
    };
    let pack = paco_pack_factor::<F>(p);
    if pack_group_layers > 0 {
        factors[0] = factors[0].compose(&pack);
    } else {
        factors.insert(0, pack);
    }

    // One uniform share per factor of the user scaling and η's 1/π, computed
    // at the working precision F. π and the root do not add an f64 waypoint;
    // the caller-supplied plan scaling itself remains an f64 API value.
    let share = (F::from(dft.scaling()).unwrap() / F::PI()).powf(F::one() / F::from(n_factors).unwrap());
    factors.iter_mut().for_each(|f| scale_factor(f, share));
    factors
}

/// The fused η-routing + pair-packing map (seqPaCo lines 14–15, minus the
/// scalars), as four diagonals on the `2C` tile.
///
/// On the 2C-periodic extraction output `v` (live values `v_m = 2i·sin(2π·m̃_m/q)`
/// on each block's lower half `m ∈ [0, C)`, exact zeros above) it produces, at
/// **every** slot `j`, the `C/2`-periodic packed value
///
/// ```text
/// w_j = −i·v_m + v_{m+C/2},   m = j mod C/2
///     = 2·sin(2π·m̃_m/q) + 2i·sin(2π·m̃_{m+C/2}/q)
/// ```
///
/// i.e. η's `−i`-on-the-real-route / `1`-on-the-imaginary-route pattern and
/// the `Tr_{2C→C/2}` fold in one linear map. Per output quarter the two reads
/// land at different offsets, giving diagonals `{0, C/2, C, 3C/2}` with
/// quarter-piecewise values in `{−i, 1}` (exact — η's scalars are applied as
/// uniform per-factor shares by [`paco_stc_factors`]); every read hits the
/// live lower halves only.
pub(crate) fn paco_pack_factor<F: DftScalar>(p: &PaCoPlan) -> ComplexDiagonals<F> {
    let c = p.c();
    let tile = 2 * c;
    let quarter = c / 2;
    let s = F::one();
    let mut re = poulpy_core::layouts::Diagonals::<F>::new(tile);
    let mut im = poulpy_core::layouts::Diagonals::<F>::new(tile);
    // Quarter q_k = [k·C/2, (k+1)·C/2): the −i read (real route) sits at
    // offset (−k·C/2) mod 2C, the +1 read (imaginary route) C/2 further.
    for k in 0..4usize {
        let neg_i_idx = ((4 - k) % 4 * quarter) as i64;
        let one_idx = (neg_i_idx + quarter as i64).rem_euclid(tile as i64);
        let mut d_im = im.get(neg_i_idx).cloned().unwrap_or_else(|| vec![F::zero(); tile]);
        let mut d_re = re.get(one_idx).cloned().unwrap_or_else(|| vec![F::zero(); tile]);
        for j in k * quarter..(k + 1) * quarter {
            d_im[j] = -s;
            d_re[j] = s;
        }
        im.set(neg_i_idx, d_im);
        re.set(one_idx, d_re);
    }
    ComplexDiagonals::new(re, im)
}

/// `M'[r][c] = M[P(r)][P(c)]` with `P = ext_bitrev_low(·, log_p)` — conjugates
/// a factor by the tiled low-bit-reversal permutation (`M' = Π·M·Π` for the
/// self-inverse permutation matrix `Π` of `P`).
///
/// Entries move rows (`r → P(r)`) and offsets
/// (`i → (P((r+i) mod m) − P(r)) mod m`); a generic diagonal would scatter
/// across offsets, but for butterfly-structured factors the half-block support
/// keeps the diagonal count unchanged (paper 2025/886 Prop. 1): a level-`ℓ`
/// layer with offsets `{0, ±2^ℓ}` maps to `{0, ±2^{log_p−1−ℓ}}` for
/// `ℓ < log_p` and is invariant for `ℓ ≥ log_p`. `P`-invariant maps (the μ
/// mask, `R_C`, [`paco_pack_factor`]) are fixed points — pinned by the
/// `conjugate_by_low_bitrev_*` tests.
pub(crate) fn conjugate_by_low_bitrev<F: DftScalar>(cd: &ComplexDiagonals<F>, log_p: usize) -> ComplexDiagonals<F> {
    let m = cd.slots();
    debug_assert!(
        m >= (1usize << log_p),
        "factor dim {m} below permutation span {}",
        1usize << log_p
    );
    let p = |x: usize| super::ops::ext_bitrev_low(x, log_p);
    let mut re = poulpy_core::layouts::Diagonals::<F>::new(m);
    let mut im = poulpy_core::layouts::Diagonals::<F>::new(m);
    for i in cd.indexes() {
        let (dre, dim) = (cd.re.get(i), cd.im.get(i));
        for r in 0..m {
            let vre = dre.map_or_else(F::zero, |d| d[r]);
            let vim = dim.map_or_else(F::zero, |d| d[r]);
            if vre == F::zero() && vim == F::zero() {
                continue;
            }
            let c_old = (r as i64 + i).rem_euclid(m as i64) as usize;
            let (r_new, c_new) = (p(r), p(c_old));
            let idx = ((c_new + m - r_new) % m) as i64;
            if vre != F::zero() {
                let mut d = re.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); m]);
                d[r_new] = d[r_new] + vre;
                re.set(idx, d);
            }
            if vim != F::zero() {
                let mut d = im.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); m]);
                d[r_new] = d[r_new] + vim;
                im.set(idx, d);
            }
        }
    }
    ComplexDiagonals::new(re, im)
}

/// `M'[r][c] = M[r][bitrev(c)]` — composes a factor with the bit-reversal
/// permutation on its input side (`M' = M · Π_bitrev`).
fn bit_reverse_columns<F: DftScalar>(cd: &ComplexDiagonals<F>) -> ComplexDiagonals<F> {
    let m = cd.slots();
    let log_m = m.trailing_zeros();
    let br = |x: usize| (x as u32).reverse_bits() as usize >> (u32::BITS - log_m);
    let mut re = poulpy_core::layouts::Diagonals::<F>::new(m);
    let mut im = poulpy_core::layouts::Diagonals::<F>::new(m);
    for i in cd.indexes() {
        let (dre, dim) = (cd.re.get(i), cd.im.get(i));
        for r in 0..m {
            let c_old = (r + i as usize) % m;
            let c_new = br(c_old);
            let idx = ((c_new + m - r) % m) as i64;
            let vre = dre.map_or(F::zero(), |d| d[r]);
            let vim = dim.map_or(F::zero(), |d| d[r]);
            if vre != F::zero() {
                let mut d = re.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); m]);
                d[r] = d[r] + vre;
                re.set(idx, d);
            }
            if vim != F::zero() {
                let mut d = im.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); m]);
                d[r] = d[r] + vim;
                im.set(idx, d);
            }
        }
    }
    ComplexDiagonals::new(re, im)
}

/// Tiled matrix–vector product against a vector of length a multiple of the
/// matrix dimension `m`, with period-`m` diagonals and full-length rotations:
/// `out[j] = Σ_i diag_i[j mod m] · v[(j+i) mod v.len()]`.
///
/// This is exactly how a sparse-encoded `m × m` [`ComplexDiagonals`] acts on
/// the full `N/2`-slot vector of a ciphertext — the cleartext oracle primitive
/// of the PaCo pipeline.
// Consumed by the (feature-gated) `test_suite::paco_lt` reference checks.
#[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
pub(crate) fn mul_vec_tiled<F: DftScalar>(cd: &ComplexDiagonals<F>, v: &[Cpx<F>]) -> Vec<Cpx<F>> {
    let m = cd.slots();
    let len = v.len();
    debug_assert!(len.is_multiple_of(m));
    let mut out = vec![Cpx::zero(); len];
    for i in cd.indexes() {
        let re = cd.re.get(i);
        let im = cd.im.get(i);
        for (j, o) in out.iter_mut().enumerate() {
            let d = Cpx::new(re.map_or_else(F::zero, |d| d[j % m]), im.map_or_else(F::zero, |d| d[j % m]));
            *o = *o + d * v[(j as i64 + i).rem_euclid(len as i64) as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(depth: Vec<usize>, scaling: f64) -> crate::layouts::PaCoDFTPlan {
        let giant_steps = vec![1; depth.len()];
        crate::layouts::PaCoDFTPlan::new(depth, giant_steps, 0, 0, scaling).unwrap()
    }

    fn with_stc(p: &PaCoPlan, depth: Vec<usize>, scaling: f64) -> PaCoPlan {
        p.clone()
            .with_evaluation(p.log_delta_bsk(), p.log_beta_budget(), p.c2s().clone(), dft(depth, scaling))
            .unwrap()
    }

    fn with_c2s(p: &PaCoPlan, depth: Vec<usize>, scaling: f64) -> PaCoPlan {
        p.clone()
            .with_evaluation(p.log_delta_bsk(), p.log_beta_budget(), dft(depth, scaling), p.stc().clone())
            .unwrap()
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<Cpx> {
        (0..n)
            .map(|i| {
                let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed)) as f64 / u64::MAX as f64;
                let y = ((i as u64 + 31).wrapping_mul(1442695040888963407).wrapping_add(seed)) as f64 / u64::MAX as f64;
                Cpx::new(2.0 * x - 1.0, 2.0 * y - 1.0)
            })
            .collect()
    }

    /// Balanced schedules; C = 2 stc is the pack alone; grouped and merged
    /// butterfly chains have identical products.
    #[test]
    fn grouping_shapes() {
        let p = PaCoPlan::new(9, 8, 4, 29).unwrap(); // log C = 2: log 2C = 3 butterfly layers
        assert_eq!(paco_c2s_factors::<f64>(&p, &[1, 1, 1]).len(), 3);
        assert_eq!(paco_c2s_factors::<f64>(&p, &[2, 1]).len(), 2);
        assert_eq!(paco_c2s_factors::<f64>(&p, &[3]).len(), 1);
        assert_eq!(paco_stc_factors::<f64>(&with_stc(&p, vec![2], 1.0)).len(), 1);
        let p2 = PaCoPlan::new(9, 8, 2, 29).unwrap(); // C = 2: no Decode part, pack factor alone
        assert_eq!(paco_stc_factors::<f64>(&with_stc(&p2, vec![1], 1.0)).len(), 1);

        let v = rand_vec(p.slots(), 9);
        let mut a = v.clone();
        for m in paco_c2s_factors::<f64>(&p, &[1, 1, 1]) {
            a = mul_vec_tiled(&m, &a);
        }
        let mut b = v;
        for m in paco_c2s_factors::<f64>(&p, &[2, 1]) {
            b = mul_vec_tiled(&m, &b);
        }
        for j in 0..a.len() {
            assert!((a[j] - b[j]).abs() < 1e-11, "slot {j}");
        }
    }

    /// Builds the extraction-layout tile the StC chain consumes: `v_m = 2i·y_m`
    /// on the lower half `m ∈ [0, C)`, exact zeros above.
    fn extraction_tile(y: &[f64], c: usize) -> Vec<Cpx> {
        let mut v = vec![Cpx::ZERO; 2 * c];
        for (m, &ym) in y.iter().enumerate() {
            v[m] = Cpx::new(0.0, 2.0 * ym);
        }
        v
    }

    /// Pins the StC chain to the encoder basis: on the extraction layout, the
    /// fused pack routes `w_j = 2·y_j + 2i·y_{j+C/2}` and the Decode part maps
    /// chain(v)[k] = (1/π)·Σ_j w_j·ξ^{5^k·j} with ξ = e^{2πi/(4·C/2)} — the
    /// encoder's decode matrix of the `C/2` sub-ring scaled by η's folded 1/π,
    /// so the output polynomial carries `m̃_j·(4/q)` at coefficient `j·N/C`
    /// (the `q/4` is the caller's scale relabel).
    #[test]
    fn stc_chain_is_encoder_vandermonde() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap(); // C = 8, C/2 = 4
        let (c, half_c) = (p.c(), p.c() / 2);
        let y: Vec<f64> = (0..c).map(|i| ((i * i + 3 * i + 1) % 17) as f64 / 8.5 - 1.0).collect();
        let v = extraction_tile(&y, c);
        let w: Vec<Cpx> = (0..half_c).map(|j| Cpx::new(2.0 * y[j], 2.0 * y[j + half_c])).collect();
        for schedule in [vec![1, 1, 1], vec![2, 1], vec![1, 2], vec![3]] {
            let g1 = format!("{schedule:?}");
            let mut got = v.clone();
            for m in paco_stc_factors::<f64>(&with_stc(&p, schedule, 1.0)) {
                got = mul_vec_tiled(&m, &got);
            }
            let xi = |e: usize| {
                let theta = 2.0 * std::f64::consts::PI * (e as f64) / ((4 * half_c) as f64);
                Cpx::new(theta.cos(), theta.sin())
            };
            let mut pow5 = 1usize;
            for (k, &got_k) in got.iter().take(half_c).enumerate() {
                let mut want = Cpx::ZERO;
                for (j, &wj) in w.iter().enumerate() {
                    want = want + wj * xi((pow5 * j) % (4 * half_c));
                }
                want = want * Cpx::new(std::f64::consts::FRAC_1_PI, 0.0);
                assert!((got_k - want).abs() < 1e-11, "g1={g1} slot {k}");
                pow5 = (pow5 * 5) % (4 * half_c);
            }
        }
    }

    /// Both ψ/μ tail forms equal the step-by-step tail: for any input,
    /// `rest → A·w + B·conj(w)` (pair) and
    /// `rest → w + conj(rot_C(w)) → μ ⊙ ·` (mask) must match
    /// `rest → M_L → μ ⊙ (x + conj(rot_C(x)))`.
    #[test]
    fn psi_c2s_tail_matches_step_by_step() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap();
        let (n, c) = (p.slots(), p.c());
        let v = rand_vec(n, 41);
        // Schedules over the 5-unit list [L1..L4, ψ]: ψ merged deep, shallow,
        // balanced, and standing alone (the operation-lean mask tail).
        for schedule in [vec![1, 1, 1, 1, 1], vec![2, 3], vec![4, 1], vec![1, 4], vec![5]] {
            let g0 = format!("{schedule:?}");
            let psi_alone = *schedule.last().unwrap() == 1;
            // Step-by-step: full raw butterfly chain, then ψ-pairing, then μ.
            let mut x = v.clone();
            for m in paco_c2s_factors::<f64>(&p, &[1, 1, 1, 1]) {
                x = mul_vec_tiled(&m, &x);
            }
            let want: Vec<Cpx> = (0..n)
                .map(|j| {
                    if j % (2 * c) < c {
                        x[j] + x[(j + c) % n].conj()
                    } else {
                        Cpx::ZERO
                    }
                })
                .collect();

            let (rest, tail) = paco_psi_c2s_factors::<f64>(&with_c2s(&p, schedule, 1.0));
            let mut w = v.clone();
            for m in &rest {
                w = mul_vec_tiled(m, &w);
            }
            let got: Vec<Cpx> = match tail {
                PaCoPsiTail::Pair([a, b]) => {
                    assert!(!psi_alone, "g0={g0}: ψ alone must emit the mask tail");
                    let cw: Vec<Cpx> = w.iter().map(|z| z.conj()).collect();
                    let got_a = mul_vec_tiled(&a, &w);
                    let got_b = mul_vec_tiled(&b, &cw);
                    got_a.iter().zip(&got_b).map(|(&x, &y)| x + y).collect()
                }
                PaCoPsiTail::Mask(mask) => {
                    assert!(psi_alone, "g0={g0}: merged ψ must emit the pair tail");
                    assert_eq!(mask.len(), 2 * c);
                    (0..n).map(|j| (w[j] + w[(j + c) % n].conj()) * mask[j % (2 * c)]).collect()
                }
            };
            for j in 0..n {
                assert!((got[j] - want[j]).abs() < 1e-11, "g0={g0} slot {j}");
            }
        }
    }

    /// `(Πv)[j] = v[P(j)]` — the vector action of the tiled low-bit-reversal.
    fn permute_low(v: &[Cpx], log_p: usize) -> Vec<Cpx> {
        (0..v.len()).map(|j| v[super::super::ops::ext_bitrev_low(j, log_p)]).collect()
    }

    /// Semantic equality of two factors: same values on every offset of
    /// either, treating an absent re/im part as all-zeros (generators may
    /// store explicitly-zero parts that a rebuild omits).
    fn assert_cd_eq(a: &ComplexDiagonals<f64>, b: &ComplexDiagonals<f64>, ctx: &str) {
        assert_eq!(a.slots(), b.slots(), "{ctx}: slots");
        let m = a.slots();
        let mut offsets = a.indexes();
        offsets.extend(b.indexes());
        offsets.sort_unstable();
        offsets.dedup();
        let zeros = vec![0.0; m];
        for i in offsets {
            for (pa, pb, part) in [(&a.re, &b.re, "re"), (&a.im, &b.im, "im")] {
                let x = pa.get(i).unwrap_or(&zeros);
                let y = pb.get(i).unwrap_or(&zeros);
                assert_eq!(x, y, "{ctx}: diag {i} {part}");
            }
        }
    }

    /// `ext_bitrev_low` is a self-inverse permutation of the low bits only,
    /// and the identity for `log_p ∈ {0, 1}`.
    #[test]
    fn ext_bitrev_low_involution() {
        use super::super::ops::ext_bitrev_low;
        for log_p in 0..=4usize {
            for x in 0..64usize {
                let y = ext_bitrev_low(x, log_p);
                assert_eq!(ext_bitrev_low(y, log_p), x, "log_p={log_p} x={x}");
                assert_eq!(y >> log_p, x >> log_p, "log_p={log_p} x={x}: high bits");
                if log_p <= 1 {
                    assert_eq!(y, x, "log_p={log_p} must be the identity");
                }
            }
        }
    }

    /// Action pin: `Π·M·Π` applied to `Πv` equals `Π(M·v)` — the conjugated
    /// factor is the same map in the relabeled basis, tiled over the full
    /// slot vector.
    #[test]
    fn conjugate_by_low_bitrev_action() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap(); // C = 8, log(C/2) = 2
        let log_p = p.log_c() - 1;
        let v = rand_vec(p.slots(), 7);
        let pv = permute_low(&v, log_p);
        for schedule in [vec![1, 1, 1, 1], vec![2, 2], vec![3, 1], vec![4]] {
            for (idx, m) in paco_c2s_factors::<f64>(&p, &schedule).iter().enumerate() {
                let mc = conjugate_by_low_bitrev(m, log_p);
                let want = permute_low(&mul_vec_tiled(m, &v), log_p);
                let got = mul_vec_tiled(&mc, &pv);
                for j in 0..v.len() {
                    assert!((got[j] - want[j]).abs() < 1e-11, "{schedule:?}[{idx}] slot {j}");
                }
            }
        }
    }

    /// Prop-1 offset pin on single-layer factors: diagonal count is preserved;
    /// a level with offset magnitude `2^ℓ < C/2` moves to `2^{log_p−1−ℓ}`, a
    /// level with `2^ℓ ≥ C/2` is a fixed point (structurally unchanged).
    #[test]
    fn conjugate_by_low_bitrev_prop1_offsets() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap(); // C = 8: layer magnitudes 1, 2, 4, 8
        let log_p = p.log_c() - 1; // 2
        for m in paco_c2s_factors::<f64>(&p, &[1, 1, 1, 1]) {
            let dim = m.slots() as i64;
            let magnitude = |cd: &ComplexDiagonals<f64>| -> i64 {
                cd.indexes().iter().map(|&i| i.min((dim - i).rem_euclid(dim))).max().unwrap()
            };
            let mag = magnitude(&m);
            let mc = conjugate_by_low_bitrev(&m, log_p);
            assert_eq!(mc.indexes().len(), m.indexes().len(), "count at magnitude {mag}");
            let l = mag.trailing_zeros() as usize;
            if l >= log_p {
                assert_cd_eq(&m, &mc, &format!("level {l} (≥ log_p)"));
            } else {
                let want = 1i64 << (log_p - 1 - l);
                assert_eq!(magnitude(&mc), want, "level {l}: magnitude");
            }
        }
    }

    /// The `P`-invariant maps of the commutation table are fixed points of the
    /// conjugation: `paco_pack_factor`, the μ mask, and the rotation by C.
    #[test]
    fn conjugate_by_low_bitrev_invariants() {
        use poulpy_core::layouts::Diagonals;
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap();
        let log_p = p.log_c() - 1;
        let two_c = 2 * p.c();

        let pack = paco_pack_factor::<f64>(&p);
        assert_cd_eq(&pack, &conjugate_by_low_bitrev(&pack, log_p), "pack");

        let mut mu = ComplexDiagonals::new(Diagonals::<f64>::new(two_c), Diagonals::<f64>::new(two_c));
        let mut d = vec![0.0; two_c];
        d[..p.c()].fill(1.0);
        mu.re.set(0, d);
        assert_cd_eq(&mu, &conjugate_by_low_bitrev(&mu, log_p), "mu");

        let mut rc = ComplexDiagonals::new(Diagonals::<f64>::new(two_c), Diagonals::<f64>::new(two_c));
        rc.re.set(p.c() as i64, vec![1.0; two_c]);
        assert_cd_eq(&rc, &conjugate_by_low_bitrev(&rc, log_p), "rot_C");
    }

    /// Measured (plain, conjugated) diagonal counts per merged c2s factor
    /// shape. Conjugation preserves the count of every shape except sparse
    /// merges spanning the `log(C/2)` boundary, where the plain offsets'
    /// arithmetic collisions (`4 − 8 = −4`) are not reproduced by the moved
    /// offsets (`{0,±1}∘{0,±8}` has none) — the C=16 `[2,2,1]` middle factor
    /// pays +2. Fully merged shapes are at their dense ladder count either
    /// way. These pins feed the `BitRevLow` cost guidance (see
    /// `docs/spec/paco_dft_convention.md`).
    #[test]
    fn conjugate_by_low_bitrev_counts() {
        for (log_n, c, schedule, want) in [
            (9usize, 8usize, vec![1, 1, 1, 1], vec![(3, 3), (3, 3), (3, 3), (3, 3)]),
            (9, 8, vec![2, 2], vec![(7, 7), (7, 7)]),
            (9, 8, vec![2, 1, 1], vec![(7, 7), (3, 3), (3, 3)]),
            (9, 8, vec![3, 1], vec![(15, 15), (3, 3)]),
            (9, 8, vec![4], vec![(31, 31)]),
            (10, 16, vec![1, 1, 1, 1, 1], vec![(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]),
            (10, 16, vec![2, 2, 1], vec![(7, 7), (7, 9), (3, 3)]),
            (10, 16, vec![3, 2], vec![(15, 15), (7, 7)]),
            (10, 16, vec![5], vec![(63, 63)]),
        ] {
            let p = PaCoPlan::new(log_n, 8, c, 29).unwrap();
            let log_p = p.log_c() - 1;
            let counts: Vec<(usize, usize)> = paco_c2s_factors::<f64>(&p, &schedule)
                .iter()
                .map(|m| (m.indexes().len(), conjugate_by_low_bitrev(m, log_p).indexes().len()))
                .collect();
            assert_eq!(counts, want, "C={c} {schedule:?}");
        }
    }

    /// `BitRevLow` c2s chain (with the ψ/μ tail) is the `Natural` chain in the
    /// relabeled basis: `chain_brl(Πv) = Π·chain_nat(v)`, for ψ merged (Pair)
    /// and standalone (Mask) schedules.
    #[test]
    fn bitrevlow_c2s_chain_is_relabeled_natural() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap(); // C = 8, log_p = 2
        let log_p = p.log_c() - 1;
        let v = rand_vec(p.slots(), 23);
        let pv = permute_low(&v, log_p);
        for schedule in [vec![1, 1, 1, 1, 1], vec![2, 3], vec![4, 1], vec![5]] {
            let g0 = format!("{schedule:?}");
            let run = |plan: &PaCoPlan, input: &[Cpx]| -> Vec<Cpx> {
                let (rest, tail) = paco_psi_c2s_factors::<f64>(&with_c2s(plan, schedule.clone(), 1.0));
                let mut w = input.to_vec();
                for m in &rest {
                    w = mul_vec_tiled(m, &w);
                }
                let n = w.len();
                let c = plan.c();
                match tail {
                    PaCoPsiTail::Pair([a, b]) => {
                        let cw: Vec<Cpx> = w.iter().map(|z| z.conj()).collect();
                        let ga = mul_vec_tiled(&a, &w);
                        let gb = mul_vec_tiled(&b, &cw);
                        ga.iter().zip(&gb).map(|(&x, &y)| x + y).collect()
                    }
                    PaCoPsiTail::Mask(mask) => (0..n).map(|j| (w[j] + w[(j + c) % n].conj()) * mask[j % (2 * c)]).collect(),
                }
            };
            let want = permute_low(&run(&p, &v), log_p);
            let got = run(&p.clone().with_slot_order(crate::layouts::PaCoSlotOrder::BitRevLow), &pv);
            for j in 0..v.len() {
                assert!((got[j] - want[j]).abs() < 1e-11, "g0={g0} slot {j}");
            }
        }
    }

    /// `BitRevLow` StC′ chain consumes the `Π`-relabeled extraction layout and
    /// produces the same natural-order output as the `Natural` chain — and its
    /// first factor carries **no folded permutation**: at `C = 32, g1 = 1` the
    /// first butterfly factor drops from the folded 16 diagonals (the dense
    /// `C/2` ceiling) to the raw 3.
    #[test]
    fn bitrevlow_stc_chain_unfolded() {
        let p = PaCoPlan::new(10, 8, 32, 29).unwrap(); // C = 32: stc units [pack, L1..L4]
        let p_brl = p.clone().with_slot_order(crate::layouts::PaCoSlotOrder::BitRevLow);
        let log_p = p.log_c() - 1;
        let y: Vec<f64> = (0..p.c()).map(|i| ((11 * i + 3) % 23) as f64 / 11.5 - 1.0).collect();
        let v = extraction_tile(&y, p.c());
        for schedule in [vec![1, 1, 1, 1, 1], vec![2, 2, 1], vec![5]] {
            let g1 = format!("{schedule:?}");
            let nat = paco_stc_factors::<f64>(&with_stc(&p, schedule.clone(), 1.0));
            let brl = paco_stc_factors::<f64>(&with_stc(&p_brl, schedule.clone(), 1.0));

            let mut want = v.clone();
            for m in &nat {
                want = mul_vec_tiled(m, &want);
            }
            let mut got = permute_low(&v, log_p);
            for m in &brl {
                got = mul_vec_tiled(m, &got);
            }
            for j in 0..v.len() {
                assert!((got[j] - want[j]).abs() < 1e-11, "g1={g1} slot {j}");
            }

            // The payoff, as measured pins: the factor carrying the fold under
            // Natural (first butterfly factor, or the pack-merged group)
            // returns to its raw sparsity under BitRevLow; fully merged chains
            // sit at the dense ceiling either way.
            let (nat_counts, brl_counts): (Vec<usize>, Vec<usize>) = (
                nat.iter().map(|m| m.indexes().len()).collect(),
                brl.iter().map(|m| m.indexes().len()).collect(),
            );
            let (want_nat, want_brl): (Vec<usize>, Vec<usize>) = match schedule.as_slice() {
                [1, 1, 1, 1, 1] => (vec![4, 14, 3, 3, 2], vec![4, 3, 3, 3, 2]),
                [2, 2, 1] => (vec![56, 7, 2], vec![12, 7, 2]),
                [5] => (vec![64], vec![64]),
                _ => unreachable!(),
            };
            assert_eq!(nat_counts, want_nat, "g1={g1} natural");
            assert_eq!(brl_counts, want_brl, "g1={g1} bitrevlow");
        }
    }

    /// Grouped and merged StC chains agree on the extraction layout (the
    /// fused first factor is a 2C tile; the remaining mod-`C/2` canonical
    /// factors see its `C/2`-periodic output).
    #[test]
    fn stc_chain_matches_dense_decode() {
        let p = PaCoPlan::new(9, 8, 8, 29).unwrap(); // C = 8: fused factor + 1 Decode factor at g1 = 1
        let y: Vec<f64> = (0..p.c()).map(|i| ((7 * i + 5) % 13) as f64 / 6.5 - 1.0).collect();
        let v = extraction_tile(&y, p.c());
        let mut got = v.clone();
        for m in paco_stc_factors::<f64>(&with_stc(&p, vec![2, 1], 1.0)) {
            got = mul_vec_tiled(&m, &got);
        }
        let mut want = v.clone();
        for m in paco_stc_factors::<f64>(&with_stc(&p, vec![3], 1.0)) {
            want = mul_vec_tiled(&m, &want);
        }
        for j in 0..v.len() {
            assert!((got[j] - want[j]).abs() < 1e-11, "slot {j}");
        }
    }
}
