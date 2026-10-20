//! Diagonals of the conversions between CKKS and a replicated slot layout.

use anyhow::{Result, ensure};
use poulpy_core::layouts::{Base2K, GetDegree, TorusPrecision};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
};

use crate::{
    GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
};

/// Diagonals of the conversions between a CKKS ciphertext of period `k` and
/// a [`ReplicatedLayout`] of `k` values, each one linear transform.
///
/// With `n = kR` slots and `h = N/(2k)`:
///
/// - `to`: `k` diagonals writing value `z_(t/R mod k)` on the data slots `R | t`
///   and zero on the guards, quantized with shaping;
/// - `from` (when `R | k`): the gather diagonals, of weight `R`, reading only
///   data slots so that the normalized trace onto period `k` returns the
///   average of the `h/R` copies of each value, followed by `log2 h` trace
///   rotations by `k, 2k, ..., (h/2)k`.
///
/// Both conversions compute `extra_bits` bits under the scale and consume the
/// diagonals' `log_delta_diag` bits of budget. `log_delta` is the scale of the
/// replicated side.
pub struct ReplicatedConversion<BE: Backend> {
    pub(crate) layout: ReplicatedLayout,
    pub(crate) log_delta: usize,
    pub(crate) extra_bits: usize,
    pub(crate) to: Vec<(i64, CKKSPlaintextOwned<BE>)>,
    pub(crate) from: Vec<(i64, CKKSPlaintextOwned<BE>)>,
    pub(crate) trace: Vec<i64>,
}

impl<BE: Backend> ReplicatedConversion<BE> {
    pub fn new<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        extra_bits: usize,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: CKKSEncodingOps<BE, F> + CKKSModuleAlloc<BE> + GetDegree,
    {
        let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
        let select = |s: usize, j: usize| {
            let read = s.is_multiple_of(r) && (s / r) % k == j;
            (if read { F::one() } else { F::zero() }, F::zero())
        };
        let to = ReplicatedTransform::from_period(module, layout, base2k, log_delta_diag, true, select)?;
        let identity = |c: usize, j: usize| (if c == j { F::one() } else { F::zero() }, F::zero());
        let from = if r <= k {
            ReplicatedTransform::gather(module, layout, base2k, log_delta_diag, false, identity)?.diagonals
        } else {
            Vec::new()
        };
        let h = module.ring_degree().as_usize() / (2 * k);
        let trace = (0..h.trailing_zeros()).map(|l| (k << l) as i64).collect();
        Ok(Self {
            layout,
            log_delta,
            extra_bits,
            to: to.diagonals,
            from,
            trace,
        })
    }

    pub fn layout(&self) -> ReplicatedLayout {
        self.layout
    }

    /// Slot rotations the conversions use, for their automorphism keys.
    pub fn rotations(&self) -> Vec<i64> {
        let mut rotations: Vec<i64> = self
            .to
            .iter()
            .chain(&self.from)
            .map(|(o, _)| *o)
            .chain(self.trace.iter().copied())
            .collect();
        rotations.retain(|&o| o != 0);
        rotations.sort_unstable();
        rotations.dedup();
        rotations
    }
}

/// Diagonals `(o, D_o)` of a linear map `y = Σ_o D_o ⊙ rot_o(x)` on the
/// `n = kR` slots of a replicated layout's period, `k` of them whatever the map.
pub struct ReplicatedTransform<BE: Backend> {
    pub(crate) diagonals: Vec<(i64, CKKSPlaintextOwned<BE>)>,
}

impl<BE: Backend> ReplicatedTransform<BE> {
    /// The map `y_s = Σ_(j<k) a(s, j)·x_j` from slots of period `k` to the `n`
    /// slots: `D_o[s] = a(s, (s + o) mod k)` for `0 <= o < k`. `shaped`
    /// quantizes the diagonals with the layout's shaping.
    pub fn from_period<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: CKKSEncodingOps<BE, F> + CKKSModuleAlloc<BE> + GetDegree,
    {
        let (k, n) = check_period(module, &layout)?;
        let weights = (0..k).map(|o| (o, (0..n).map(|s| a(s, (s + o) % k)).collect())).collect();
        Self::encode(module, layout, base2k, log_delta_diag, shaped, weights)
    }

    /// The gather `y` reading only the data slots, such that the normalized
    /// trace onto period `k`, which averages the `R` positions `c + k·i` of
    /// fiber `c`, returns `(Az)_c = Σ_j a(c, j)·z_j` for the data values `z`.
    /// Offset `o = R·u − (c mod R)` at `c + k·i` reads data value
    /// `m = ⌊c/R⌋ + (k/R)·i + u mod k`, weighted `R·a(c, m)`. Needs `R | k`.
    pub fn gather<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: CKKSEncodingOps<BE, F> + CKKSModuleAlloc<BE> + GetDegree,
    {
        let (k, n) = check_period(module, &layout)?;
        let r = 1usize << layout.log_r();
        ensure!(r <= k, "a gather needs the period {r} to divide the batch size {k}");
        let weight = F::from_usize(r).unwrap();
        let mut weights: Vec<(usize, Vec<(F, F)>)> = Vec::new();
        for s in 0..n {
            let (c, i) = (s % k, s / k);
            for u in 0..k / r {
                let o = (r * u + n - c % r) % n;
                let m = (c / r + (k / r) * i + u) % k;
                let (re, im) = a(c, m);
                let index = match weights.iter().position(|(offset, _)| *offset == o) {
                    Some(index) => index,
                    None => {
                        weights.push((o, vec![(F::zero(), F::zero()); n]));
                        weights.len() - 1
                    }
                };
                weights[index].1[s] = (weight * re, weight * im);
            }
        }
        weights.sort_by_key(|(o, _)| *o);
        Self::encode(module, layout, base2k, log_delta_diag, shaped, weights)
    }

    /// Slot rotations of the diagonals, for their automorphism keys.
    pub fn rotations(&self) -> Vec<i64> {
        self.diagonals.iter().map(|(o, _)| *o).filter(|&o| o != 0).collect()
    }

    fn encode<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        weights: Vec<(usize, Vec<(F, F)>)>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: CKKSEncodingOps<BE, F> + CKKSModuleAlloc<BE>,
    {
        let n = weights.first().map_or(0, |(_, w)| w.len());
        let largest = weights
            .iter()
            .flat_map(|(_, w)| w.iter().map(|(re, im)| re.abs().max(im.abs())))
            .fold(F::zero(), F::max);
        // Bits above the scale for the largest weight, plus a sign and a carry.
        let headroom = (usize::BITS - (largest.to_f64().unwrap_or(1.0).ceil() as usize).max(1).leading_zeros()) as usize + 2;
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(n));
        let label = if shaped {
            SlotLayout::Replicated(layout)
        } else {
            SlotLayout::Full
        };
        let diagonals = weights
            .into_iter()
            .map(|(o, w)| {
                let mut pt = module.ckks_pt_vec_alloc(n, base2k, TorusPrecision((log_delta_diag + headroom) as u32));
                pt.set_meta(GLWEPlaintextMeta {
                    scale: Scale::Log(log_delta_diag),
                    slots: SlotsKind::Complex,
                    log_sparsity: 0,
                    slot_layout: label,
                });
                let (re, im): (Vec<F>, Vec<F>) = w.into_iter().unzip();
                module.ckks_encode_reim_slots_into(&mut pt, &re, &im, &mut scratch.borrow())?;
                // The diagonals multiply a ciphertext of the full layout.
                pt.set_slot_layout(SlotLayout::Full);
                Ok((o as i64, pt))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { diagonals })
    }
}

/// `(k, kR)` of a layout that fits the ring's slots.
fn check_period<BE: Backend>(module: &Module<BE>, layout: &ReplicatedLayout) -> Result<(usize, usize)>
where
    Module<BE>: GetDegree,
{
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let half = module.ring_degree().as_usize() / 2;
    ensure!(k * r <= half, "{k} values of period {r} need more than the {half} slots");
    Ok((k, k * r))
}
