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
        let slots = k * r;
        let half = module.ring_degree().as_usize() / 2;
        ensure!(slots <= half, "{k} values of period {r} need more than the {half} slots");
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
        let mut diagonal = |slot_layout: SlotLayout, weights: &[F]| -> Result<CKKSPlaintextOwned<BE>> {
            // Room for the weights, at most `R`, over the diagonal scale.
            let mut pt = module.ckks_pt_vec_alloc(
                weights.len(),
                base2k,
                TorusPrecision((log_delta_diag + layout.log_r() + 2) as u32),
            );
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(log_delta_diag),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout,
            });
            let zeros = vec![F::zero(); weights.len()];
            module.ckks_encode_reim_slots_into(&mut pt, weights, &zeros, &mut scratch.borrow())?;
            // The diagonals multiply a ciphertext of the full layout.
            pt.set_slot_layout(SlotLayout::Full);
            Ok(pt)
        };

        let to = (0..k)
            .map(|o| {
                let weights: Vec<F> = (0..slots)
                    .map(|s| {
                        let read = s % r == 0 && (s / r) % k == (s + o) % k;
                        if read { F::one() } else { F::zero() }
                    })
                    .collect();
                Ok((o as i64, diagonal(SlotLayout::Replicated(layout), &weights)?))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut from = Vec::new();
        if r <= k {
            let weight = F::from_usize(r).unwrap();
            let mut gathered: Vec<(usize, Vec<F>)> = Vec::new();
            for s in 0..slots {
                let (c, i) = (s % k, s / k);
                for u in 0..k / r {
                    let o = (r * u + slots - c % r) % slots;
                    // The data slot read by offset `o` at `s` carries value `m`.
                    let m = (c / r + (k / r) * i + u) % k;
                    if m != c {
                        continue;
                    }
                    let index = match gathered.iter().position(|(offset, _)| *offset == o) {
                        Some(index) => index,
                        None => {
                            gathered.push((o, vec![F::zero(); slots]));
                            gathered.len() - 1
                        }
                    };
                    gathered[index].1[s] = weight;
                }
            }
            gathered.sort_by_key(|(o, _)| *o);
            from = gathered
                .into_iter()
                .map(|(o, weights)| Ok((o as i64, diagonal(SlotLayout::Full, &weights)?)))
                .collect::<Result<Vec<_>>>()?;
        }
        let h = half / k;
        let trace = (0..h.trailing_zeros()).map(|l| (k << l) as i64).collect();
        Ok(Self {
            layout,
            log_delta,
            extra_bits,
            to,
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
