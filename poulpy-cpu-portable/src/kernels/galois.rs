//! The stored slot a convolution reads for each destination slot of its right operand, through a sparse
//! embedding and a Galois automorphism, on the bit-reversed transforms of the CPU backends.

use poulpy_hal::layouts::Ring;

use super::sparse_log_gap_portable;

/// `σ_p` on the first `stored` slots of a bit-reversed spectrum of `full` slots.
#[derive(Clone, Copy, Debug)]
pub struct GaloisSlots {
    log_full: u32,
    p: i64,
    conj: bool,
}

impl GaloisSlots {
    /// `None` when `σ_p` fixes every slot. `stored` is `full` or `full / 2`.
    pub fn new(stored: usize, full: usize, p: i64) -> Option<Self> {
        assert!(p & 1 == 1, "convolution: b_galois must be odd");
        debug_assert!(full.is_power_of_two() && (stored == full || 2 * stored == full));
        let two_f = 2 * full as i64;
        // A half spectrum holds the exponents 1 mod 4: -p sends them back into it.
        let flip = stored < full && p.rem_euclid(4) != 1;
        let p = if flip { -p } else { p }.rem_euclid(two_f);
        // p = -1 on a complex half spectrum moves no slot but still conjugates.
        (p != 1 || flip).then_some(Self {
            log_full: full.trailing_zeros(),
            p,
            conj: flip,
        })
    }

    /// The stored slot whose value lands in slot `i`.
    #[inline(always)]
    pub fn source(&self, i: usize) -> usize {
        let shift = usize::BITS - self.log_full;
        let mask = (2i64 << self.log_full) - 1;
        let e = 2 * (i.reverse_bits() >> shift) as i64 + 1;
        let e_src = (self.p * e) & mask;
        (((e_src - 1) >> 1) as usize).reverse_bits() >> shift
    }

    /// Whether the read conjugates every complex slot (half spectrum, `p = 3 mod 4`); real slots ignore it.
    #[inline(always)]
    pub fn conj(&self) -> bool {
        self.conj
    }
}

/// How a degree-`N` convolution reads its right operand: the sparse shift, then the Galois map.
#[derive(Clone, Copy, Debug)]
pub struct RightRead {
    log_gap: usize,
    galois: Option<GaloisSlots>,
}

impl RightRead {
    /// `stored` and `full` describe the degree-`b_n` operand's transform.
    pub fn new(module_n: usize, b_n: usize, b_galois: i64, stored: usize, full: usize) -> Self {
        Self {
            log_gap: sparse_log_gap_portable(module_n, b_n),
            galois: GaloisSlots::new(stored, full, b_galois),
        }
    }

    /// The identity read of a dense or sparse operand.
    pub fn sparse(module_n: usize, b_n: usize) -> Self {
        Self {
            log_gap: sparse_log_gap_portable(module_n, b_n),
            galois: None,
        }
    }

    /// Dense and identity: the operand is read in place.
    #[inline(always)]
    pub fn is_direct(&self) -> bool {
        self.log_gap == 0 && self.galois.is_none()
    }

    #[inline(always)]
    pub fn log_gap(&self) -> usize {
        self.log_gap
    }

    #[inline(always)]
    pub fn galois(&self) -> Option<GaloisSlots> {
        self.galois
    }

    /// The stored slot that destination slot `slot` reads.
    #[inline(always)]
    pub fn source(&self, slot: usize) -> usize {
        let compact = slot >> self.log_gap;
        self.galois.map_or(compact, |g| g.source(compact))
    }

    #[inline(always)]
    pub fn conj(&self) -> bool {
        self.galois.is_some_and(|g| g.conj())
    }

    /// Lane `l` of every aligned block of `W` destination slots reads lane `table[key][l]` of stored block
    /// `source(W blk) / W`, where `key = source(W blk) % W`.
    pub fn lane_table<const W: usize>(&self) -> [[u8; W]; W] {
        debug_assert!(W.is_power_of_two() && W <= 256);
        // The lane bits left by the sparse shift, which `σ_p` maps to `key + p x` in bit-reversed order.
        let bits = W.trailing_zeros().saturating_sub(self.log_gap as u32);
        let mask = (1usize << bits) - 1;
        let p = self.galois.map_or(1, |g| g.p as usize);
        let rev = |x: usize| x.reverse_bits().checked_shr(usize::BITS - bits).unwrap_or(0);
        std::array::from_fn(|key| {
            std::array::from_fn(|l| ((key & !mask) | rev((rev(key & mask) + p * rev(l >> self.log_gap)) & mask)) as u8)
        })
    }
}

/// The read of a degree-`b_n` FFT64 right operand: complex half spectrum on the standard ring, real slots of the
/// degree-`2 b_n` spectrum on the conjugate-invariant ring.
pub fn right_read_fft64<R: Ring>(module_n: usize, b_n: usize, b_galois: i64) -> RightRead {
    let full = b_n * R::CYCLOTOMIC_ORDER_FACTOR as usize / 2;
    let stored = if full == b_n { b_n / 2 } else { b_n };
    RightRead::new(module_n, b_n, b_galois, stored, full)
}

/// The read of a degree-`b_n` NTT right operand: every slot of the degree-`b_n` spectrum on the standard ring,
/// the first half of the degree-`2 b_n` spectrum on the conjugate-invariant ring.
pub fn right_read_ntt<R: Ring>(module_n: usize, b_n: usize, b_galois: i64) -> RightRead {
    let full = b_n * R::CYCLOTOMIC_ORDER_FACTOR as usize / 2;
    RightRead::new(module_n, b_n, b_galois, b_n, full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::fft64::standard::build_fft64_automorphism_plan_portable;
    use crate::kernels::ntt4x30::standard::build_ntt4x30_automorphism_plan_portable;

    fn odd_ps(full: usize) -> Vec<i64> {
        let two_f = 2 * full as i64;
        (1..two_f).step_by(2).chain([-1, -5, two_f + 5, 3 * two_f - 1]).collect()
    }

    #[test]
    fn source_matches_ntt_standard_plan() {
        for log_n in 3..=10 {
            let n = 1usize << log_n;
            for p in odd_ps(n) {
                let plan = build_ntt4x30_automorphism_plan_portable(n, p);
                let g = GaloisSlots::new(n, n, p);
                for i in 0..n {
                    assert_eq!(g.map_or(i, |g| g.source(i)), plan.perm[i] as usize, "n={n} p={p} i={i}");
                }
            }
        }
    }

    #[test]
    fn source_matches_fft64_standard_plan() {
        for log_n in 3..=10 {
            let n = 1usize << log_n;
            for p in odd_ps(n) {
                let plan = build_fft64_automorphism_plan_portable(n, p);
                let g = GaloisSlots::new(n / 2, n, p);
                assert_eq!(g.is_some_and(|g| g.conj()), plan.conj, "n={n} p={p}");
                for i in 0..n / 2 {
                    assert_eq!(g.map_or(i, |g| g.source(i)), plan.perm[i] as usize, "n={n} p={p} i={i}");
                }
            }
        }
    }

    #[test]
    fn source_matches_conjugate_invariant_plans() {
        // Both conjugate-invariant rings keep the first n slots of the standard degree-2n layout.
        for log_n in 3..=10 {
            let n = 1usize << log_n;
            for p in odd_ps(2 * n) {
                let plan = build_ntt4x30_automorphism_plan_portable(2 * n, if p.rem_euclid(4) == 1 { p } else { -p });
                let g = GaloisSlots::new(n, 2 * n, p);
                for i in 0..n {
                    assert_eq!(g.map_or(i, |g| g.source(i)), plan.perm[i] as usize, "n={n} p={p} i={i}");
                }
            }
        }
    }

    #[test]
    fn every_block_reads_one_block() {
        for log_f in 3..=9 {
            let full = 1usize << log_f;
            for stored in [full, full / 2] {
                for p in odd_ps(full) {
                    let Some(g) = GaloisSlots::new(stored, full, p) else {
                        continue;
                    };
                    for w in [2usize, 4, 8].into_iter().filter(|&w| w <= stored) {
                        for blk in 0..stored / w {
                            let src = g.source(w * blk) / w;
                            for l in 1..w {
                                assert_eq!(
                                    g.source(w * blk + l) / w,
                                    src,
                                    "F={full} stored={stored} p={p} w={w} blk={blk}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lane_table_matches_source() {
        fn check<const W: usize>(read: RightRead, dest: usize, label: &str) {
            let table = read.lane_table::<W>();
            for blk in 0..dest / W {
                let first = read.source(W * blk);
                for (l, &lane) in table[first % W].iter().enumerate() {
                    let src = read.source(W * blk + l);
                    assert_eq!(src / W, first / W, "{label} W={W} blk={blk}");
                    assert_eq!(src % W, lane as usize, "{label} W={W} blk={blk} l={l}");
                }
            }
        }
        for log_f in 3..=8 {
            let full = 1usize << log_f;
            for stored in [full, full / 2] {
                for p in odd_ps(full) {
                    for log_gap in 0..=3 {
                        let read = RightRead::new(full << log_gap, full, p, stored, full);
                        let (dest, label) = (stored << log_gap, format!("F={full} stored={stored} p={p} log_gap={log_gap}"));
                        check::<2>(read, dest, &label);
                        check::<4>(read, dest, &label);
                        check::<8>(read, dest, &label);
                    }
                }
            }
        }
    }

    #[test]
    fn right_read_composes_sparse_then_galois() {
        let (module_n, b_n, p) = (64usize, 16usize, 5i64);
        let read = RightRead::new(module_n, b_n, p, b_n, b_n);
        let g = GaloisSlots::new(b_n, b_n, p).unwrap();
        for i in 0..module_n {
            assert_eq!(read.source(i), g.source(i >> 2));
        }
        assert!(RightRead::new(module_n, module_n, 1, module_n, module_n).is_direct());
        assert!(!RightRead::new(module_n, b_n, 1, b_n, b_n).is_direct());
        assert!(RightRead::new(module_n, module_n, 1 + 2 * module_n as i64, module_n, module_n).is_direct());
    }

    #[test]
    #[should_panic(expected = "convolution: b_galois must be odd")]
    fn even_galois_rejected() {
        let _ = RightRead::new(64, 64, 4, 64, 64);
    }
}
