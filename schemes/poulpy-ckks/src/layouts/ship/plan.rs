//! SHIP bootstrapping plan: instance dimensions and derived widths.
//!
//! A [`ShipPlan`] carries the SHIP half-bootstrap parameterization (Cheon,
//! Hanrot, Kim, Stehlé): the encoding gap, the working precision of the omega
//! ciphertexts, the residual output budget, and the sparse-secret / mux
//! blind-rotation dimensions. The bottom modulus is one limb (`q0 = 2^base2k`)
//! and the raised precision derives from the product-tree depth.

use anyhow::{Result, ensure};

/// Validated SHIP bootstrapping parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShipPlan {
    log_n: usize,
    log_gamma: usize,
    log_delta_work: usize,
    log_budget_out: usize,
    sparse_hamming_weight: usize,
    window: usize,
    mux_base: usize,
    theta: usize,
}

impl ShipPlan {
    /// Builds and validates a SHIP plan.
    ///
    /// `log_gamma` is the `log2` of the encoding gap, `log_delta_work` the
    /// working precision of the omega ciphertexts, `log_budget_out` the budget
    /// remaining on the output, `sparse_hamming_weight` the weight `h` of the
    /// sparse bottom secret, `window` the half-width `w` of the offset window
    /// `[-w, w]` around `k*n/h`, `mux_base` the base `B` of the mux digit
    /// decomposition, and `theta` the low-digit base absorbed into masking.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        log_n: usize,
        log_gamma: usize,
        log_delta_work: usize,
        log_budget_out: usize,
        sparse_hamming_weight: usize,
        window: usize,
        mux_base: usize,
        theta: usize,
    ) -> Result<Self> {
        ensure!((2..=32).contains(&log_n), "SHIP log_n must be in [2, 32], got {log_n}");
        let n = 1usize << log_n;
        ensure!(log_gamma >= 1, "SHIP log_gamma must be at least 1, got {log_gamma}");
        ensure!(
            log_delta_work >= 1,
            "SHIP log_delta_work must be at least 1, got {log_delta_work}"
        );
        let h = sparse_hamming_weight;
        ensure!(
            h > 0 && n.is_multiple_of(h),
            "SHIP sparse Hamming weight {h} must be positive and divide N = {n}"
        );
        ensure!(window >= 1, "SHIP window must be at least 1, got {window}");
        ensure!(2 * window < n, "SHIP window {window} exceeds the ring degree {n}");
        ensure!(mux_base >= 2, "SHIP mux base must be at least 2, got {mux_base}");
        ensure!(
            (1..=2 * window + 1).contains(&theta),
            "SHIP theta must be in [1, 2*window + 1 = {}], got {theta}",
            2 * window + 1
        );
        Ok(Self {
            log_n,
            log_gamma,
            log_delta_work,
            log_budget_out,
            sparse_hamming_weight,
            window,
            mux_base,
            theta,
        })
    }

    /// Ring degree exponent (`N = 2^log_n`).
    pub fn log_n(&self) -> usize {
        self.log_n
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    /// Slot count `m = N/2`.
    pub fn half_n(&self) -> usize {
        self.n() / 2
    }

    /// `log2` of the encoding gap `gamma`.
    pub fn log_gamma(&self) -> usize {
        self.log_gamma
    }

    /// Working precision of the omega ciphertexts, masks and plaintexts.
    pub fn log_delta_work(&self) -> usize {
        self.log_delta_work
    }

    /// Budget remaining on the output ciphertext.
    pub fn log_budget_out(&self) -> usize {
        self.log_budget_out
    }

    /// Hamming weight `h` of the sparse bottom secret.
    pub fn sparse_hamming_weight(&self) -> usize {
        self.sparse_hamming_weight
    }

    /// Half-width `w` of the secret offset window.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Base `B` of the mux blind-rotation digit decomposition.
    pub fn mux_base(&self) -> usize {
        self.mux_base
    }

    /// Low-digit base absorbed into the masking step.
    pub fn theta(&self) -> usize {
        self.theta
    }

    /// Spacing `N/h` between consecutive support positions.
    pub fn spacing(&self) -> usize {
        self.n() / self.sparse_hamming_weight
    }

    /// Bottom modulus `k0` in bits: a single limb.
    pub fn k0(&self, base2k: usize) -> usize {
        base2k
    }

    /// Depth of the product tree over `h + 1` leaves.
    pub fn tree_depth(&self) -> usize {
        (self.sparse_hamming_weight + 1).next_power_of_two().ilog2() as usize
    }

    /// Raised torus precision: one masking plaintext product plus `tree_depth`
    /// ciphertext products, the output value and its budget, limb-aligned.
    pub fn raised_k(&self, base2k: usize) -> usize {
        ((self.tree_depth() + 2) * self.log_delta_work + self.log_budget_out).next_multiple_of(base2k)
    }

    /// Rotation candidates left above the absorbed low digit.
    pub fn mux_candidates(&self) -> usize {
        (2 * self.window + 1).div_ceil(self.theta)
    }

    /// Mixed-radix digit bases of the mux blind rotation: positions of base at
    /// most `mux_base` whose product covers [`Self::mux_candidates`].
    pub fn mux_bases(&self) -> Vec<usize> {
        let values = self.mux_candidates();
        let mut bases = Vec::new();
        let mut cap = 1usize;
        while cap < values {
            let b = self.mux_base.min(values.div_ceil(cap));
            bases.push(b);
            cap *= b;
        }
        bases
    }

    /// Public rotation `p_k = (k*N/h - w) mod m` folded into slot `k`'s masks.
    pub fn mask_rotation(&self, slot: usize) -> usize {
        let m = self.half_n() as i64;
        ((slot * self.spacing()) as i64 - self.window as i64).rem_euclid(m) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_validates_dimensions() {
        assert!(ShipPlan::new(10, 6, 36, 24, 32, 175, 5, 1).is_ok());
        // h does not divide N.
        assert!(ShipPlan::new(10, 6, 36, 24, 33, 175, 5, 1).is_err());
        // theta exceeds the candidate count.
        assert!(ShipPlan::new(10, 6, 36, 24, 32, 2, 5, 6).is_err());
        // Degenerate mux base.
        assert!(ShipPlan::new(10, 6, 36, 24, 32, 175, 1, 1).is_err());
    }

    #[test]
    fn derived_widths() {
        let plan = ShipPlan::new(10, 6, 24, 12, 32, 175, 5, 1).unwrap();
        assert_eq!(plan.spacing(), 32);
        assert_eq!(plan.tree_depth(), 6); // 33 leaves -> 64.
        assert_eq!(plan.raised_k(52), (8usize * 24 + 12).next_multiple_of(52));
        let bases = plan.mux_bases();
        assert!(bases.iter().all(|&b| b <= 5));
        assert!(bases.iter().product::<usize>() >= plan.mux_candidates());
    }
}
