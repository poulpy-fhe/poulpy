//! Regularly-spaced sparse SHIP secret: support sampling and validation.

use anyhow::{Result, ensure};
use poulpy_core::layouts::{GLWESecret, LWEInfos};
use poulpy_hal::{layouts::ZnxViewMut, source::Source};

use super::plan::ShipPlan;

/// The support of a regularly-spaced sparse SHIP secret: the `k`-th nonzero
/// coefficient sits at `(k*N/h + delta_k) mod N` with `delta_k` in `[-w, w]`
/// and a uniform sign, all indices distinct, in `k` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShipSecretSpec {
    support: Vec<(usize, i64)>,
}

impl ShipSecretSpec {
    /// Samples a fresh regularly-spaced support for `plan`.
    pub fn sample(plan: &ShipPlan, source: &mut Source) -> Self {
        let n = plan.n();
        let h = plan.sparse_hamming_weight();
        let w = plan.window();
        let spacing = plan.spacing();
        let values = (2 * w + 1) as u64;
        let mask = values.next_power_of_two() - 1;
        let mut used = vec![false; n];
        let mut support = Vec::with_capacity(h);
        for k in 0..h {
            loop {
                let delta = source.next_u64n(values, mask) as i64 - w as i64;
                let idx = ((k * spacing) as i64 + delta).rem_euclid(n as i64) as usize;
                if !used[idx] {
                    used[idx] = true;
                    let sign = if source.next_u64n(2, 1) == 0 { 1 } else { -1 };
                    support.push((idx, sign));
                    break;
                }
            }
        }
        Self { support }
    }

    /// Validates an externally supplied support against `plan`.
    pub fn from_support(plan: &ShipPlan, support: Vec<(usize, i64)>) -> Result<Self> {
        let n = plan.n();
        let h = plan.sparse_hamming_weight();
        ensure!(
            support.len() == h,
            "SHIP support has {} entries, expected the sparse Hamming weight {h}",
            support.len()
        );
        let mut used = vec![false; n];
        for (k, &(idx, sign)) in support.iter().enumerate() {
            ensure!(idx < n, "SHIP support index {idx} exceeds the ring degree {n}");
            ensure!(
                sign == 1 || sign == -1,
                "SHIP support sign at slot {k} must be +-1, got {sign}"
            );
            ensure!(!used[idx], "SHIP support index {idx} is duplicated");
            used[idx] = true;
            ensure!(
                offset_of(plan, k, idx) <= 2 * plan.window(),
                "SHIP support index {idx} lies outside the window of slot {k}"
            );
        }
        Ok(Self { support })
    }

    /// The `(index, sign)` pairs in `k` order.
    pub fn support(&self) -> &[(usize, i64)] {
        &self.support
    }

    /// Windowed offset `u_k = (j_k - k*N/h + w) mod N` of slot `k`.
    pub fn offset(&self, plan: &ShipPlan, slot: usize) -> usize {
        offset_of(plan, slot, self.support[slot].0)
    }

    /// Writes the support into the coefficients of a host GLWE secret.
    pub fn fill_glwe_secret(&self, plan: &ShipPlan, sk: &mut GLWESecret<Vec<u8>, i64>) -> Result<()> {
        ensure!(
            sk.n().as_usize() == plan.n(),
            "SHIP secret degree {} does not match plan degree {}",
            sk.n(),
            plan.n()
        );
        sk.fill_zero();
        let col = sk.data_mut().at_mut(0, 0);
        for &(idx, sign) in &self.support {
            col[idx] = sign;
        }
        Ok(())
    }
}

fn offset_of(plan: &ShipPlan, slot: usize, idx: usize) -> usize {
    let n = plan.n() as i64;
    (idx as i64 - (slot * plan.spacing()) as i64 + plan.window() as i64).rem_euclid(n) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_support_validates() {
        let plan = ShipPlan::new(8, 6, 24, 12, 16, 7, 5, 1).unwrap();
        let mut source = Source::new([1u8; 32]);
        let spec = ShipSecretSpec::sample(&plan, &mut source);
        let revalidated = ShipSecretSpec::from_support(&plan, spec.support().to_vec()).unwrap();
        assert_eq!(spec, revalidated);
        for k in 0..plan.sparse_hamming_weight() {
            assert!(spec.offset(&plan, k) <= 2 * plan.window());
        }
    }

    #[test]
    fn rejects_out_of_window_support() {
        let plan = ShipPlan::new(8, 6, 24, 12, 16, 2, 5, 1).unwrap();
        let mut support: Vec<(usize, i64)> = (0..16).map(|k| (k * 16, 1)).collect();
        support[3] = (3 * 16 + 5, 1);
        assert!(ShipSecretSpec::from_support(&plan, support).is_err());
    }
}
