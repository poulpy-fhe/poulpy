//! Cleartext SHIP selector masks (Lemma 1 / Eq. (5) of the SHIP paper).

use crate::api::ShipScalar;
use crate::layouts::ShipPlan;

/// The four pre-rotated slot masks `m'^(k) = Rot_{-j}(m^(k))`, in `pi_k`
/// order: `pi1 = w1(a)`, `pi2 = w1(-a)`, `pi3 = w2(a)`, `pi4 = w2(-a)`.
/// `Rot_r` follows the paper: `out[i] = in[(i - r) mod m]`.
pub(crate) fn ship_pre_rotated_masks(j: usize, s_j: i64, m: usize) -> [Vec<f64>; 4] {
    debug_assert!(j < 2 * m && (s_j == 1 || s_j == -1));
    let (j_eff, swap) = if j < m { (j, false) } else { (j - m, true) };
    let pos = ((1 + s_j) / 2) as f64;
    let neg = ((1 - s_j) / 2) as f64;
    let mut masks = [vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]];
    for i in 0..m {
        // Pre-rotation: m'[i] = m[(i + j) mod m].
        let mj = if (i + j_eff) % m >= j_eff { 1.0 } else { 0.0 };
        let (k1, k2, k3, k4) = if swap {
            (neg * (1.0 - mj), pos * (1.0 - mj), neg * mj, pos * mj)
        } else {
            (pos * mj, neg * mj, neg * (1.0 - mj), pos * (1.0 - mj))
        };
        for (mask, v) in masks.iter_mut().zip([k1, k2, k3, k4]) {
            mask[i] = v;
        }
    }
    masks
}

/// Second-half masks (Lemma 4): `omega_2(a~_j) = omega_1(a~'_j)` with `a~'_j`
/// of rotation index `j + N/2` and sign `-s_j` for `j < N/2` (`X^{N/2}`-shift
/// with wraparound sign), index `j - N/2` and sign `s_j` otherwise.
pub(crate) fn ship_pre_rotated_masks_omega2(j: usize, s_j: i64, m: usize) -> [Vec<f64>; 4] {
    if j < m {
        ship_pre_rotated_masks(j + m, -s_j, m)
    } else {
        ship_pre_rotated_masks(j - m, s_j, m)
    }
}

/// The `4*theta` encrypted-selector slot vectors of one support slot,
/// candidate-major (`1_{c = u mod theta} x Rot_{p + c}(mask)`), the secret
/// part and the public rotation `p` both folded in; the column/mux hybrid of
/// SHIP §4.4 in the low-digit-column variant used by the paper's own
/// implementation (end of §4.4). All imaginary parts are zero.
pub(crate) fn ship_mask_slot_vectors<F: ShipScalar>(
    plan: &ShipPlan,
    slot: usize,
    j: usize,
    s_j: i64,
    u: usize,
    omega2: bool,
) -> Vec<Vec<F>> {
    let m = plan.half_n();
    let theta = plan.theta();
    let p = plan.mask_rotation(slot);
    let u0 = u % theta;
    let bands = if omega2 {
        ship_pre_rotated_masks_omega2(j, s_j, m)
    } else {
        ship_pre_rotated_masks(j, s_j, m)
    };
    let mut vectors = Vec::with_capacity(4 * theta);
    for c in 0..theta {
        for band in &bands {
            let re: Vec<F> = (0..m)
                .map(|i| {
                    if c == u0 {
                        F::from_f64(band[(i + 2 * m - (p + c)) % m]).expect("band values are exact")
                    } else {
                        F::zero()
                    }
                })
                .collect();
            vectors.push(re);
        }
    }
    vectors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_partition_unity() {
        let m = 16;
        for (j, s_j) in [(3usize, 1i64), (3, -1), (19, 1), (19, -1), (0, 1)] {
            let masks = ship_pre_rotated_masks(j, s_j, m);
            for i in 0..m {
                let total: f64 = masks.iter().map(|w| w[i]).sum();
                assert_eq!(total, 1.0, "j={j} s_j={s_j} i={i}");
            }
        }
    }

    #[test]
    fn omega2_wraps_half_turn() {
        let m = 16;
        assert_eq!(ship_pre_rotated_masks_omega2(3, 1, m), ship_pre_rotated_masks(19, -1, m));
        assert_eq!(ship_pre_rotated_masks_omega2(19, 1, m), ship_pre_rotated_masks(3, 1, m));
    }
}
