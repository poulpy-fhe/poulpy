//! Eight Keccak states in the 2^14 slots of one ciphertext, and the maps of a round on them.
//!
//! Bit `z` of lane `(x, y)` of a state sits at position `x + 5y + 32z`, and position `p` of state `s` at slot `8p + s`.

use crate::keccak::RHO;

/// Keccak states per ciphertext.
pub const STATES: usize = 8;
/// Positions of one state.
pub const POSITIONS: usize = 2048;
/// Slots of the ciphertext.
pub const SLOTS: usize = STATES * POSITIONS;

/// Position of bit `z` of lane `(x, y)` in its state.
const fn position(x: usize, y: usize, z: usize) -> usize {
    x + 5 * y + 32 * z
}

/// Slot of position `p` of state `s`.
const fn slot(s: usize, p: usize) -> usize {
    STATES * p + s
}

/// The lane bits of a state, `(x, y, z)`.
fn bits() -> impl Iterator<Item = (usize, usize, usize)> {
    (0..64).flat_map(|z| (0..5).flat_map(move |y| (0..5).map(move |x| (x, y, z))))
}

/// The states as slot values, 0 or 1, the unused slots 0.
pub fn to_slots(states: &[[u64; 25]; STATES]) -> Vec<f64> {
    let mut v = vec![0.0; SLOTS];
    for (s, a) in states.iter().enumerate() {
        for (x, y, z) in bits() {
            v[slot(s, position(x, y, z))] = ((a[x + 5 * y] >> z) & 1) as f64;
        }
    }
    v
}

/// The states of slot values, each rounded and reduced mod 2.
pub fn from_slots(v: &[f64]) -> [[u64; 25]; STATES] {
    std::array::from_fn(|s| {
        let mut a = [0u64; 25];
        for (x, y, z) in bits() {
            a[x + 5 * y] |= ((v[slot(s, position(x, y, z))].round() as i64).rem_euclid(2) as u64) << z;
        }
        a
    })
}

/// A linear map on the slots, as `(out, in, weight)` terms.
pub type SlotMap = Vec<(usize, usize, i64)>;

/// The map that applies `terms`, a map on the positions of one state, to every state.
fn on_every_state(terms: impl Iterator<Item = (usize, usize, i64)>) -> SlotMap {
    terms
        .flat_map(|(o, i, w)| (0..STATES).map(move |s| (slot(s, o), slot(s, i), w)))
        .collect()
}

/// theta, rho and pi of one round, over the integers: XOR is addition, reduced mod 2 by the next bootstrapping.
pub fn linear_layer() -> SlotMap {
    on_every_state(bits().flat_map(|(x, y, z)| {
        // pi and rho: output (x, y, z) reads theta's output at (x + 3y, x, z - RHO[x + 3y][x]).
        let (sx, sy) = ((x + 3 * y) % 5, x);
        let sz = (z + 64 - RHO[sx][sy] as usize) % 64;
        let out = position(x, y, z);
        // theta: A[x, y, z] + sum_y' A[x - 1, y', z] + sum_y' A[x + 1, y', z - 1]
        std::iter::once((out, position(sx, sy, sz), 1)).chain((0..5).flat_map(move |yy| {
            [
                (out, position((sx + 4) % 5, yy, sz), 1),
                (out, position((sx + 1) % 5, yy, (sz + 63) % 64), 1),
            ]
        }))
    }))
}

/// The alignment of chi: output `(x, y, z)` reads `(x + k, y, z)`.
pub fn shift_x(k: usize) -> SlotMap {
    on_every_state(bits().map(move |(x, y, z)| (position(x, y, z), position((x + k) % 5, y, z), 1)))
}

/// The slots of lane `(0, 0)` that the round constant `rc` flips, in every state.
pub fn iota_slots(rc: u64) -> impl Iterator<Item = usize> {
    (0..64)
        .filter(move |z| (rc >> z) & 1 == 1)
        .flat_map(|z| (0..STATES).map(move |s| slot(s, position(0, 0, z))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::{ROUND_CONSTANTS, keccak_f};

    fn apply(map: &SlotMap, v: &[i64]) -> Vec<i64> {
        let mut y = vec![0; v.len()];
        for &(o, i, w) in map {
            y[o] += w * v[i];
        }
        y
    }

    /// The server's circuit on plain integers, the bootstrapping a reduction mod 2, is Keccak-f.
    #[test]
    fn round_maps_are_keccak() {
        let (linear, shift1, shift2) = (linear_layer(), shift_x(1), shift_x(2));
        let states: [[u64; 25]; STATES] =
            std::array::from_fn(|s| std::array::from_fn(|i| ((s * 25 + i) as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)));
        let mut c: Vec<i64> = to_slots(&states).iter().map(|&x| x as i64).collect();
        for rc in ROUND_CONSTANTS {
            let b: Vec<i64> = apply(&linear, &c).iter().map(|x| x.rem_euclid(2)).collect();
            let (a1, a2) = (apply(&shift1, &b), apply(&shift2, &b));
            c = (0..SLOTS).map(|i| b[i] - a2[i] + a1[i] * a2[i]).collect();
            for s in iota_slots(rc) {
                c[s] -= 1;
            }
        }
        let got = from_slots(&c.iter().map(|&x| x as f64).collect::<Vec<_>>());
        for (got, mut want) in got.into_iter().zip(states) {
            keccak_f(&mut want);
            assert_eq!(got, want);
        }
    }
}
