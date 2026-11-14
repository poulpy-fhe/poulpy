//! Plain Keccak-f[1600], the reference of the encrypted circuit.
//!
//! The state is 25 lanes of 64 bits, lane `(x, y)` at index `x + 5y`, bit `z` of a lane at weight `2^z`.

/// Round constants of iota.
pub const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets of rho, `RHO[x][y]`.
pub const RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// One round of Keccak-f[1600] with round constant `rc`.
fn round(a: &mut [u64; 25], rc: u64) {
    // theta
    let c: [u64; 5] = std::array::from_fn(|x| (0..5).fold(0, |acc, y| acc ^ a[x + 5 * y]));
    for x in 0..5 {
        let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        for y in 0..5 {
            a[x + 5 * y] ^= d;
        }
    }
    // rho and pi: B[y, 2x + 3y] = rot(A[x, y], RHO[x][y])
    let mut b = [0u64; 25];
    for x in 0..5 {
        for y in 0..5 {
            b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(RHO[x][y]);
        }
    }
    // chi
    for x in 0..5 {
        for y in 0..5 {
            a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
        }
    }
    // iota
    a[0] ^= rc;
}

/// Keccak-f[1600].
pub fn keccak_f(a: &mut [u64; 25]) {
    for rc in ROUND_CONSTANTS {
        round(a, rc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_f_of_zero() {
        let mut a = [0u64; 25];
        keccak_f(&mut a);
        assert_eq!(a[0], 0xF1258F7940E1DDE7);
    }
}
