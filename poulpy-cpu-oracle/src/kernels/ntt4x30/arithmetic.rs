use super::primes::PrimeSetCrt4;

pub fn b_from_znx64_ref<P: PrimeSetCrt4>(nn: usize, res: &mut [u64], x: &[i64]) {
    assert!(res.len() >= 4 * nn);
    assert!(x.len() >= nn);
    for j in 0..nn {
        for k in 0..4 {
            res[4 * j + k] = x[j].rem_euclid(P::Q[k] as i64) as u64;
        }
    }
}

pub fn b_to_znx128_ref<P: PrimeSetCrt4>(nn: usize, res: &mut [i128], x: &[u64]) {
    assert!(res.len() >= nn);
    assert!(x.len() >= 4 * nn);

    let q: [i128; 4] = P::Q.map(|qi| qi as i128);
    let total_q: i128 = q[0] * q[1] * q[2] * q[3];
    // qm[k] = Q / Q[k]
    let qm: [i128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let crt: [i128; 4] = std::array::from_fn(|k| {
        let (mut a, mut exponent, mut result) = (qm[k] % q[k], q[k] - 2, 1i128);
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * a % q[k];
            }
            a = a * a % q[k];
            exponent >>= 1;
        }
        result
    });

    for j in 0..nn {
        let mut tmp: i128 = 0;
        for k in 0..4 {
            let xk = (x[4 * j + k] % P::Q[k] as u64) as i128;
            let t = (xk * crt[k]) % q[k];
            tmp += t * qm[k];
        }
        tmp %= total_q;
        let half = (total_q + 1) / 2;
        res[j] = if tmp >= half { tmp - total_q } else { tmp };
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::super::primes::{PrimeSet, Primes30};
    use super::*;

    #[test]
    fn crt_reconstructs_signed_boundary_values() {
        let modulus: i128 = Primes30::Q.iter().map(|q| *q as i128).product();
        let half = modulus / 2;
        let expected = [-half, -half + 1, i64::MIN as i128, -1, 0, 1, i64::MAX as i128, half - 1, half];
        let residues: Vec<u64> = expected
            .iter()
            .flat_map(|x| Primes30::Q.map(|q| x.rem_euclid(q as i128) as u64))
            .collect();
        let mut actual = [0; 9];
        b_to_znx128_ref::<Primes30>(expected.len(), &mut actual, &residues);
        assert_eq!(actual, expected);
    }
}
