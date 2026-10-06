use crate::{DFTFamily, Fft64, Ntt4x30};

fn check_embedding<F: DFTFamily>(min_log_n: usize, matches: impl Fn(F::Dft, F::Dft) -> bool) {
    for log_n in min_log_n..=6 {
        let n = 1 << log_n;
        let small_table = F::table(n);
        for log_big in log_n..=8 {
            let big = 1 << log_big;
            let big_table = F::table(big);
            // Check every coefficient position, including both halves of the FFT layout.
            for position in 0..=n {
                let mut input = vec![0; n];
                if position < n {
                    input[position] = -7;
                } else {
                    for (i, x) in input.iter_mut().enumerate() {
                        *x = (i * 13 % 31) as i64 - 15;
                    }
                }
                let mut coefficients = vec![0; big];
                for (i, &x) in input.iter().enumerate() {
                    coefficients[i * (big / n)] = x;
                }
                let mut small = vec![F::Dft::default(); n];
                F::forward(&small_table, &mut small, &input);
                let mut actual = vec![F::Dft::default(); big];
                F::dft_embed(&mut actual, &small);
                let mut expected = vec![F::Dft::default(); big];
                F::forward(&big_table, &mut expected, &coefficients);
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert!(matches(actual, expected), "embedding {n} -> {big} at coefficient {position}");
                }
            }
        }
    }
}

#[test]
fn fft_embedding_matches_coefficient_embedding() {
    check_embedding::<Fft64>(1, |a, b| (a - b).abs() < 1e-9);
}

#[test]
fn ntt_embedding_matches_coefficient_embedding() {
    check_embedding::<Ntt4x30>(0, |a, b| a.0 == b.0);
}

#[test]
fn ntt_embedding_preserves_coefficients_wider_than_i64() {
    let mut transformed = vec![Default::default(); 8];
    let mut input = vec![0; 8];
    input[3] = 1 << 40;
    Ntt4x30::forward(&Ntt4x30::table(8), &mut transformed, &input);
    let factor = transformed.clone();
    Ntt4x30::mul_assign(&mut transformed, &factor);
    let mut embedded = vec![Default::default(); 64];
    Ntt4x30::dft_embed(&mut embedded, &transformed);
    let mut actual = vec![0; 64];
    Ntt4x30::inverse(&Ntt4x30::table(64), &mut actual, &embedded);
    let mut expected = vec![0; 64];
    expected[48] = 1i128 << 80;
    assert!(actual == expected);
}
