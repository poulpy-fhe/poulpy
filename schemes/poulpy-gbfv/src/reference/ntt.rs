//! Reference negacyclic NTT over `Z/q`, on host-visible backend storage.

use anyhow::{Result, ensure};
use dashu_int::{UBig, fast_div::ConstDivisor, modular::Reduced};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module},
};

use crate::layouts::{GBFVNttBufferToBackendMut, GBFVNttPlanToBackendRef, read_residue, write_residue};

/// Reference bodies of [`GBFVNttImpl`](crate::oep::GBFVNttImpl).
///
/// The forward transform is the Cooley-Tukey negacyclic NTT: for `m = 1, 2, ..., n/2`,
/// with `t = n / 2m`, every block `i < m` applies the butterfly
/// `(u, v) -> (u + w v, u - w v)` to the pairs `(j, j + t)`, `2 i t <= j < (2 i + 1) t`,
/// with `w = psi^(brv(m + i))`. The inverse transform is the Gentleman-Sande negacyclic
/// NTT: for `m = n, n/2, ..., 2`, with `h = m/2` and `t = n / m`, every block `i < h`
/// applies `(u, v) -> (u + v, (u - v) w)` with `w = psi^(-brv(h + i))`, followed by a
/// multiplication by `n^(-1)`. All arithmetic is exact modulo `q`.
///
/// The bodies read and write the backend storage as host bytes.
pub trait GBFVNttReference<BE: Backend> {
    fn gbfv_ntt_forward_assign_reference<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        Self: ModuleN,
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        for<'a> BE::BufMut<'a>: AsMut<[u8]>,
    {
        ntt_assign::<BE, P, V>(self.n(), plan, values, false)
    }

    fn gbfv_ntt_inverse_assign_reference<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        Self: ModuleN,
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        for<'a> BE::BufMut<'a>: AsMut<[u8]>,
    {
        ntt_assign::<BE, P, V>(self.n(), plan, values, true)
    }
}

impl<BE: Backend> GBFVNttReference<BE> for Module<BE> {}

fn ntt_assign<BE, P, V>(module_n: usize, plan: &P, values: &mut V, inverse: bool) -> Result<()>
where
    BE: Backend,
    P: GBFVNttPlanToBackendRef<BE>,
    V: GBFVNttBufferToBackendMut<BE>,
    for<'a> BE::BufRef<'a>: AsRef<[u8]>,
    for<'a> BE::BufMut<'a>: AsMut<[u8]>,
{
    ensure!(plan.log_n() == values.log_n(), "gbfv ntt: plan and buffer degrees differ");
    ensure!(
        plan.limbs() == values.limbs(),
        "gbfv ntt: plan and buffer residue widths differ"
    );
    ensure!(values.n() <= module_n, "gbfv ntt: degree exceeds the module degree");

    let (n, limbs): (usize, usize) = (values.n(), values.limbs());
    let plan = plan.to_backend_ref();
    let table: &[u8] = plan.data().as_ref();
    let modulus: UBig = read_residue(table, 0, limbs);
    let ring: ConstDivisor = ConstDivisor::new(modulus.clone());
    let twiddle = |entry: usize| ring.reduce(read_residue(table, entry, limbs));

    let mut view = values.to_backend_mut();
    let bytes: &mut [u8] = view.data.as_mut();
    let mut a: Vec<Reduced<'_>> = (0..n)
        .map(|index| {
            let value: UBig = read_residue(bytes, index, limbs);
            debug_assert!(value < modulus, "gbfv ntt: residue out of range");
            ring.reduce(value)
        })
        .collect();

    if inverse {
        let mut t: usize = 1;
        let mut m: usize = n;
        while m > 1 {
            let h: usize = m / 2;
            for i in 0..h {
                let w = twiddle(2 + n + h + i);
                let start: usize = 2 * i * t;
                for j in start..start + t {
                    let (u, v) = (a[j].clone(), a[j + t].clone());
                    a[j] = &u + &v;
                    a[j + t] = (u - v) * &w;
                }
            }
            t *= 2;
            m = h;
        }
        let n_inv = twiddle(1);
        for x in a.iter_mut() {
            *x *= &n_inv;
        }
    } else {
        let mut t: usize = n;
        let mut m: usize = 1;
        while m < n {
            t /= 2;
            for i in 0..m {
                let w = twiddle(2 + m + i);
                let start: usize = 2 * i * t;
                for j in start..start + t {
                    let (u, v) = (a[j].clone(), &a[j + t] * &w);
                    a[j] = &u + &v;
                    a[j + t] = u - v;
                }
            }
            m *= 2;
        }
    }

    for (index, x) in a.iter().enumerate() {
        write_residue(bytes, index, limbs, &x.residue());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use dashu_int::{IBig, ops::BitTest};
    use poulpy_hal::layouts::HostBytesBackend;

    use super::*;
    use crate::{
        api::GBFVNttOps,
        layouts::{GBFVBatchEncoder, GBFVNttBuffer, GBFVNttInfos, GBFVNttPlan, GBFVPlaintextModulus, bit_reverse},
    };

    crate::impl_gbfv_ntt_reference_full!(HostBytesBackend);

    type BE = HostBytesBackend;
    type Buffer = GBFVNttBuffer<<BE as Backend>::OwnedBuf>;
    type Plan = GBFVNttPlan<<BE as Backend>::OwnedBuf>;

    /// `q = p^e` and the batch encoder's primitive `2n`-th root modulo `q`.
    fn modulus_and_root(t: &[i64], e: usize, log_n: usize) -> (UBig, UBig) {
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(t).unwrap();
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, e, log_n).unwrap();
        (encoder.characteristic().pow(e), encoder.root_of_unity().clone())
    }

    /// Deterministic residues in `[0, q)`.
    fn pseudo_random(n: usize, seed: u64, q: &UBig) -> Vec<UBig> {
        let mut state: u64 = seed;
        (0..n)
            .map(|_| {
                let mut value: UBig = UBig::ZERO;
                for _ in 0..q.bit_len().div_ceil(32) + 1 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    value = (value << 32) + UBig::from(state >> 32);
                }
                value % q
            })
            .collect()
    }

    /// The contract: `a_hat[i] = sum_j a[j] psi^((2 brv(i) + 1) j) mod q`.
    fn naive_forward(a: &[UBig], q: &UBig, psi: &UBig) -> Vec<UBig> {
        let ring: ConstDivisor = ConstDivisor::new(q.clone());
        let log_n: usize = a.len().trailing_zeros() as usize;
        (0..a.len())
            .map(|i| {
                let x = ring.reduce(psi.clone()).pow(&UBig::from(2 * bit_reverse(i, log_n) + 1));
                a.iter()
                    .rev()
                    .fold(ring.reduce(0u8), |acc, c| acc * &x + ring.reduce(c.clone()))
                    .residue()
            })
            .collect()
    }

    /// `a * b` in `(Z/q)[X]/(X^n + 1)`.
    fn naive_negacyclic_mul(a: &[UBig], b: &[UBig], q: &UBig) -> Vec<UBig> {
        let n: usize = a.len();
        let mut res: Vec<IBig> = vec![IBig::ZERO; n];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                let prod: IBig = IBig::from(x * y);
                if i + j < n {
                    res[i + j] += prod;
                } else {
                    res[i + j - n] -= prod;
                }
            }
        }
        let q: IBig = IBig::from(q.clone());
        res.into_iter()
            .map(|c| UBig::try_from(((c % &q) + &q) % &q).unwrap())
            .collect()
    }

    fn forward(module: &Module<BE>, plan: &Plan, a: &[UBig]) -> Vec<UBig> {
        let mut buffer: Buffer = Buffer::from_host::<BE>(a, plan.limbs()).unwrap();
        module.gbfv_ntt_forward_assign(plan, &mut buffer).unwrap();
        buffer.to_host::<BE>()
    }

    fn inverse(module: &Module<BE>, plan: &Plan, a: &[UBig]) -> Vec<UBig> {
        let mut buffer: Buffer = Buffer::from_host::<BE>(a, plan.limbs()).unwrap();
        module.gbfv_ntt_inverse_assign(plan, &mut buffer).unwrap();
        buffer.to_host::<BE>()
    }

    /// Checks the forward transform against its definition, the inverse against the
    /// forward, and the convolution property.
    fn check(log_n: usize, q: &UBig, psi: &UBig) {
        let module: Module<BE> = Module::<BE>::new(1 << log_n.max(3));
        let plan: Plan = Plan::new::<BE>(log_n, q, psi).unwrap();
        let n: usize = 1 << log_n;
        let a: Vec<UBig> = pseudo_random(n, 1, q);
        let b: Vec<UBig> = pseudo_random(n, 2, q);

        let a_hat: Vec<UBig> = forward(&module, &plan, &a);
        assert_eq!(a_hat, naive_forward(&a, q, psi));
        assert_eq!(inverse(&module, &plan, &a_hat), a);

        let b_hat: Vec<UBig> = forward(&module, &plan, &b);
        let product_hat: Vec<UBig> = a_hat.iter().zip(&b_hat).map(|(x, y)| (x * y) % q).collect();
        assert_eq!(inverse(&module, &plan, &product_hat), naive_negacyclic_mul(&a, &b, q));
    }

    #[test]
    fn ntt_modulo_prime_power() {
        // `p = 2^16 + 1`, `q = p^3`, one limb.
        let mut t: Vec<i64> = vec![0; 16];
        t[0] = -2;
        t[1] = 1;
        let (q, psi) = modulus_and_root(&t, 3, 6);
        assert_eq!(q.bit_len().div_ceil(64), 1);
        check(6, &q, &psi);
    }

    #[test]
    fn ntt_beyond_128_bits() {
        // `p = 29 * 2^57 + 1`, `q = p^3` of 186 bits, three limbs.
        let (q, psi) = modulus_and_root(&[4_179_340_454_199_820_289], 3, 5);
        assert!(q.bit_len() > 128);
        assert_eq!(q.bit_len().div_ceil(64), 3);
        check(5, &q, &psi);
    }

    #[test]
    fn ntt_small_degrees() {
        // Degrees 1 and 2: a single butterfly level or none.
        let (q, psi) = modulus_and_root(&[65537], 2, 1);
        check(1, &q, &psi);
        let minus_one: UBig = &q - UBig::ONE;
        check(0, &q, &minus_one);
    }

    #[test]
    fn plan_rejects_invalid_parameters() {
        let (q, psi) = modulus_and_root(&[65537], 1, 4);
        assert!(Plan::new::<BE>(4, &q, &psi).is_ok());
        assert!(Plan::new::<BE>(4, &(&q + UBig::ONE), &psi).is_err());
        assert!(Plan::new::<BE>(4, &UBig::ONE, &psi).is_err());
        // `psi^2` has order 16, not 32.
        assert!(Plan::new::<BE>(4, &q, &((&psi * &psi) % &q)).is_err());
        assert!(Plan::new::<BE>(5, &q, &psi).is_err());
    }

    #[test]
    fn ntt_rejects_mismatched_operands() {
        let (q, psi) = modulus_and_root(&[65537], 1, 4);
        let module: Module<BE> = Module::<BE>::new(16);
        let plan: Plan = Plan::new::<BE>(4, &q, &psi).unwrap();
        let mut short: Buffer = Buffer::alloc::<BE>(3, 1);
        assert!(module.gbfv_ntt_forward_assign(&plan, &mut short).is_err());
        let mut wide: Buffer = Buffer::alloc::<BE>(4, 2);
        assert!(module.gbfv_ntt_inverse_assign(&plan, &mut wide).is_err());
        let small: Module<BE> = Module::<BE>::new(8);
        let mut buffer: Buffer = Buffer::alloc::<BE>(4, 1);
        assert!(small.gbfv_ntt_forward_assign(&plan, &mut buffer).is_err());
    }

    #[test]
    fn buffer_rejects_values_wider_than_limbs() {
        assert!(Buffer::from_host::<BE>(&[UBig::ONE << 64], 1).is_err());
        assert!(Buffer::from_host::<BE>(&[UBig::ONE, UBig::ONE, UBig::ONE], 1).is_err());
        let values: Vec<UBig> = vec![UBig::ONE << 100, UBig::from(7u8)];
        assert_eq!(Buffer::from_host::<BE>(&values, 2).unwrap().to_host::<BE>(), values);
    }
}
