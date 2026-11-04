use anyhow::Result;
use poulpy_hal::layouts::Backend;

use crate::layouts::{GBFVNttBufferToBackendMut, GBFVNttPlanToBackendRef};

/// Negacyclic number-theoretic transform over `Z/q`, for a modulus `q` of any size.
///
/// A [`GBFVNttPlan`](crate::layouts::GBFVNttPlan) of degree `n = 2^log_n` fixes the odd
/// modulus `q` and a root `psi` with `psi^n = -1 mod q`. With `brv` the reversal of
/// the `log_n` low bits, the forward transform of `a` is
///
/// ```text
/// a_hat[i] = sum_{0 <= j < n} a[j] * psi^((2 brv(i) + 1) j) mod q,
/// ```
///
/// the evaluations of `a(X)` at the odd powers of `psi`, the roots of `X^n + 1`, in
/// bit-reversed order. The inverse transform is its exact inverse. Both take and
/// return residues in `[0, q)`, in place; inputs outside `[0, q)` are a caller
/// obligation.
///
/// Errors before writing when the buffer and the plan differ in degree or residue
/// width, or when the degree exceeds the module degree.
pub trait GBFVNttOps<BE: Backend> {
    /// Replaces `values` by its forward transform under `plan`.
    fn gbfv_ntt_forward_assign<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>;

    /// Replaces `values` by its inverse transform under `plan`.
    fn gbfv_ntt_inverse_assign<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>;
}
