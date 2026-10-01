use poulpy_core::layouts::{GLWEInfos, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEShamirLayout, GLWEShamirPolynomialOwned, GLWEShamirShareOwned, GLWEWideSecretOwned};

/// Shamir thresholdization over the Galois ring `GR(2^k, gr_degree)`: every
/// party shares its secret with a Shamir polynomial, sends its evaluation at
/// every party's point, and aggregates the shares it receives into a
/// t-out-of-N share of the sum of the parties' secrets; every party of an
/// active set of at least the threshold finalizes its share into an additive
/// share of the secret, an integer polynomial modulo `2^k`, and the active
/// parties' additive shares sum to the secret.
///
/// Party `i`, `1 <= i < 2^gr_degree`, has the point whose coefficients are the
/// bits of `i`.
///
/// Shares are secret: a party sends each one over a private channel, since
/// any `threshold` of them reveal the secret.
pub trait GLWEShamirMHEProtocol<BE: Backend> {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize;

    /// Writes into `res` a Shamir polynomial whose constant term is `sk` and
    /// whose other coefficients are uniform over the Galois ring, drawn from
    /// `source_xm`, which must stay secret to the party and never be replayed.
    fn mhe_glwe_shamir_polynomial_gen<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos;

    fn mhe_glwe_shamir_share_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize;

    /// Writes into `res` the evaluation of `poly` at the point of party
    /// `recipient`.
    fn mhe_glwe_shamir_share_gen(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_shamir_share_aggregate(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>);

    fn mhe_glwe_shamir_share_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res` the additive share of party `own` among `actives`.
    fn mhe_glwe_shamir_share_finalize(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    );
}
