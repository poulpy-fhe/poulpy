//! Cleartext PaCo encodings: the circle-group embedding `psi(a) =
//! exp(2*pi*i*a/q)`, the complex scalar it lives on, and the public-coefficient
//! encodings (getCoeffEnc, Algorithm 3) consumed by the blind rotation.

pub(crate) mod coeff_enc;
pub(crate) mod cpx;
