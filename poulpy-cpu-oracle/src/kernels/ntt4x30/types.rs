use super::primes::Primes30;

/// Four canonical residues, one for each NTT prime.
pub type Q120bScalar = poulpy_hal::layouts::CrtWord<Primes30, u64>;
