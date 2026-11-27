/// The degree of a precomputed negacyclic transform table over `Z/q`.
///
/// ```text
/// op         ZqN::n()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the positive power-of-two transform degree n
/// test       test_zq_ntt
/// ```
pub trait ZqN {
    /// Returns the transform degree `n`.
    fn n(&self) -> usize;
}

/// The forward negacyclic number-theoretic transform of a precomputed table over `Z/q`.
pub trait ZqNTT<W>: ZqN {
    /// Applies the forward negacyclic transform in place.
    ///
    /// ```text
    /// op         ZqNTT::zq_ntt(data)
    /// class      basis
    /// mutation   in-place
    /// definition data[r] = sum_{0 <= t < n} old(data)[t] * psi^((2 rev_n(r) + 1) t) mod q for every 0 <= r < n
    /// domain     data: n residues in [0, q)
    /// ensures    every element of data is written in [0, q); the result is the evaluation of old(data) at the n roots of X^n + 1 in bit-reversed order
    /// test       test_zq_ntt, test_zq_ntt_rejects_wrong_length
    /// ```
    fn zq_ntt(&self, data: &mut [W]);
}

/// The normalized inverse negacyclic number-theoretic transform of a precomputed table over `Z/q`.
pub trait ZqINTT<W>: ZqN {
    /// Applies the inverse negacyclic transform in place.
    ///
    /// ```text
    /// op         ZqINTT::zq_intt(data)
    /// class      basis
    /// mutation   in-place
    /// definition data[t] = n^(-1) * sum_{0 <= r < n} old(data)[r] * psi^(-(2 rev_n(r) + 1) t) mod q for every 0 <= t < n
    /// domain     data: n residues in [0, q)
    /// ensures    every element of data is written in [0, q); zq_intt after zq_ntt returns the original residues exactly
    /// test       test_zq_ntt, test_zq_ntt_rejects_wrong_length
    /// ```
    fn zq_intt(&self, data: &mut [W]);
}

/// Constructs a precomputed negacyclic transform table over `Z/q`.
///
/// ```text
/// op         ZqNew::new(n, q, psi)
/// class      support
/// mutation   none
/// domain     n: a positive power of two; q: odd and greater than one; psi: psi^n = -1 mod q
/// ensures    returns a table of degree n for q and psi; panics outside the domain
/// test       test_zq_new_rejects
/// ```
pub trait ZqNew<W>: ZqNTT<W> + ZqINTT<W> + Sized {
    /// Returns a table of degree `n` modulo `q` for the root `psi`.
    fn new(n: usize, q: &W, psi: &W) -> Self;
}
