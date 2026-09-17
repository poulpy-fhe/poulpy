/// A precomputed complex transform table with separate real and imaginary halves.
///
/// ```text
/// op         NegacyclicFFT::m()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the positive power-of-two complex transform length m
/// test       none
/// ```
pub trait NegacyclicFFT<F> {
    /// Returns the complex transform length `m`.
    fn m(&self) -> usize;

    /// Applies the forward complex transform in place.
    ///
    /// ```text
    /// op         NegacyclicFFT::fft(data)
    /// class      basis
    /// mutation   in-place
    /// definition z(data,r) = sum_{0 <= t < m} z(old(data),t) * omega(m,r)^t for every 0 <= r < m
    /// domain     data: 2*m real scalars, with real parts in data[0..m] and imaginary parts in data[m..2*m]
    /// ensures    every element of data is written with the forward transform
    /// test       test_negacyclic_fft
    /// ```
    fn fft(&self, data: &mut [F]);

    /// Applies the unnormalized inverse complex transform in place.
    ///
    /// ```text
    /// op         NegacyclicFFT::ifft(data)
    /// class      basis
    /// mutation   in-place
    /// definition z(data,t) = sum_{0 <= r < m} z(old(data),r) * omega(m,r)^(-t) for every 0 <= t < m
    /// domain     data: 2*m real scalars, with real parts in data[0..m] and imaginary parts in data[m..2*m]
    /// ensures    applying ifft after fft multiplies each original element by m, up to floating arithmetic error
    /// test       test_negacyclic_fft
    /// ```
    fn ifft(&self, data: &mut [F]);
}

/// Constructs a precomputed complex transform table.
///
/// ```text
/// op         NegacyclicFFTNew::new(m)
/// class      support
/// mutation   none
/// domain     m: a positive power of two
/// ensures    returns a table of complex transform length m
/// test       none
/// ```
pub trait NegacyclicFFTNew<F>: NegacyclicFFT<F> + Sized {
    /// Returns a table of complex transform length `m`.
    fn new(m: usize) -> Self;
}
