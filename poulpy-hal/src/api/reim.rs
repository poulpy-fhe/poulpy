/// The transform length of a precomputed complex transform table with separate real and imaginary halves.
///
/// ```text
/// op         ReimM::m()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the positive power-of-two complex transform length m
/// test       none
/// ```
pub trait ReimM {
    /// Returns the complex transform length `m`.
    fn m(&self) -> usize;
}

/// The forward complex transform of a precomputed table.
pub trait ReimFFT<F>: ReimM {
    /// Applies the forward complex transform in place.
    ///
    /// ```text
    /// op         ReimFFT::reim_fft(data)
    /// class      basis
    /// mutation   in-place
    /// definition z(data,r) = sum_{0 <= t < m} z(old(data),t) * omega(m,r)^t for every 0 <= r < m
    /// domain     data: 2*m real scalars, with real parts in data[0..m] and imaginary parts in data[m..2*m]
    /// ensures    every element of data is written with the forward transform
    /// test       test_reim_fft
    /// ```
    fn reim_fft(&self, data: &mut [F]);
}

/// The unnormalized inverse complex transform of a precomputed table.
pub trait ReimIFFT<F>: ReimM {
    /// Applies the unnormalized inverse complex transform in place.
    ///
    /// ```text
    /// op         ReimIFFT::reim_ifft(data)
    /// class      basis
    /// mutation   in-place
    /// definition z(data,t) = sum_{0 <= r < m} z(old(data),r) * omega(m,r)^(-t) for every 0 <= t < m
    /// domain     data: 2*m real scalars, with real parts in data[0..m] and imaginary parts in data[m..2*m]
    /// ensures    applying reim_ifft after reim_fft multiplies each original element by m, up to floating arithmetic error
    /// test       test_reim_fft
    /// ```
    fn reim_ifft(&self, data: &mut [F]);
}

/// Constructs a precomputed complex transform table.
///
/// ```text
/// op         ReimNew::new(m)
/// class      support
/// mutation   none
/// domain     m: a positive power of two
/// ensures    returns a table of complex transform length m
/// test       none
/// ```
pub trait ReimNew<F>: ReimFFT<F> + ReimIFFT<F> + Sized {
    /// Returns a table of complex transform length `m`.
    fn new(m: usize) -> Self;
}
