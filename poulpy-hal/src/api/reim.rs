/// Abstract precomputed table for the negacyclic (reim) FFT and its inverse.
///
/// Implementors hold precomputed twiddle factors for ring degree `m` and expose
/// in-place forward (`fft`) and inverse (`ifft`) transforms over a data slice of
/// length `2m` in the real/imaginary interleaved layout.
pub trait NegacyclicFFT<F> {
    fn m(&self) -> usize;
    fn fft(&self, data: &mut [F]);
    fn ifft(&self, data: &mut [F]);
}

/// Extension of [`NegacyclicFFT`] that also provides a constructor.
///
/// Separated from the base trait so that generic bounds can distinguish between
/// "needs a precomputed table" (`NegacyclicFFTNew`) and "just needs to call fft/ifft"
/// (`NegacyclicFFT`).
pub trait NegacyclicFFTNew<F>: NegacyclicFFT<F> + Sized {
    fn new(m: usize) -> Self;
}
