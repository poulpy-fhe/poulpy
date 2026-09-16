/// Abstract precomputed table for the negacyclic (reim) FFT and its inverse.
///
/// Implementors hold precomputed twiddle factors for ring degree `m` and expose
/// in-place forward (`fft`) and inverse (`ifft`) transforms over a data slice of
/// length `2m` in the real/imaginary interleaved layout.
///
/// ```text
/// op         NegacyclicFFT::{m, fft, ifft}(data)
/// class      support
/// mutation   in-place
/// domain     data: 2m floats in the table's interleaved real/imaginary layout
/// ensures    fft is the negacyclic forward transform for degree m and ifft its inverse, both in place; the pair round-trips up to the float error of the implementation
/// test       none
/// ```
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
///
/// ```text
/// op         NegacyclicFFTNew::new(m)
/// class      support
/// mutation   none
/// domain     m: a power of two
/// ensures    returns a table of twiddle factors for degree m; split from NegacyclicFFT so a bound can ask for "transforms" without asking for "builds a table"
/// test       none
/// ```
pub trait NegacyclicFFTNew<F>: NegacyclicFFT<F> + Sized {
    fn new(m: usize) -> Self;
}
