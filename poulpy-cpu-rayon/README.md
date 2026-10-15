# poulpy-cpu-rayon

Rayon-scheduled task executor shared by the Poulpy CPU backends.

This crate implements `poulpy_hal::execution::TaskExecutor` on top of the active
Rayon thread pool and provides the wrapper macros that turn a serial CPU backend
into its `*Rayon` counterpart. It is enabled through the `enable-rayon` feature
of `poulpy-cpu-avx`, `poulpy-cpu-avx512` and `poulpy-cpu-arm`; applications
select a parallel backend by its marker type rather than by depending on this
crate directly.

It also defines `FFT64PortableRayon` and `NTT4x30PortableRayon`, the Rayon variants
of the portable backends, with their conjugate-invariant aliases. They live here
because this crate depends on `poulpy-cpu-portable`. The `enable-core`,
`enable-ckks`, `enable-bin-fhe` and `enable-mhe` features wire the scheme layers
into them.
