# poulpy-cpu-rayon

Rayon-scheduled task executor shared by the Poulpy CPU backends.

This crate implements `poulpy_hal::execution::TaskExecutor` on top of the active
Rayon thread pool and provides the wrapper macros that turn a serial CPU backend
into its `*Rayon` counterpart. It is enabled through the `enable-rayon` feature
of `poulpy-cpu-avx`, `poulpy-cpu-avx512` and `poulpy-cpu-arm`; applications
select a parallel backend by its marker type rather than by depending on this
crate directly.
