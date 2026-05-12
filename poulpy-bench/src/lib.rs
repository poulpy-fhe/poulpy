#![cfg(any(
    feature = "hal-bench",
    feature = "core-bench",
    feature = "bin-fhe-bench",
    feature = "ckks-bench"
))]

//! Shared helpers for `poulpy-bench` benchmark binaries.
//!
//! # Public dispatch macros
//!
//! Three macros are exported for bench files to use:
//!
//! ```text
//! for_each_fft_backend!(path [, leading_arg]* ; criterion_ref)
//! for_each_ntt_backend!(path [, leading_arg]* ; criterion_ref)
//! for_each_backend!(path [, leading_arg]* ; criterion_ref)
//! ```
//!
//! Each expands to one call per matching backend:
//!
//! ```text
//! path::<BackendType>(leading_args..., criterion_ref, "backend-label");
//! ```
//!
//! Use:
//! - `for_each_fft_backend!` for FFT64-specific operations (`fft` primitive benches)
//! - `for_each_ntt_backend!` for NTT-family-specific operations (`ntt` primitive benches)
//! - `for_each_backend!` for operations that work with any backend (generic GLWE ops, vec_znx, etc.)
//!
//! # Adding a new backend
//!
//! 1. Add the crate to `[dependencies]` in `poulpy-bench/Cargo.toml` (optionally behind a feature).
//! 2. Add one `#[cfg(...)] { use $fn as __f; __f::<NewType>(...); }` line to the appropriate
//!    private family macro below (`for_each_fft_backend_family!` or `for_each_ntt_backend_family!`).
//! 3. No bench files need to change.

pub mod bench_suite;
pub mod params;

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
use poulpy_core::{
    api::ModuleTransfer,
    layouts::{GGSW, GLWE, GLWEPlaintext, LWE, LWESecret},
};
use poulpy_hal::layouts::{
    Backend, CnvPVecL, CnvPVecR, DataView, MatZnx, MatZnxBackendRef, MatZnxToBackendRef, ScalarZnx, ScalarZnxBackendRef,
    ScalarZnxToBackendRef, SvpPPol, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
    VecZnxBigBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef,
    VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, VmpPMat,
};
#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
use poulpy_hal::layouts::{Module, TransferFrom};
use poulpy_hal::source::Source;
use rand::Rng;

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
type BenchHostBackend = poulpy_cpu_ref::FFT64Ref;

fn random_aligned_host_bytes(len: usize, source: &mut Source) -> Vec<u8> {
    let mut bytes = poulpy_hal::alloc_aligned_custom::<u8>(len, poulpy_hal::DEFAULTALIGN);
    source.fill_bytes(&mut bytes);
    bytes
}

/// Return the shared Criterion configuration used by all bench binaries.
///
/// Uses 100 samples with a 5-second measurement budget per benchmark.
/// Fast benchmarks complete in ~5 s; for slow benchmarks whose single
/// iteration exceeds the per-sample budget Criterion automatically extends
/// the run to collect at least a few samples (it will never cut a sample
/// short), so scheme-level benchmarks (blind rotate, CBS) may take longer.
pub fn criterion_config() -> criterion::Criterion {
    criterion::Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
}

pub fn ckks_criterion_config() -> criterion::Criterion {
    criterion_config()
}

pub fn upload_host_vec_znx<BE: Backend>(src: &VecZnx<Vec<u8>>) -> VecZnx<BE::OwnedBuf> {
    VecZnx::from_data_with_max_size(
        BE::from_host_bytes(src.data()),
        src.n(),
        src.cols(),
        src.size(),
        src.max_size(),
    )
}

pub fn upload_host_scalar_znx<BE: Backend>(src: &ScalarZnx<Vec<u8>>) -> ScalarZnx<BE::OwnedBuf> {
    ScalarZnx::from_data(BE::from_host_bytes(src.data()), src.n(), src.cols())
}

pub fn upload_host_mat_znx<BE: Backend>(src: &MatZnx<Vec<u8>>) -> MatZnx<BE::OwnedBuf> {
    MatZnx::from_data(
        BE::from_host_bytes(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

pub fn random_host_scalar_znx(n: usize, cols: usize, source: &mut Source) -> ScalarZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n, cols), source);
    ScalarZnx::from_bytes(n, cols, bytes)
}

pub fn random_host_vec_znx(n: usize, cols: usize, size: usize, source: &mut Source) -> VecZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size), source);
    VecZnx::from_bytes(n, cols, size, bytes)
}

pub fn random_host_mat_znx(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> MatZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(MatZnx::<Vec<u8>>::bytes_of(n, rows, cols_in, cols_out, size), source);
    MatZnx::from_bytes(n, rows, cols_in, cols_out, size, bytes)
}

pub fn random_backend_vec_znx_dft<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> VecZnxDft<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vec_znx_dft(n, cols, size)];
    source.fill_bytes(&mut bytes);
    VecZnxDft::from_bytes(n, cols, size, bytes)
}

pub fn random_backend_vec_znx_big<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> VecZnxBig<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vec_znx_big(n, cols, size)];
    source.fill_bytes(&mut bytes);
    VecZnxBig::from_bytes(n, cols, size, bytes)
}

pub fn random_backend_svp_ppol<BE: Backend>(n: usize, cols: usize, source: &mut Source) -> SvpPPol<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_svp_ppol(n, cols)];
    source.fill_bytes(&mut bytes);
    SvpPPol::from_data(BE::from_host_bytes(&bytes), n, cols)
}

pub fn random_backend_vmp_pmat<BE: Backend>(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> VmpPMat<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size)];
    source.fill_bytes(&mut bytes);
    VmpPMat::from_data(BE::from_host_bytes(&bytes), n, rows, cols_in, cols_out, size)
}

pub fn random_backend_cnv_pvec_left<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecL<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_left(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecL::from_bytes(n, cols, size, bytes)
}

pub fn random_backend_cnv_pvec_right<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecR<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_right(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecR::from_bytes(n, cols, size, bytes)
}

pub fn scalar_znx_backend_ref<'a, BE: Backend>(src: &'a ScalarZnx<BE::OwnedBuf>) -> ScalarZnxBackendRef<'a, BE> {
    <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_backend_ref<'a, BE: Backend>(src: &'a VecZnx<BE::OwnedBuf>) -> VecZnxBackendRef<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_backend_mut<'a, BE: Backend>(src: &'a mut VecZnx<BE::OwnedBuf>) -> VecZnxBackendMut<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(src)
}

pub fn mat_znx_backend_ref<'a, BE: Backend>(src: &'a MatZnx<BE::OwnedBuf>) -> MatZnxBackendRef<'a, BE> {
    <MatZnx<BE::OwnedBuf> as MatZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_dft_backend_ref<'a, BE: Backend>(src: &'a VecZnxDft<BE::OwnedBuf, BE>) -> VecZnxDftBackendRef<'a, BE> {
    <VecZnxDft<BE::OwnedBuf, BE> as VecZnxDftToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_dft_backend_mut<'a, BE: Backend>(src: &'a mut VecZnxDft<BE::OwnedBuf, BE>) -> VecZnxDftBackendMut<'a, BE> {
    <VecZnxDft<BE::OwnedBuf, BE> as VecZnxDftToBackendMut<BE>>::to_backend_mut(src)
}

pub fn vec_znx_big_backend_ref<'a, BE: Backend>(src: &'a VecZnxBig<BE::OwnedBuf, BE>) -> VecZnxBigBackendRef<'a, BE> {
    <VecZnxBig<BE::OwnedBuf, BE> as VecZnxBigToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_big_backend_mut<'a, BE: Backend>(src: &'a mut VecZnxBig<BE::OwnedBuf, BE>) -> VecZnxBigBackendMut<'a, BE> {
    <VecZnxBig<BE::OwnedBuf, BE> as VecZnxBigToBackendMut<BE>>::to_backend_mut(src)
}

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
pub fn upload_host_glwe<BE>(module: &Module<BE>, src: &GLWE<Vec<u8>>) -> GLWE<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<BenchHostBackend>,
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_glwe::<BenchHostBackend>(src)
}

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
pub fn upload_host_lwe<BE>(module: &Module<BE>, src: &LWE<Vec<u8>>) -> LWE<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<BenchHostBackend>,
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_lwe::<BenchHostBackend>(src)
}

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
pub fn upload_host_lwe_secret<BE>(module: &Module<BE>, src: &LWESecret<Vec<u8>>) -> LWESecret<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<BenchHostBackend>,
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_lwe_secret::<BenchHostBackend>(src)
}

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
pub fn upload_host_glwe_plaintext<BE>(module: &Module<BE>, src: &GLWEPlaintext<Vec<u8>>) -> GLWEPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<BenchHostBackend>,
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_glwe_plaintext::<BenchHostBackend>(src)
}

#[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
pub fn upload_host_ggsw<BE>(module: &Module<BE>, src: &GGSW<Vec<u8>>) -> GGSW<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<BenchHostBackend>,
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_ggsw::<BenchHostBackend>(src)
}

/// Private: expands to every FFT64 backend in tier order (ref → avx → gpu).
#[doc(hidden)]
#[macro_export]
macro_rules! for_each_fft_backend_family {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        {
            use $fn as __f;
            __f::<poulpy_cpu_ref::FFT64Ref>($($arg,)* $c, "fft64-ref");
        }
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx::FFT64Avx>($($arg,)* $c, "fft64-avx");
        }
        #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::FFT64Avx512>($($arg,)* $c, "fft64-avx512");
        }
        // #[cfg(feature = "enable-gpu")]
        // { use $fn as __f; __f::<poulpy_gpu::FFT64GPU>($($arg,)* $c, "fft64-gpu"); }
    }};
}

/// Private: expands to every NTT-family backend in tier order
/// (ntt120-ref → ntt120-avx → ntt120-avx512 → ntt-ifma → gpu).
#[doc(hidden)]
#[macro_export]
macro_rules! for_each_ntt_backend_family {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        {
            use $fn as __f;
            __f::<poulpy_cpu_ref::NTT120Ref>($($arg,)* $c, "ntt120-ref");
        }
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx::NTT120Avx>($($arg,)* $c, "ntt120-avx");
        }
        #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::NTT120Avx512>($($arg,)* $c, "ntt120-avx512");
        }
        #[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
        {
            use $fn as __f;
            __f::<poulpy_cpu_avx512::NTT126Ifma>($($arg,)* $c, "ntt-ifma");
        }
        // #[cfg(feature = "enable-gpu")]
        // { use $fn as __f; __f::<poulpy_gpu::NTT120GPU>($($arg,)* $c, "ntt120-gpu"); }
    }};
}

/// Run a bench function against every FFT64 backend.
///
/// Use for operations that are specific to the FFT64 transform domain
/// (DFT, convolution, VMP/SVP with DFT).
#[macro_export]
macro_rules! for_each_fft_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_fft_backend_family!($fn $(, $arg)* ; $c);
    }};
}

/// Run a bench function against every NTT120 backend.
///
/// Use for operations that are specific to the NTT120 transform domain.
#[macro_export]
macro_rules! for_each_ntt_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_ntt_backend_family!($fn $(, $arg)* ; $c);
    }};
}

/// Run a bench function against every available backend (FFT64 and NTT-family).
///
/// Use for operations that work with any backend: generic GLWE operations,
/// `vec_znx` / `vec_znx_big` arithmetic, encryption, decryption, key-switching, etc.
#[macro_export]
macro_rules! for_each_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        poulpy_bench::for_each_fft_backend_family!($fn $(, $arg)* ; $c);
        poulpy_bench::for_each_ntt_backend_family!($fn $(, $arg)* ; $c);
    }};
}
