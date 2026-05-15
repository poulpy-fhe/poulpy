//! # poulpy-hal
//!
//! A trait-based Hardware Abstraction Layer (HAL) for lattice-based polynomial
//! arithmetic over the cyclotomic ring `Z[X]/(X^N + 1)`.
//!
//! This crate provides backend-agnostic data layouts and a trait-based API for
//! polynomial operations commonly used in lattice-based cryptography (LWE/Module-LWE
//! ciphertexts, key-switching matrices, external products, etc.). It is designed
//! so that cryptographic schemes can be written once against the [`api`] traits and
//! then executed on any backend (CPU with AVX2/AVX-512, GPU, FPGA, ...) that
//! implements the [`oep`] (Open Extension Point) traits.
//!
//! ## Core Concepts
//!
//! **Ring:** All polynomials live in `Z[X]/(X^N + 1)` where `N` is a power of
//! two (the *ring degree*). A [`layouts::Module`] encapsulates `N` together with
//! an optional backend-specific handle (e.g. precomputed FFT twiddle factors).
//!
//! **Limbed representation (base-2^k):** Large coefficients are decomposed into
//! a vector of `size` limbs, each carrying at most `base2k` bits. This is the
//! *bivariate* view `Z[X, Y]` with `Y = 2^{-k}`, central to gadget
//! decomposition and normalization.
//!
//! **Layout types** ([`layouts`]):
//! - [`layouts::ScalarZnx`] -- single polynomial with `i64` coefficients.
//! - [`layouts::VecZnx`] -- vector of `cols` polynomials, each with `size` limbs.
//! - [`layouts::MatZnx`] -- matrix of polynomials (`rows x cols_in`, each entry a [`layouts::VecZnx`] of `cols_out` polynomials).
//! - [`layouts::VecZnxBig`] -- vector of polynomials with backend-specific large-coefficient scalars (result accumulator).
//! - [`layouts::VecZnxDft`] -- vector of polynomials in DFT/NTT domain (backend-specific prepared scalars).
//! - [`layouts::SvpPPol`] -- prepared scalar polynomial for scalar-vector products.
//! - [`layouts::VmpPMat`] -- prepared matrix for vector-matrix products.
//! - [`layouts::CnvPVecL`], [`layouts::CnvPVecR`] -- prepared left/right operands for bivariate convolution.
//! - [`layouts::ScratchArena`], [`layouts::ScratchOwned`] -- aligned scratch memory for temporary workspace.
//!
//! All layout types are generic over a data container `D` (owned `Vec<u8>`, borrowed
//! `&[u8]` / `&mut [u8]`), enabling zero-copy views and arena-style allocation via
//! [`layouts::ScratchArena`].
//!
//! ## Architecture
//!
//! The crate is organized into a four-layer stack:
//!
//! 1. **[`api`]** -- Safe, user-facing trait definitions (e.g. [`api::VecZnxAddIntoBackend`],
//!    [`api::VmpApplyDftToDft`]). Scheme authors program against these.
//! 2. **[`oep`]** -- Unsafe extension-point layer of per-family backend traits.
//!    Backend crates implement only the families they own and may reuse helper
//!    macros or defaults where convenient.
//! 3. **[`delegates`]** -- Blanket `impl` glue that connects each [`api`] trait to
//!    the corresponding backend family method on [`layouts::Module`].
//! 4. **Reference implementations** live in the `poulpy-cpu-ref` crate, which provides
//!    the portable default backend used by tests and benchmarks.
//!
//! ## Testing and Benchmarking
//!
//! The `poulpy-bench` crate provides generic, backend-parametric test helpers
//! (`test_suite`) and Criterion benchmark harnesses (`bench_suite`). Backend
//! crates instantiate tests via the [`backend_test_suite!`] and
//! [`cross_backend_test_suite!`] macros to validate correctness against the
//! reference implementation.
//!
//! ## Safety Contract
//!
//! All [`oep`] extension points are `unsafe` to implement. Implementors must uphold the
//! contract documented in [`doc::backend_safety`], covering memory domains,
//! alignment, scratch lifetime, synchronization, aliasing, and numerical
//! exactness.
//!
//! ## Non-Goals
//!
//! - This crate does **not** provide a complete cryptographic scheme. It is a
//!   low-level arithmetic layer consumed by higher-level crates such as
//!   `poulpy-core` and `poulpy-bin-fhe`.
//! - It does **not** perform constant-time enforcement. Side-channel resistance
//!   is the responsibility of the backend and the caller.
//!
//! ## Compatibility
//!
//! - Requires **nightly** Rust (uses `#![feature(trait_alias)]`).
//! - All memory allocations are aligned to [`DEFAULTALIGN`] (64 bytes).
//! - Types matching the API of **spqlios-arithmetic**.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![feature(trait_alias)]

/// Safe, user-facing trait definitions for polynomial arithmetic operations.
///
/// Scheme authors program against these traits; the actual computation is
/// dispatched to a backend via the [`oep`] extension points.
pub mod api;

/// Criterion-based benchmark harnesses, generic over any backend.
/// Blanket implementations connecting [`api`] traits to [`oep`] traits on
/// [`layouts::Module`].
///
/// This module contains no user-facing logic; it exists solely to wire
/// the safe API layer to the unsafe backend implementations.
pub mod delegates;

/// Backend-agnostic data layout types for polynomials, vectors, matrices,
/// and prepared (DFT-domain) representations.
///
/// All types are generic over a data container `D` (`Vec<u8>`, `&[u8]`,
/// `&mut [u8]`) enabling owned, borrowed, and scratch-backed usage.
pub mod layouts;

/// Open Extension Points: the `unsafe` backend extension layer of per-family
/// backend traits.
///
/// Backend crates implement only the families they own and may delegate to
/// helper defaults provided by a backend crate (for example `poulpy-cpu-ref`). See
/// [`doc::backend_safety`] for the safety contract.
pub mod oep;

/// Deterministic pseudorandom number generation based on ChaCha8.
pub mod source;

/// Fully generic, backend-parametric test functions.
///
/// Backend crates instantiate these via the [`backend_test_suite!`] and
/// [`cross_backend_test_suite!`] macros.
pub mod test_suite;

/// Embedded safety contract documentation for backend implementors.
pub mod doc {
    /// Safety contract that all [`crate::oep`] trait implementations must uphold.
    ///
    /// Covers memory domains, alignment, scratch lifetime, synchronization,
    /// aliasing, and numerical exactness requirements.
    #[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/backend_safety_contract.md"))]
    pub mod backend_safety {
        pub const _PLACEHOLDER: () = ();
    }
}

/// Default generator of the Galois group `(Z/2NZ)*` for the cyclotomic ring
/// `Z[X]/(X^N + 1)`.
///
/// Used to compute Galois automorphisms `X -> X^{5^k}` and their inverses.
pub const GALOISGENERATOR: u64 = 5;

/// Default memory alignment in bytes for all allocated buffers.
///
/// Set to 64 bytes to match the cache-line size of modern x86 processors
/// and the alignment required by AVX-512 instructions.
pub const DEFAULTALIGN: usize = 64;

fn is_aligned_custom<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize).is_multiple_of(align)
}

/// Returns `true` if `ptr` is aligned to [`DEFAULTALIGN`] bytes.
pub fn is_aligned<T>(ptr: *const T) -> bool {
    is_aligned_custom(ptr, DEFAULTALIGN)
}

/// Panics if `ptr` is not aligned to [`DEFAULTALIGN`] bytes.
///
/// # Panics
///
/// Panics with a descriptive message when the pointer does not satisfy the
/// default alignment requirement.
pub fn assert_alignment<T>(ptr: *const T) {
    assert!(
        is_aligned(ptr),
        "invalid alignment: ensure passed bytes have been allocated with [alloc_aligned_u8] or [alloc_aligned]"
    )
}

/// Deprecated spelling variant. Use [`assert_alignment`] instead.
#[inline]
pub fn assert_alignement<T>(ptr: *const T) {
    assert_alignment(ptr)
}

/// Reinterprets a `&[T]` as a `&[V]`.
///
/// # Safety (via assertions)
/// - `V` must not be zero-sized.
/// - The pointer must be aligned for `V`.
/// - The total byte length must be a multiple of `size_of::<V>()`.
pub fn cast<T, V>(data: &[T]) -> &[V] {
    assert!(size_of::<V>() > 0, "cast: target type V must not be zero-sized");
    let byte_len: usize = std::mem::size_of_val(data);
    assert!(
        byte_len.is_multiple_of(size_of::<V>()),
        "cast: byte length {} is not a multiple of target size {}",
        byte_len,
        size_of::<V>()
    );
    let ptr: *const V = data.as_ptr() as *const V;
    assert!(
        ptr.align_offset(align_of::<V>()) == 0,
        "cast: pointer {:p} is not aligned to {} bytes",
        ptr,
        align_of::<V>()
    );
    let len: usize = byte_len / size_of::<V>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Reinterprets a `&mut [T]` as a `&mut [V]`.
///
/// # Safety (via assertions)
/// - `V` must not be zero-sized.
/// - The pointer must be aligned for `V`.
/// - The total byte length must be a multiple of `size_of::<V>()`.
pub fn cast_mut<T, V>(data: &mut [T]) -> &mut [V] {
    assert!(size_of::<V>() > 0, "cast_mut: target type V must not be zero-sized");
    let byte_len: usize = std::mem::size_of_val(data);
    assert!(
        byte_len.is_multiple_of(size_of::<V>()),
        "cast_mut: byte length {} is not a multiple of target size {}",
        byte_len,
        size_of::<V>()
    );
    let ptr: *mut V = data.as_mut_ptr() as *mut V;
    assert!(
        ptr.align_offset(align_of::<V>()) == 0,
        "cast_mut: pointer {:p} is not aligned to {} bytes",
        ptr,
        align_of::<V>()
    );
    let len: usize = byte_len / size_of::<V>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Allocates a block of bytes with a custom alignment.
/// Alignment must be a power of two and size a multiple of the alignment.
/// Allocated memory is initialized to zero.
///
/// # Known issue (CRITICAL-2)
/// The returned `Vec<u8>` was allocated with custom alignment via `std::alloc::alloc`,
/// but `Vec::drop` will call `std::alloc::dealloc` with `align_of::<u8>() = 1`.
/// This is technically UB per the `GlobalAlloc` contract (mismatched layout).
/// In practice it works on all major allocators (glibc, jemalloc, mimalloc) because
/// they ignore the alignment parameter during deallocation. A proper fix requires
/// replacing `Vec<u8>` with a custom `AlignedBuf` type that tracks the layout.
fn alloc_aligned_custom_u8(size: usize, align: usize) -> Vec<u8> {
    assert!(align.is_power_of_two(), "Alignment must be a power of two but is {align}");
    assert_eq!(
        (size * size_of::<u8>()) % align,
        0,
        "size={size} must be a multiple of align={align}"
    );
    unsafe {
        let layout: std::alloc::Layout = std::alloc::Layout::from_size_align(size, align).expect("Invalid alignment");
        let ptr: *mut u8 = std::alloc::alloc(layout);
        if ptr.is_null() {
            panic!("Memory allocation failed");
        }
        assert!(
            is_aligned_custom(ptr, align),
            "Memory allocation at {ptr:p} is not aligned to {align} bytes"
        );
        // Init allocated memory to zero
        std::ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Allocates a zero-initialized `Vec<T>` with custom alignment.
///
/// The total byte size (`size * size_of::<T>()`) must be a multiple of `align`,
/// and `align` must be a power of two.
///
/// # Panics
///
/// - If `T` is zero-sized.
/// - If `align` is not a power of two.
/// - If `size * size_of::<T>()` is not a multiple of `align`.
pub fn alloc_aligned_custom<T>(size: usize, align: usize) -> Vec<T> {
    assert!(size_of::<T>() > 0, "alloc_aligned_custom: zero-sized types are not supported");
    assert!(align.is_power_of_two(), "Alignment must be a power of two but is {align}");

    assert_eq!(
        (size * size_of::<T>()) % align,
        0,
        "size*size_of::<T>()={} must be a multiple of align={align}",
        size * size_of::<T>(),
    );

    let mut vec_u8: Vec<u8> = alloc_aligned_custom_u8(size_of::<T>() * size, align);
    let ptr: *mut T = vec_u8.as_mut_ptr() as *mut T;
    let len: usize = vec_u8.len() / size_of::<T>();
    let cap: usize = vec_u8.capacity() / size_of::<T>();
    std::mem::forget(vec_u8);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Allocates a zero-initialized `Vec<T>` aligned to [`DEFAULTALIGN`] bytes.
///
/// The allocation is padded so that the total byte size is a multiple of
/// [`DEFAULTALIGN`]. This is the primary allocation entry point for all
/// layout types in the crate.
///
/// # Panics
///
/// Panics if `T` is zero-sized.
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        (size * size_of::<T>()).next_multiple_of(DEFAULTALIGN) / size_of::<T>(),
        DEFAULTALIGN,
    )
}
