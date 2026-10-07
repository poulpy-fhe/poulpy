//! Bounded large-ring tensor coverage for emulated CI.
//!
//! Native suites in `tests.rs` sweep precisions and offsets at rank one. These
//! focused cases run the rank-one, prepared and streaming specializations at
//! large degrees, with partial top limbs and both aligned and unaligned
//! convolution offsets. Unit tests lower the specialization thresholds, so the
//! small-ring suites take the same paths with their complete sweeps.

use poulpy_core::{
    layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision},
    test_suite::parity::test_glwe_tensor_parity_for_layout,
};
use poulpy_hal::layouts::Module;

macro_rules! tensor_case {
    ($name:ident, $comparison:ty, $tested:ty, $log_n:literal) => {
        #[test]
        fn $name() {
            let layout = GLWELayout {
                n: Degree(1 << $log_n),
                base2k: Base2K(52),
                k: TorusPrecision(53),
                rank: Rank(1),
            };
            let comparison = Module::<$comparison>::new(u64::from(layout.n.0));
            let tested = Module::<$tested>::new(u64::from(layout.n.0));
            test_glwe_tensor_parity_for_layout(&layout, &[0, 51], &comparison, &tested);
        }
    };
}

tensor_case!(ntt4x30_n15, poulpy_cpu_avx::NTT4x30Avx, crate::NTT4x30Avx512, 15);
tensor_case!(ntt4x30_n16, poulpy_cpu_avx::NTT4x30Avx, crate::NTT4x30Avx512, 16);

#[cfg(feature = "enable-rayon")]
tensor_case!(ntt4x30_rayon_n15, crate::NTT4x30Avx512, crate::NTT4x30Avx512Rayon, 15);

#[cfg(feature = "enable-ifma")]
tensor_case!(ntt3x42_ifma_n15, crate::NTT4x30Avx512, crate::NTT3x42Ifma, 15);
#[cfg(feature = "enable-ifma")]
tensor_case!(ntt3x42_ifma_n16, crate::NTT4x30Avx512, crate::NTT3x42Ifma, 16);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
tensor_case!(ntt3x42_ifma_rayon_n15, crate::NTT3x42Ifma, crate::NTT3x42IfmaRayon, 15);
