//! Bounded large-ring tensor coverage for emulated CI.
//!
//! The exhaustive suites in `tests.rs` remain enabled for native runs. These
//! focused cases exercise every rank-one specialization (degree 2^15 or 2^16),
//! with partial top limbs and both aligned and unaligned convolution offsets.
//! Small-ring suites retain the complete rank, precision and offset sweeps.

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

tensor_case!(ntt4x30_n15, poulpy_cpu_ref::NTT4x30Ref, crate::NTT4x30Avx512, 15);
tensor_case!(ntt4x30_n16, poulpy_cpu_ref::NTT4x30Ref, crate::NTT4x30Avx512, 16);

#[cfg(feature = "enable-rayon")]
tensor_case!(ntt4x30_rayon_n15, crate::NTT4x30Avx512, crate::NTT4x30Avx512Rayon, 15);

#[cfg(feature = "enable-ifma")]
tensor_case!(ntt3x42_ifma_n15, poulpy_cpu_ref::NTT4x30Ref, crate::NTT3x42Ifma, 15);
#[cfg(feature = "enable-ifma")]
tensor_case!(ntt3x42_ifma_n16, poulpy_cpu_ref::NTT4x30Ref, crate::NTT3x42Ifma, 16);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
tensor_case!(ntt3x42_ifma_rayon_n15, crate::NTT3x42Ifma, crate::NTT3x42IfmaRayon, 15);
