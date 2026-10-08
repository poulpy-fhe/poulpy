use poulpy_core::test_suite::parity::{ParityShapes, test_tensor_relinearize_decrypt_parity};
use poulpy_hal::{layouts::Module, test_suite::TestParams};

macro_rules! one_pass_case {
    ($name:ident, $tested:ty, $oracle:ty, $n:expr, $module_n:expr) => {
        #[test]
        fn $name() {
            test_tensor_relinearize_decrypt_parity(
                &TestParams {
                    n: $n,
                    size: $module_n,
                    base2k: 52,
                },
                &ParityShapes {
                    ranks: vec![1],
                    dsizes: Some(vec![1, 2]),
                },
                &Module::<$oracle>::new($module_n as u64),
                &Module::<$tested>::new($module_n as u64),
            );
        }
    };
}

one_pass_case!(ifma_fallback, crate::NTT3x42Ifma, poulpy_cpu_oracle::NTT4x30Oracle, 256, 512);
one_pass_case!(ifma_n16, crate::NTT3x42Ifma, poulpy_cpu_oracle::NTT4x30Oracle, 65536, 65536);
one_pass_case!(
    ifma_ci_n16,
    crate::NTT3x42CIIfma,
    poulpy_cpu_oracle::NTT4x30CIOracle,
    65536,
    65536
);
#[cfg(feature = "enable-rayon")]
one_pass_case!(
    ifma_rayon_fallback,
    crate::NTT3x42IfmaRayon,
    poulpy_cpu_oracle::NTT4x30Oracle,
    256,
    512
);
#[cfg(feature = "enable-rayon")]
one_pass_case!(
    ifma_rayon_n16,
    crate::NTT3x42IfmaRayon,
    poulpy_cpu_oracle::NTT4x30Oracle,
    65536,
    65536
);
#[cfg(feature = "enable-rayon")]
one_pass_case!(
    ifma_ci_rayon_n16,
    crate::NTT3x42CIIfmaRayon,
    poulpy_cpu_oracle::NTT4x30CIOracle,
    65536,
    65536
);
