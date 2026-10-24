//! Backend-generic libm test suites, instantiated per backend in `backend_tests`.

pub mod approximation;
pub mod arithmetic;
pub mod compare;
pub mod divsqrt;
pub mod erf;
pub mod exp;
mod helpers;
pub mod hyperbolic;
pub mod inverse;
pub mod inverse_trig;
pub mod log;
pub mod pow;
pub mod predicate;
pub mod reduce;
pub mod roots;
pub mod sign;
pub mod softmax;
pub mod special;
pub mod trig;

#[macro_export]
macro_rules! ckks_libm_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        scalar = $scalar:ty,
        encoder = $encoder:ty,
        params = $params:expr $(,)?
    ) => {
        mod $modname {
            use std::sync::LazyLock;

            use poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> =
                LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        use $path as test_fn;
                        assert_eq!($params.n, 256);
                        test_fn::<$backend, $scalar, $encoder>($params, &MODULE, &HOST_MODULE);
                    }
                };
            }

            run_test!(
                sign_composite,
                $crate::test_suite::sign::test_sign_composite
            );
            run_test!(
                step_composite,
                $crate::test_suite::sign::test_step_composite
            );
            run_test!(sign_minimax, $crate::test_suite::sign::test_sign_minimax);
            run_test!(max_min, $crate::test_suite::compare::test_max_min);
            run_test!(arithmetic, $crate::test_suite::arithmetic::test_arithmetic);
            run_test!(
                approximation,
                $crate::test_suite::approximation::test_approximation
            );
            run_test!(
                precision_tuning,
                $crate::test_suite::approximation::test_precision_tuning
            );
            run_test!(
                goldschmidt_division,
                $crate::test_suite::inverse::test_goldschmidt_division
            );
            run_test!(rsqrt, $crate::test_suite::inverse::test_rsqrt);
            run_test!(
                inverse_negative_domain,
                $crate::test_suite::inverse::test_inverse_negative_domain
            );
            run_test!(
                inverse_full_domain,
                $crate::test_suite::inverse::test_inverse_full_domain
            );
            run_test!(
                interval_normalization,
                $crate::test_suite::inverse::test_interval_normalization
            );
            run_test!(fabs, $crate::test_suite::predicate::test_fabs);
            run_test!(
                fdim_copysign,
                $crate::test_suite::predicate::test_fdim_copysign
            );
            run_test!(compare, $crate::test_suite::predicate::test_compare);
            run_test!(
                indicator_eq,
                $crate::test_suite::predicate::test_indicator_eq
            );
            run_test!(select, $crate::test_suite::predicate::test_select);
            run_test!(div, $crate::test_suite::divsqrt::test_div);
            run_test!(sqrt, $crate::test_suite::divsqrt::test_sqrt);
            run_test!(exp, $crate::test_suite::exp::test_exp);
            run_test!(exp2_expm1, $crate::test_suite::exp::test_exp2_expm1);
            run_test!(exp10, $crate::test_suite::exp::test_exp10);
            run_test!(erf_family, $crate::test_suite::erf::test_erf_family);
            run_test!(log_family, $crate::test_suite::log::test_log_family);
            run_test!(pow, $crate::test_suite::pow::test_pow);
            run_test!(powi, $crate::test_suite::pow::test_powi);
            run_test!(root_family, $crate::test_suite::roots::test_root_family);
            run_test!(sum_slots, $crate::test_suite::reduce::test_sum_slots);
            run_test!(
                extrema_slots,
                $crate::test_suite::reduce::test_extrema_slots
            );
            run_test!(softmax, $crate::test_suite::softmax::test_softmax);
            run_test!(
                special_functions,
                $crate::test_suite::special::test_special_functions
            );
            run_test!(trig_family, $crate::test_suite::trig::test_trig_family);
            run_test!(
                inverse_trig_family,
                $crate::test_suite::inverse_trig::test_inverse_trig_family
            );
            run_test!(atan2, $crate::test_suite::inverse_trig::test_atan2);
            run_test!(
                hyperbolic_family,
                $crate::test_suite::hyperbolic::test_hyperbolic_family
            );
            run_test!(
                inverse_hyperbolic,
                $crate::test_suite::hyperbolic::test_inverse_hyperbolic
            );
        }
    };
}
