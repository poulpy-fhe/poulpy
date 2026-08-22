use std::sync::LazyLock;

use poulpy_cpu_avx::FFT64AvxRayon;

use crate::{bdd_arithmetic::tests::test_suite, blind_rotation::CGGI};

static TEST_CONTEXT: LazyLock<test_suite::TestContext<CGGI, FFT64AvxRayon>> =
    LazyLock::new(test_suite::TestContext::<CGGI, FFT64AvxRayon>::new);

#[test]
fn bdd_add_single_and_multi_worker_agree() {
    test_suite::test_bdd_add(&TEST_CONTEXT);
}
