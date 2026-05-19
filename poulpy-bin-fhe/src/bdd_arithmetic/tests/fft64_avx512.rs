use std::sync::LazyLock;

use poulpy_cpu_avx512::FFT64Avx512;

use crate::{bdd_arithmetic::tests::test_suite, blind_rotation::CGGI};

static TEST_CONTEXT_CGGI_FFT64_IFMA: LazyLock<test_suite::TestContext<CGGI, FFT64Avx512>> =
    LazyLock::new(test_suite::TestContext::<CGGI, FFT64Avx512>::new);

#[test]
fn glwe_blind_retriever() {
    test_suite::test_glwe_blind_retriever(&TEST_CONTEXT_CGGI_FFT64_IFMA);
}

#[test]
fn glwe_blind_retrieval_statefull() {
    test_suite::test_glwe_blind_retrieval_statefull(&TEST_CONTEXT_CGGI_FFT64_IFMA);
}

#[test]
fn fhe_uint_swap() {
    test_suite::test_fhe_uint_swap(&TEST_CONTEXT_CGGI_FFT64_IFMA);
}

#[test]
fn fhe_uint_get_bit_glwe() {
    test_suite::test_fhe_uint_get_bit_glwe(&TEST_CONTEXT_CGGI_FFT64_IFMA);
}

#[test]
fn fhe_uint_sext() {
    test_suite::test_fhe_uint_sext(&TEST_CONTEXT_CGGI_FFT64_IFMA);
}

#[test]
fn glwe_blind_selection() {
    test_suite::test_glwe_blind_selection(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn fhe_uint_splice_u8() {
    test_suite::test_fhe_uint_splice_u8(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn fhe_uint_splice_u16() {
    test_suite::test_fhe_uint_splice_u16(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn glwe_to_glwe_blind_rotation() {
    test_suite::test_glwe_to_glwe_blind_rotation(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn scalar_to_ggsw_blind_rotation() {
    test_suite::test_scalar_to_ggsw_blind_rotation(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_prepare() {
    test_suite::test_bdd_prepare(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_add() {
    test_suite::test_bdd_add(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_and() {
    test_suite::test_bdd_and(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_or() {
    test_suite::test_bdd_or(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_sll() {
    test_suite::test_bdd_sll(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_slt() {
    test_suite::test_bdd_slt(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_sltu() {
    test_suite::test_bdd_sltu(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_sra() {
    test_suite::test_bdd_sra(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_srl() {
    test_suite::test_bdd_srl(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_sub() {
    test_suite::test_bdd_sub(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}

#[test]
fn bdd_xor() {
    test_suite::test_bdd_xor(&TEST_CONTEXT_CGGI_FFT64_IFMA)
}
