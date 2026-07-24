use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGSW, GLWE, GLWEAutomorphismKey, GLWESwitchingKey, GLWETensorKey, GLWEToLWEKey, LWE,
    LWESwitchingKey, LWEToGLWEKey, Rank, TorusPrecision,
    compressed::{
        GGLWECompressed, GGSWCompressed, GLWEAutomorphismKeyCompressed, GLWECompressed, GLWESwitchingKeyCompressed,
        GLWETensorKeyCompressed, GLWEToLWESwitchingKeyCompressed, LWECompressed, LWESwitchingKeyCompressed,
        LWEToGLWEKeyCompressed,
    },
};

const N_GLWE: Degree = Degree(64);
const N_LWE: Degree = Degree(32);
const BASE2K: Base2K = Base2K(12);
const K: TorusPrecision = TorusPrecision(33);
const DNUM: Dnum = Dnum(3);
const RANK: Rank = Rank(2);
const DSIZE: Dsize = Dsize(1);
// Auxiliary guard precision (old total `K` minus gadget `dnum * dsize * base2k`).
/// Auxiliary guard of a key: one full gadget digit (`dsize * base2k`) plus
/// `log2(n)`. Must always be at least `dsize * base2k`.
const K_KEY_AUX: TorusPrecision = TorusPrecision(DSIZE.0 * BASE2K.0 + N_GLWE.0.ilog2());

#[test]
fn glwe_serialization() {
    let original: GLWE<Vec<u8>> = GLWE::alloc(N_GLWE, BASE2K, K, RANK);
    poulpy_hal::test_suite::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let original: GLWECompressed<Vec<u8>> = GLWECompressed::alloc(N_GLWE, BASE2K, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let original: LWE<Vec<u8>> = LWE::alloc(N_LWE, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECompressed<Vec<u8>> = LWECompressed::alloc(BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let original: GGLWE<Vec<u8>> = GGLWE::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let original: GGLWECompressed<Vec<u8>> = GGLWECompressed::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let original: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GLWESwitchingKeyCompressed<Vec<u8>> =
        GLWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: GLWEAutomorphismKeyCompressed<Vec<u8>> =
        GLWEAutomorphismKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: GLWETensorKeyCompressed<Vec<u8>> = GLWETensorKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_key_serialization() {
    let original: GLWEToLWEKey<Vec<u8>> = GLWEToLWEKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> =
        GLWEToLWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_key_serialization() {
    let original: LWEToGLWEKey<Vec<u8>> = LWEToGLWEKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_key_compressed_serialization() {
    let original: LWEToGLWEKeyCompressed<Vec<u8>> = LWEToGLWEKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let original: GGSW<Vec<u8>> = GGSW::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let original: GGSWCompressed<Vec<u8>> = GGSWCompressed::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK);
    test_reader_writer_interface(original);
}
