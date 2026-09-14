//! Every CKKS operation leaves its output canonical at the `k` it reports.
//!
//! Canonical at `k` means, coefficient by coefficient: the low
//! `(-k) mod base2k` bits of the last live limb are zero, every limb past
//! `ceil(k / base2k)` is zero, and no digit exceeds `2^(base2k - 1)` in
//! magnitude. The first two are what the convolution consumers depend on:
//! their `cnv_offset` rescales by `2^k`, so any content an operand carries
//! below its own `k` comes back at full magnitude in the product.
//!
//! The digit bound is the closed range, not the half-open one `normalize`
//! produces: rotation, automorphism and negation are negacyclic and send the
//! single digit `-2^(base2k - 1)` to `+2^(base2k - 1)`. That changes no value
//! and nothing below `k`, so it is allowed here.
//!
//! The destinations here are allocated wider than the `k` they end up
//! labelled with, and at a `k` that is not limb-aligned, which is the shape
//! that exposed `ckks_rotate_into` normalizing at a stale `k`.

use std::collections::HashMap;

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, ckks_encrypt, gen_atk, gen_sk_with_raw, gen_tsk,
    test_vector_1, test_vector_2,
};
use crate::{
    api::{CKKSConjugateOps, CKKSCopyOps, CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops, CKKSRotateOps},
    layouts::CKKSCiphertextOwned,
    test_suite::{CKKSTestParams, reference_encoder::ReferenceEncoder},
};
use poulpy_core::{GLWEAutomorphism, GLWEShift, layouts::LWEInfos};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{GaloisElement, HostBytesBackend, Module, ScratchArena, ZnxView},
};

/// Asserts that the data of `ct` is canonical at `ct.k()`.
///
/// `BE` cannot be inferred from `CKKSCiphertextOwned<BE>`, which is a type
/// alias over the backend's buffer and word types, so callers pass it.
pub fn assert_canonical_at_k<BE: TestContextBackend>(label: &str, ct: &CKKSCiphertextOwned<BE>) {
    let host = ct.to_host_owned::<BE>();
    let base2k: usize = host.base2k().as_usize();
    let k: usize = host.k().as_usize();
    let live: usize = k.div_ceil(base2k);
    let pad: usize = (base2k - k % base2k) % base2k;
    let half: i64 = 1i64 << (base2k - 1);
    let data = host.data();
    for col in 0..data.cols() {
        for limb in 0..live.min(data.size()) {
            for (i, &digit) in data.at(col, limb).iter().enumerate() {
                assert!(
                    (-half..=half).contains(&digit),
                    "{label}: col {col} limb {limb} coeff {i}: digit {digit} exceeds 2^(base2k - 1) (base2k {base2k}, k {k}, live limbs {live}, pad {pad})"
                );
            }
        }
        if live > 0 && pad != 0 {
            let mask: i64 = (1i64 << pad) - 1;
            let dirty: usize = data.at(col, live - 1).iter().filter(|&&d| d & mask != 0).count();
            assert_eq!(
                dirty,
                0,
                "{label}: col {col} limb {}: {dirty} coefficients carry bits below k = {k}",
                live - 1
            );
        }
        for limb in live..data.size() {
            let dirty: usize = data.at(col, limb).iter().filter(|&&d| d != 0).count();
            assert_eq!(
                dirty, 0,
                "{label}: col {col} limb {limb} past k = {k}: {dirty} non-zero coefficients"
            );
        }
    }
}

/// A destination width one or two bits below `k` that is not limb-aligned, so
/// the last live limb has padding bits the operation must leave at zero.
fn narrow_k(params: &CKKSTestParams) -> usize {
    let k = params.k - 1;
    if k.is_multiple_of(params.base2k) { params.k - 2 } else { k }
}

pub fn test_canonical_at_k<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphism<BE> + GLWEShift<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let rot: i64 = 1;
    let gal = module.galois_element(rot);
    let mut atks = HashMap::new();
    atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let src = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let src2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    assert_canonical_at_k::<BE>("encrypt", &src);

    let k_dst = narrow_k(&params);
    assert_ne!(k_dst % params.base2k, 0, "the narrow destination must not be limb-aligned");

    // Out-of-place into a destination allocated at `params.k` but labelled
    // `k_dst`: the write has to target the label, not the allocation.
    let mut dst = alloc_ct(&params, module, k_dst);
    module
        .ckks_rotate_into(&mut dst, &src, rot, &atks, &mut scratch.borrow())
        .unwrap();
    assert_canonical_at_k::<BE>("rotate_into", &dst);
    module
        .ckks_conjugate_into(&mut dst, &src, &conj_key, &mut scratch.borrow())
        .unwrap();
    assert_canonical_at_k::<BE>("conjugate_into", &dst);
    module.ckks_copy(&mut dst, &src, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("copy", &dst);
    module.ckks_neg_into(&mut dst, &src, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("neg_into", &dst);
    module.ckks_mul_i_into(&mut dst, &src, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_i_into", &dst);
    module.ckks_mul_pow2_into(&mut dst, &src, 1, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_pow2_into", &dst);
    module.ckks_div_pow2_into(&mut dst, &src, 1, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("div_pow2_into", &dst);
    module
        .ckks_mul_into(&mut dst, &src, &src2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_canonical_at_k::<BE>("mul_into", &dst);

    // In place on a ciphertext whose `k` has dropped below its allocation,
    // the shape a chain of multiplications produces.
    let mut ct = alloc_ct(&params, module, params.k);
    module.ckks_copy(&mut ct, &src, &mut scratch.borrow()).unwrap();
    module.ckks_mul_assign(&mut ct, &src2, &tsk, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_assign", &ct);
    assert!(ct.k().as_usize() < params.k, "the multiplication is expected to lower k");
    module.ckks_rotate_assign(&mut ct, rot, &atks, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("rotate_assign", &ct);
    module
        .ckks_conjugate_assign(&mut ct, &conj_key, &mut scratch.borrow())
        .unwrap();
    assert_canonical_at_k::<BE>("conjugate_assign", &ct);
    module.ckks_neg_assign(&mut ct).unwrap();
    assert_canonical_at_k::<BE>("neg_assign", &ct);
    module.ckks_mul_i_assign(&mut ct, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_i_assign", &ct);
    module.ckks_mul_pow2_assign(&mut ct, 1, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_pow2_assign", &ct);
    module.ckks_div_pow2_assign(&mut ct, 1).unwrap();
    assert_canonical_at_k::<BE>("div_pow2_assign", &ct);
    module.ckks_mul_assign(&mut ct, &src2, &tsk, &mut scratch.borrow()).unwrap();
    assert_canonical_at_k::<BE>("mul_assign twice", &ct);
}
