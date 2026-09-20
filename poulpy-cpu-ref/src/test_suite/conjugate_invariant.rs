/// Multiplies through an independent degree-2n negacyclic embedding.
pub fn ambient_product(a: &[i64], b: &[i64]) -> Vec<i64> {
    let n = a.len();
    let unfold = |a: &[i64]| {
        let mut r = vec![0; 2 * n];
        r[..n].copy_from_slice(a);
        for j in 1..n {
            r[2 * n - j] = -a[j];
        }
        r
    };
    let (a, b) = (unfold(a), unfold(b));
    let mut r = vec![0; 2 * n];
    for (i, a) in a.iter().enumerate().filter(|(_, a)| **a != 0) {
        for (j, b) in b.iter().enumerate().filter(|(_, b)| **b != 0) {
            r[(i + j) % (2 * n)] += if i + j < 2 * n { a * b } else { -a * b };
        }
    }
    assert_eq!(r[n], 0);
    for j in 1..n {
        assert_eq!(r[2 * n - j], -r[j]);
    }
    r.truncate(n);
    r
}

/// Substitutes in the ambient ring and restricts back to invariant coefficients.
pub fn ambient_automorphism(a: &[i64], p: i64) -> Vec<i64> {
    let n = a.len();
    let mut r = vec![0; 2 * n];
    let p = p.rem_euclid(4 * n as i64) as usize;
    for j in 0..2 * n {
        let a = if j < n {
            a[j]
        } else if j == n {
            0
        } else {
            -a[2 * n - j]
        };
        let e = (j * p) % (4 * n);
        r[e % (2 * n)] += if e < 2 * n { a } else { -a };
    }
    r.truncate(n);
    r
}

/// Shared invariant ring correctness and cross-backend product tests.
#[macro_export]
macro_rules! conjugate_invariant_test_suite {
    ($name:ident, $backend:ty, $config:expr) => {
        mod $name {
            use poulpy_hal::{api::*, layouts::*};
            type BE = $backend;
            fn module(n: u64) -> Module<BE> {
                ($config).new_module::<BE>(n)
            }

            #[test]
            fn conjugate_invariant_ring_metadata() {
                fn check<B: Backend>(module: &Module<B>, order: i64) {
                    assert_eq!(module.cyclotomic_order(), order);
                    assert_eq!(module.galois_element(2), 25 % order);
                    let p = order - 3;
                    assert_eq!(p * module.galois_element_inv(p) % order, 1);
                }

                for n in [BE::MIN_DEGREE as u64, 32768] {
                    check(&Module::<BE>::new(n), 2 * n as i64);
                    check(&module(n), 4 * n as i64);
                }
            }

            #[test]
            fn conjugate_invariant_ring() {
                let module = module(32768);
                for n in [8, 32, 256, 8192, 32768].into_iter().filter(|&n| n >= BE::MIN_DEGREE) {
                    let mut a = module.vec_znx_alloc(n, 1, 2);
                    let mut b = module.scalar_znx_alloc(n, 1);
                    for j in 0..n.min(256) {
                        a.at_mut(0, 0)[j] = (j * 7 % 17) as i64 - 8;
                        a.at_mut(0, 1)[j] = (j * 13 % 23) as i64 - 11;
                        b.at_mut(0, 0)[j] = (j * 11 % 19) as i64 - 9;
                    }
                    let mut dft = module.vec_znx_dft_alloc(n, 1, 3);
                    let mut big = module.vec_znx_big_alloc(n, 1, 3);
                    let mut auto = module.vec_znx_alloc(n, 1, 3);
                    let mut auto_dft = module.vec_znx_dft_alloc(n, 1, 3);
                    let mut scratch = ScratchOwned::<BE>::alloc(
                        module
                            .vec_znx_automorphism_assign_tmp_bytes()
                            .max(module.vec_znx_big_automorphism_assign_tmp_bytes())
                            .max(module.svp_apply_dft_tmp_bytes(2))
                            .max(module.vec_znx_dft_automorphism_add_with_plan_tmp_bytes(3, 3)),
                    );
                    for p in [1, -1, 5, 17, 25, 29, -29, i64::MAX, i64::MIN + 1] {
                        module.vec_znx_automorphism(
                            p,
                            &mut vec_znx_backend_mut::<BE>(&mut auto),
                            0,
                            &vec_znx_backend_ref::<BE>(&a),
                            0,
                        );
                        for limb in 0..2 {
                            assert_eq!(
                                auto.at(0, limb),
                                $crate::test_suite::conjugate_invariant::ambient_automorphism(a.at(0, limb), p)
                            );
                        }
                        module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a), 0);
                        module.vec_znx_dft_automorphism(p, &mut auto_dft.to_backend_mut(), 0, &dft.to_backend_ref(), 0);
                        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut auto_dft.to_backend_mut(), 0);
                        for limb in 0..3 {
                            for j in 0..n {
                                assert_eq!(
                                    big.at(0, limb)[j] as i128,
                                    auto.at(0, limb)[j] as i128,
                                    "DFT automorphism n={n}, p={p}"
                                );
                            }
                        }
                        module.vec_znx_automorphism_assign(p, &mut vec_znx_backend_mut::<BE>(&mut auto), 0, &mut scratch.arena());
                        module.vec_znx_big_automorphism_assign(p, &mut big.to_backend_mut(), 0, &mut scratch.arena());
                        for limb in 0..3 {
                            for j in 0..n {
                                assert_eq!(big.at(0, limb)[j] as i128, auto.at(0, limb)[j] as i128);
                            }
                        }
                        module.vec_znx_dft_apply(
                            1,
                            0,
                            &mut auto_dft.to_backend_mut(),
                            0,
                            &vec_znx_backend_ref::<BE>(&a),
                            0,
                        );
                        let plan = module.vec_znx_dft_automorphism_plan(n, p);
                        module.vec_znx_dft_automorphism_add_with_plan(
                            &plan,
                            &mut auto_dft.to_backend_mut(),
                            0,
                            &dft.to_backend_ref(),
                            0,
                            &mut scratch.arena(),
                        );
                        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut auto_dft.to_backend_mut(), 0);
                        for limb in 0..2 {
                            let want = $crate::test_suite::conjugate_invariant::ambient_automorphism(a.at(0, limb), p);
                            for j in 0..n {
                                assert_eq!(
                                    big.at(0, limb)[j] as i128,
                                    (a.at(0, limb)[j] + want[j]) as i128,
                                    "accumulating automorphism n={n}, p={p}"
                                );
                            }
                        }
                    }
                    let mut ppol = module.svp_ppol_alloc(n, 1, PrepareHint::Reuse);
                    module.svp_prepare(
                        &mut ppol.to_backend_mut(),
                        0,
                        &poulpy_hal::test_suite::scalar_znx_backend_ref::<BE>(&b),
                        0,
                    );
                    module.svp_apply_dft(
                        &mut dft.to_backend_mut(),
                        0,
                        &ppol.to_backend_ref(),
                        0,
                        &vec_znx_backend_ref::<BE>(&a),
                        0,
                        &mut scratch.arena(),
                    );
                    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), 0);
                    for limb in 0..2 {
                        let want = $crate::test_suite::conjugate_invariant::ambient_product(a.at(0, limb), b.at(0, 0));
                        for j in 0..n {
                            assert_eq!(big.at(0, limb)[j] as i128, want[j] as i128, "product n={n}");
                        }
                    }
                    assert!(big.at(0, 2).iter().all(|&v| v == 0));
                }
            }

            #[test]
            fn conjugate_invariant_sparse_convolution() {
                let module = module(256);
                for n in [8, 128, 256].into_iter().filter(|&n| n >= BE::MIN_DEGREE) {
                    let params = poulpy_hal::test_suite::TestParams {
                        size: 256,
                        n,
                        base2k: 10,
                    };
                    poulpy_hal::test_suite::convolution::test_convolution_sparse(&params, &module);
                    poulpy_hal::test_suite::sparse::test_vec_znx_sparse_add_sub(&params, &module);
                    poulpy_hal::test_suite::sparse::test_vec_znx_big_sparse_add_sub(&params, &module);
                }
            }

            #[test]
            fn conjugate_invariant_products_parity() {
                let module = module(256);
                let reference = $crate::FFT64ModuleConfig::conjugate_invariant().new_module::<$crate::FFT64Ref>(256);
                let host = Module::<HostBytesBackend>::new(256);
                for n in [8, 256].into_iter().filter(|&n| n >= BE::MIN_DEGREE) {
                    let params = poulpy_hal::test_suite::TestParams {
                        size: 256,
                        n,
                        base2k: 10,
                    };
                    poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft(&params, &host, &reference, &module);
                    poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign(&params, &host, &reference, &module);
                    poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft(&params, &host, &reference, &module);
                    poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add(&params, &host, &reference, &module);
                }
            }

            #[test]
            fn conjugate_invariant_limb_convolution() {
                let module = module(64);
                let n = 32;
                let mut a = module.vec_znx_alloc(n, 2, 2);
                for col in 0..2 {
                    for limb in 0..2 {
                        for j in 0..n {
                            a.at_mut(col, limb)[j] = ((j * 7 + col * 5 + limb * 3) % 17) as i64 - 8;
                        }
                    }
                }
                let mut left = module.cnv_pvec_left_alloc(n, 2, 2, PrepareHint::Reuse);
                let mut right = module.cnv_pvec_right_alloc(n, 2, 2, PrepareHint::Reuse);
                let mut scratch = ScratchOwned::<BE>::alloc(
                    module
                        .cnv_prepare_self_tmp_bytes(2, 2)
                        .max(module.cnv_pairwise_apply_dft_tmp_bytes(0, 4, 2, 2)),
                );
                module.cnv_prepare_self(
                    &mut left.to_backend_mut(),
                    &mut right.to_backend_mut(),
                    &vec_znx_backend_ref::<BE>(&a),
                    &mut scratch.arena(),
                );
                let mut dft = module.vec_znx_dft_alloc(n, 1, 4);
                let mut big = module.vec_znx_big_alloc(n, 1, 4);
                for second in [0, 1] {
                    let input = (0..2)
                        .map(|limb| {
                            (0..n)
                                .map(|j| a.at(0, limb)[j] + if second == 0 { 0 } else { a.at(1, limb)[j] })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut want = vec![vec![0; n]; 3];
                    for i in 0..2 {
                        for j in 0..2 {
                            let product = $crate::test_suite::conjugate_invariant::ambient_product(&input[i], &input[j]);
                            for k in 0..n {
                                want[i + j][k] += product[k];
                            }
                        }
                    }
                    for offset in 0..3 {
                        module.cnv_pairwise_apply_dft(
                            offset,
                            &mut dft.to_backend_mut(),
                            0,
                            &left.to_backend_ref(),
                            &right.to_backend_ref(),
                            0,
                            second,
                            &mut scratch.arena(),
                        );
                        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), 0);
                        for limb in 0..4 {
                            for j in 0..n {
                                assert_eq!(
                                    big.at(0, limb)[j] as i128,
                                    if limb + offset < 3 {
                                        want[limb + offset][j] as i128
                                    } else {
                                        0
                                    }
                                );
                            }
                        }
                    }
                }
                poulpy_hal::test_suite::convolution::test_convolution_by_const::<_, BE>(&module, 32, 12);
                poulpy_hal::test_suite::convolution::test_convolution_by_const_add::<_, BE>(&module, 32, 12);
            }

            #[test]
            fn conjugate_invariant_large_radix_products() {
                let module = module(8192);
                let reference = $crate::NTTModuleConfig::conjugate_invariant().new_module::<$crate::NTT4x30Ref>(8192);
                let host = Module::<HostBytesBackend>::new(8192);
                let params = poulpy_hal::test_suite::TestParams {
                    size: 8192,
                    n: 8192,
                    base2k: 19,
                };
                poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft(&params, &host, &reference, &module);
                poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add(&params, &host, &reference, &module);
            }

            #[test]
            #[should_panic(expected = "monomial multiplication is not defined")]
            fn conjugate_invariant_rejects_monomial() {
                let n = BE::MIN_DEGREE;
                let module = module(n as u64);
                let mut a = module.vec_znx_alloc(n, 1, 1);
                let b = module.vec_znx_alloc(n, 1, 1);
                module.vec_znx_rotate(
                    1,
                    &mut vec_znx_backend_mut::<BE>(&mut a),
                    0,
                    &vec_znx_backend_ref::<BE>(&b),
                    0,
                );
            }
        }
    };
}

/// Core operation parity under an invariant module configuration.
#[macro_export]
macro_rules! conjugate_invariant_core_test_suite {
    ($name:ident, $backend:ty, $config:expr) => {
        mod $name {
            #[test]
            fn conjugate_invariant_core_parity() {
                let reference = $crate::FFT64ModuleConfig::conjugate_invariant().new_module::<$crate::FFT64Ref>(256);
                let module = ($config).new_module::<$backend>(256);
                let params = poulpy_hal::test_suite::TestParams {
                    size: 256,
                    n: 256,
                    base2k: 10,
                };
                let shapes = poulpy_core::test_suite::parity::ParityShapes {
                    ranks: vec![1, 2],
                    dsizes: Some(vec![1, 2]),
                };
                poulpy_core::test_suite::parity::test_glwe_keyswitch_parity(&params, &shapes, &reference, &module);
                poulpy_core::test_suite::parity::test_glwe_external_product_parity(&params, &shapes, &reference, &module);
                poulpy_core::test_suite::parity::test_glwe_automorphism_parity(&params, &shapes, &reference, &module);
                poulpy_core::test_suite::parity::test_glwe_automorphism_coarsened(&params, &module);
            }
        }
    };
}

#[cfg(all(test, feature = "enable-core"))]
mod tests {
    #[test]
    fn conjugate_invariant_encryption() {
        let params = poulpy_hal::test_suite::TestParams {
            size: 256,
            n: 256,
            base2k: 10,
        };
        let module = crate::NTTModuleConfig::conjugate_invariant().new_module::<crate::NTT4x30Ref>(256);
        poulpy_core::test_suite::noise::encryption::test_glwe_encrypt_sk(&params, &module);
        poulpy_core::test_suite::noise::automorphism::test_glwe_automorphism(&params, &module);
        poulpy_core::test_suite::noise::encryption::test_gglwe_automorphism_key_encrypt_sk(&params, &module);
        poulpy_core::test_suite::noise::linear_transformation::test_glwe_hoisted_baby_rotations_match_automorphism(
            &params, &module,
        );
    }
}

#[cfg(all(test, feature = "enable-core"))]
#[test]
fn conjugate_invariant_key_composition() {
    use poulpy_core::{EncryptionLayout, GGLWENoise, GLWEAutomorphismKeyAutomorphism, GLWEAutomorphismKeyEncryptSk, layouts::*};
    use poulpy_hal::{api::*, layouts::*, source::Source};
    type BE = crate::NTT4x30Ref;
    let module = crate::NTTModuleConfig::conjugate_invariant().new_module::<BE>(8);
    let infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: 8u32.into(),
        base2k: 10u32.into(),
        dnum: 4u32.into(),
        k_aux: 30u32.into(),
        dsize: 1u32.into(),
        rank: 1u32.into(),
    })
    .unwrap();
    let mut source_s = Source::new([0; 32]);
    let mut source_e = Source::new([1; 32]);
    let mut source_a = Source::new([2; 32]);
    let mut sk = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_s);
    let mut input = module.glwe_automorphism_key_alloc_from_infos(&infos);
    let mut applied = module.glwe_automorphism_key_alloc_from_infos(&infos);
    let mut output = module.glwe_automorphism_key_alloc_from_infos(&infos);
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_automorphism_key_encrypt_sk_tmp_bytes(&infos)
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&infos))
            .max(module.gglwe_noise_tmp_bytes(&infos))
            .max(module.glwe_automorphism_key_automorphism_tmp_bytes(&infos, &infos, &infos)),
    );
    module.glwe_automorphism_key_encrypt_sk(
        &mut input,
        25,
        &sk,
        &infos,
        &mut source_e,
        &mut source_a,
        &mut scratch.arena(),
    );
    module.glwe_automorphism_key_encrypt_sk(
        &mut applied,
        25,
        &sk,
        &infos,
        &mut source_e,
        &mut source_a,
        &mut scratch.arena(),
    );
    let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&infos);
    module.glwe_automorphism_key_prepare(&mut prepared, &applied, &mut scratch.arena());
    module.glwe_automorphism_key_automorphism(
        &mut output,
        &input,
        &GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(&prepared),
        &mut scratch.arena(),
    );
    assert_eq!(output.p(), 17);
    let expected = ambient_automorphism(sk.data().at(0, 0), 17);
    let mut transformed_sk = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_zero(&mut transformed_sk);
    transformed_sk.data_mut().at_mut(0, 0).copy_from_slice(&expected);
    let mut transformed_prepared = module.glwe_secret_prepared_alloc_from_infos(&infos);
    module.glwe_secret_prepare(&mut transformed_prepared, &transformed_sk);
    for row in 0..4 {
        let error = module.gglwe_noise(
            &output,
            row,
            0,
            &sk.data().to_ref(),
            &transformed_prepared,
            &mut scratch.arena(),
        );
        assert!(error.std() < 1e-6, "composed secret relation: {}", error.std());
    }
}
