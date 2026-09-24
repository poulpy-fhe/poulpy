//! Blind rotation compares coefficient-domain fixtures prepared separately.
use super::{ParityBackend, fixture_ggsw, fixture_glwe, snapshot_glwe, with_scratch};
use crate::{api::*, blind_rotation::*, reference::blind_rotation::*};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
use std::marker::PhantomData;

fn key<B: ParityBackend>(
    module: &Module<B>,
    layout: &BlindRotationKeyLayout,
    distribution: Distribution,
) -> BlindRotationKey<B::OwnedBuf, CGGI, i64> {
    BlindRotationKey {
        keys: (0..layout.n_lwe.as_usize())
            .map(|i| fixture_ggsw(module, layout, 20 + i as u8))
            .collect(),
        dist: distribution,
        _phantom: PhantomData,
    }
}
fn lut<B: ParityBackend>(
    module: &Module<B>,
    layout: &GLWELayout,
    extension: usize,
    direction: LookUpTableRotationDirection,
) -> LookupTable<B::OwnedBuf, i64> {
    let layout = GLWELayout {
        rank: 0usize.into(),
        ..*layout
    };
    LookupTable {
        data: (0..extension).map(|i| fixture_glwe(module, &layout, 50 + i as u8)).collect(),
        rot_dir: direction,
        base2k: layout.base2k,
        k: layout.k,
        drift: 0,
    }
}

fn reject_mismatched_preparation<B: ParityBackend>(
    module: &Module<B>,
    prepared: &mut BlindRotationKeyPrepared<B::OwnedBuf, CGGI, B>,
    layout: &BlindRotationKeyLayout,
) where
    Module<B>: BlindRotationKeyPreparedFactory<CGGI, B>,
{
    let snapshot = |key: &BlindRotationKeyPrepared<B::OwnedBuf, CGGI, B>| {
        (
            key.dist,
            key.data
                .iter()
                .map(|element| (element.ggsw_layout(), B::to_host_bytes(element.data().data())))
                .collect::<Vec<_>>(),
            key.x_pow_a.as_ref().map(|table| {
                table
                    .iter()
                    .map(|element| B::to_host_bytes(element.data()))
                    .collect::<Vec<_>>()
            }),
        )
    };
    let before = snapshot(prepared);
    let invalid_layout = BlindRotationKeyLayout {
        n_lwe: (layout.n_lwe.as_usize() - 1).into(),
        ..*layout
    };
    let invalid_key = key(module, &invalid_layout, Distribution::BinaryBlock(1));
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_scratch::<B, _>(module.blind_rotation_key_prepare_tmp_bytes(layout), |scratch| {
            module.prepare_blind_rotation_key(prepared, &invalid_key, scratch)
        });
    }));
    assert!(failure.is_err(), "mismatched key dimensions must be rejected");
    assert_eq!(before, snapshot(prepared), "failed preparation changed the destination");
}

fn output_fixture<B: ParityBackend>(
    module: &Module<B>,
    allocation: &GLWELayout,
    precision: TorusPrecision,
) -> GLWE<B::OwnedBuf, i64> {
    let mut output = fixture_glwe(module, allocation, 88);
    output.set_k(precision);
    let mut host = output.to_host_owned::<B>();
    super::canonicalize(&mut host);
    host.transfer_into(&mut output);
    output
}

/// Exact coefficient, metadata, and scratch parity for standard, block, and
/// extended CGGI, including both rotation directions, spare output capacity,
/// and key preparation reuse.
pub fn test_blind_rotation_parity<BR, BT>(reference: &Module<BR>, tested: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    Module<BT>: BlindRotationExecute<CGGI, BT> + BlindRotationKeyPreparedFactory<CGGI, BT> + BlindRotationModSwitch<BT>,
    Module<BR>: BlindRotationExecute<CGGI, BR> + BlindRotationKeyPreparedFactory<CGGI, BR> + BlindRotationModSwitch<BR>,
{
    assert_eq!(reference.n(), tested.n());
    let host = Module::<HostBytesBackend>::new(reference.n() as u64);
    for rank in [1usize, 2] {
        let output_layout = GLWELayout {
            n: reference.n().into(),
            base2k: 12usize.into(),
            k: 24usize.into(),
            rank: rank.into(),
        };
        let key_layout = BlindRotationKeyLayout {
            n_glwe: reference.n().into(),
            n_lwe: 4usize.into(),
            base2k: 12usize.into(),
            dnum: 2usize.into(),
            k_aux: 12usize.into(),
            rank: rank.into(),
        };
        let lwe_layout = LWELayout {
            n: 4usize.into(),
            base2k: 4usize.into(),
            k: 16usize.into(),
        };
        let mut lwe_host = host.lwe_alloc_from_infos(&lwe_layout);
        lwe_host.fill_uniform(4, &mut Source::new([77; 32]));
        let mut lwe_r = reference.lwe_alloc_from_infos(&lwe_layout);
        lwe_host.transfer_into(&mut lwe_r);
        let mut lwe_t = tested.lwe_alloc_from_infos(&lwe_layout);
        lwe_host.transfer_into(&mut lwe_t);
        let mut prepared_r = reference.blind_rotation_key_prepared_alloc(&key_layout);
        let mut prepared_t = tested.blind_rotation_key_prepared_alloc(&key_layout);
        // Block first, then standard exercises clearing stale monomial tables.
        for (distribution, extension) in [
            (Distribution::BinaryBlock(2), 1usize),
            (Distribution::BinaryBlock(4), 2),
            (Distribution::BinaryBlock(2), 4),
            (Distribution::BinaryBlock(1), 2),
            (Distribution::BinaryFixed(2), 1),
            (Distribution::BinaryProb(0.5), 1),
            (Distribution::ZERO, 1),
        ] {
            let key_r = key(reference, &key_layout, distribution);
            let key_t = key(tested, &key_layout, distribution);
            with_scratch::<BR, _>(reference.blind_rotation_key_prepare_tmp_bytes(&key_layout), |s| {
                reference.prepare_blind_rotation_key(&mut prepared_r, &key_r, s)
            });
            with_scratch::<BT, _>(tested.blind_rotation_key_prepare_tmp_bytes(&key_layout), |s| {
                tested.prepare_blind_rotation_key(&mut prepared_t, &key_t, s)
            });
            assert_eq!(prepared_r.dist, prepared_t.dist);
            assert_eq!(prepared_r.x_pow_a.is_some(), prepared_t.x_pow_a.is_some());
            for direction in [LookUpTableRotationDirection::Left, LookUpTableRotationDirection::Right] {
                // Distinct source/destination capacities exercise both copy directions.
                // A copy override may need more scratch for either compact or spare storage.
                for (precision, capacity) in [(24usize, 24usize), (24, 60), (60, 84)] {
                    let logical_layout = GLWELayout {
                        k: precision.into(),
                        ..output_layout
                    };
                    let allocation = GLWELayout {
                        k: capacity.into(),
                        ..logical_layout
                    };
                    let lut_r = lut(reference, &logical_layout, extension, direction);
                    let lut_t = lut(tested, &logical_layout, extension, direction);
                    let mut switched_r = vec![0; 5];
                    let mut switched_t = vec![0; 5];
                    reference.blind_rotation_mod_switch(2 * reference.n() * extension, &mut switched_r, &lwe_r, direction);
                    tested.blind_rotation_mod_switch(2 * tested.n() * extension, &mut switched_t, &lwe_t, direction);
                    assert_eq!(switched_r, switched_t, "modulus-switch parity");
                    let mut output_r = output_fixture(reference, &allocation, logical_layout.k);
                    let mut output_t = output_fixture(tested, &allocation, logical_layout.k);
                    with_scratch::<BR, _>(
                        reference.blind_rotation_execute_tmp_bytes(key_r.block_size(), extension, &output_r, &key_layout),
                        |s| reference.blind_rotation_execute(&mut output_r, &lwe_r, &lut_r, &prepared_r, s),
                    );
                    with_scratch::<BT, _>(
                        tested.blind_rotation_execute_tmp_bytes(key_t.block_size(), extension, &output_t, &key_layout),
                        |s| tested.blind_rotation_execute(&mut output_t, &lwe_t, &lut_t, &prepared_t, s),
                    );
                    assert_eq!(
                        snapshot_glwe::<BR, _>(&output_r),
                        snapshot_glwe::<BT, _>(&output_t),
                        "blind-rotation parity"
                    );
                }
            }
        }
        reject_mismatched_preparation(reference, &mut prepared_r, &key_layout);
        reject_mismatched_preparation(tested, &mut prepared_t, &key_layout);
    }
}

/// Checks key encryption and decompression against their independently callable
/// compositions on the same backend, so both sides consume identical sampler
/// outputs without requiring a seed-to-stream convention across backends.
pub fn test_blind_rotation_key_lifecycle<B>(module: &Module<B>)
where
    B: ParityBackend,
    Module<B>: BlindRotationKeyEncryptSk<CGGI, B>
        + BlindRotationKeyCompressedEncryptSk<B, CGGI>
        + BlindRotationKeyCompressedFactory<CGGI, B>
        + BlindRotationKeyDecompress<CGGI, B>
        + GGSWEncryptSk<B>
        + GGSWCompressedEncryptSk<B>
        + GLWESecretPreparedFactory<B>
        + GGSWDecompress
        + GLWEDecompress<Backend = B>,
{
    let host = Module::<HostBytesBackend>::new(module.n() as u64);
    let layout = BlindRotationKeyLayout {
        n_glwe: module.n().into(),
        n_lwe: 4usize.into(),
        base2k: 12usize.into(),
        dnum: 2usize.into(),
        k_aux: 12usize.into(),
        rank: 1usize.into(),
    };
    let enc = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let mut secret_host = host.glwe_secret_alloc_from_infos(&layout);
    for (i, word) in secret_host.data_mut().at_mut(0, 0).iter_mut().enumerate() {
        *word = (i % 3) as i64 - 1;
    }
    *secret_host.dist_mut() = Distribution::TernaryProb(0.5);
    let mut secret = module.glwe_secret_alloc_from_infos(&layout);
    secret_host.transfer_into(&mut secret);
    let mut secret_prepared = module.glwe_secret_prepared_alloc_from_infos(&layout);
    module.glwe_secret_prepare(&mut secret_prepared, &secret);
    for distribution in [
        Distribution::BinaryBlock(2),
        Distribution::BinaryFixed(2),
        Distribution::BinaryProb(0.5),
        Distribution::ZERO,
    ] {
        let mut lwe_secret_host = host.lwe_secret_alloc(4usize.into());
        lwe_secret_host
            .data_mut()
            .at_mut(0, 0)
            .copy_from_slice(if matches!(distribution, Distribution::ZERO) {
                &[0, 0, 0, 0]
            } else {
                &[0, 1, 1, 0]
            });
        *lwe_secret_host.dist_mut() = distribution;
        let mut lwe_secret = module.lwe_secret_alloc(4usize.into());
        lwe_secret_host.transfer_into(&mut lwe_secret);
        let mut expected = BlindRotationKey::<B::OwnedBuf, CGGI, i64>::alloc(module, &layout);
        let mut actual = BlindRotationKey::<B::OwnedBuf, CGGI, i64>::alloc(module, &layout);
        with_scratch::<B, _>(blind_rotation_key_encrypt_sk_tmp_bytes_ref(module, &layout), |s| {
            blind_rotation_key_encrypt_sk_ref(
                module,
                &mut expected,
                &secret_prepared,
                &lwe_secret,
                &enc,
                &mut Source::new([21; 32]),
                &mut Source::new([22; 32]),
                s,
            );
        });
        with_scratch::<B, _>(module.blind_rotation_key_encrypt_sk_tmp_bytes(&layout), |s| {
            module.blind_rotation_key_encrypt_sk(
                &mut actual,
                &secret_prepared,
                &lwe_secret,
                &enc,
                &mut Source::new([21; 32]),
                &mut Source::new([22; 32]),
                s,
            );
        });
        assert_eq!(actual.dist, expected.dist);
        for (a, b) in actual.keys.iter().zip(&expected.keys) {
            assert_eq!(
                super::snapshot_ggsw::<B, _>(a),
                super::snapshot_ggsw::<B, _>(b),
                "key encryption parity"
            );
        }
        let mut compressed_expected = blind_rotation_key_compressed_alloc_ref(module, &layout);
        let mut compressed_actual = BlindRotationKeyCompressed::<poulpy_hal::AlignedBuf, CGGI, i64>::alloc(module, &layout);
        with_scratch::<B, _>(blind_rotation_key_compressed_encrypt_sk_tmp_bytes_ref(module, &layout), |s| {
            blind_rotation_key_compressed_encrypt_sk_ref(
                module,
                &mut compressed_expected,
                &secret_prepared,
                &lwe_secret,
                [33; 32],
                &enc,
                &mut Source::new([34; 32]),
                s,
            );
        });
        with_scratch::<B, _>(module.blind_rotation_key_compressed_encrypt_sk_tmp_bytes(&layout), |s| {
            module.blind_rotation_key_compressed_encrypt_sk(
                &mut compressed_actual,
                &secret_prepared,
                &lwe_secret,
                [33; 32],
                &enc,
                &mut Source::new([34; 32]),
                s,
            );
        });
        assert_eq!(compressed_actual.dist, compressed_expected.dist);
        // Decompression checks every body limb and recreated mask rather than
        // requiring the prepared representation to be shared across backends.
        with_scratch::<B, _>(blind_rotation_key_decompress_tmp_bytes_ref(module, &layout), |s| {
            blind_rotation_key_decompress_ref(module, &mut expected, &compressed_expected, s)
        });
        with_scratch::<B, _>(module.blind_rotation_key_decompress_tmp_bytes(&layout), |s| {
            module.blind_rotation_key_decompress(&mut actual, &compressed_actual, s)
        });
        assert_eq!(actual.dist, expected.dist);
        for (a, b) in actual.keys.iter().zip(&expected.keys) {
            assert_eq!(
                super::snapshot_ggsw::<B, _>(a),
                super::snapshot_ggsw::<B, _>(b),
                "compressed key/decompression parity"
            );
        }
    }
}
