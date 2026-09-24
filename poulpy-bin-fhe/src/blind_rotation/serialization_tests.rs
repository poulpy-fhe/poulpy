use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{HostBytesBackend as HB, Module};
use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

use crate::blind_rotation::{BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyLayout, CGGI};

crate::impl_bin_fhe_blind_rotation_key_compressed_factory_reference!(HB);

type HostModule = Module<HB>;

#[test]
fn test_cggi_blind_rotation_key_serialization() {
    let layout: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: 256_u32.into(),
        n_lwe: 64_usize.into(),
        base2k: 12_usize.into(),
        dnum: 2_usize.into(),
        k_aux: 30_usize.into(),
        rank: 2_usize.into(),
    };
    let module = HostModule::new(layout.n_glwe.as_u32() as u64);
    let original: BlindRotationKey<AlignedBuf, CGGI, i64> = BlindRotationKey::alloc(&module, &layout);
    test_reader_writer_interface(original);
}

#[test]
fn test_cggi_blind_rotation_key_compressed_serialization() {
    let layout: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: 256_u32.into(),
        n_lwe: 64_usize.into(),
        base2k: 12_usize.into(),
        dnum: 2_usize.into(),
        k_aux: 30_usize.into(),
        rank: 2_usize.into(),
    };
    let module = HostModule::new(layout.n_glwe.as_u32() as u64);
    let original: BlindRotationKeyCompressed<AlignedBuf, CGGI, i64> = BlindRotationKeyCompressed::alloc(&module, &layout);
    test_reader_writer_interface(original);
}
