//! PAT layout tests: allocation, serialization, and interop with core
//! compressed encryption and decompression.

use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk,
    layouts::{
        Base2K, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWESecret, GLWESecretPrepared,
        GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, TorusPrecision,
        compressed::{GLWECompressed, GLWEDecompress},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ReaderFrom, ScratchOwned, WriterTo},
    source::Source,
    test_suite::serialization::test_reader_writer_interface,
};

use crate::layouts::{GLWEPatCompressed, MHEModuleAlloc};

const BASE2K: Base2K = Base2K(12);
const K: TorusPrecision = TorusPrecision(33);
const RANK: Rank = Rank(2);
const SEEDS: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
const SEED_XE: [u8; 32] = [3u8; 32];

/// Allocation, serialization and core interop of [`GLWEPatCompressed`].
pub fn test_glwe_pat_compressed<BE>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWECompressedEncryptSk<BE>
        + GLWEDecompress<Backend = BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let alloc: GLWEPatCompressed<AlignedBuf, i64> = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    assert_eq!(alloc.glwe_layout(), layout);
    assert!(alloc.is_canonical());

    let (_, sk) = secret(module);
    let pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
        n: layout.n,
        base2k: BASE2K,
        k: K,
    });
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_compressed_encrypt_sk_tmp_bytes(&layout));

    let mut pats: [GLWEPatCompressed<AlignedBuf, i64>; 2] = [(); 2].map(|_| module.glwe_pat_compressed_alloc_from_infos(&layout));
    for (pat, seed) in pats.iter_mut().zip(SEEDS) {
        module.glwe_compressed_encrypt_sk(
            pat,
            &pt,
            &sk,
            seed,
            &enc_infos,
            &mut Source::new(SEED_XE),
            &mut scratch.borrow(),
        );
    }

    let mut core: GLWECompressed<AlignedBuf, i64> = module.glwe_compressed_alloc_from_infos(&layout);
    module.glwe_compressed_encrypt_sk(
        &mut core,
        &pt,
        &sk,
        SEEDS[0],
        &enc_infos,
        &mut Source::new(SEED_XE),
        &mut scratch.borrow(),
    );
    assert_eq!(pats[0].inner, core);

    let mut have: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut want: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.decompress_glwe(&mut have, &pats[0]);
    module.decompress_glwe(&mut want, &core);
    assert_eq!(have, want);

    let mut flagged = pats[0].clone();
    flagged.set_canonical(false);
    assert_write_rejects(&flagged);
    let mut bytes: Vec<u8> = Vec::new();
    pats[0].write_to(&mut bytes).unwrap();
    flagged.read_from(&mut bytes.as_slice()).unwrap();
    assert!(flagged.is_canonical());

    test_reader_writer_interface(pats);
}

fn secret<BE>(module: &Module<BE>) -> (GLWESecret<AlignedBuf, i64>, GLWESecretPrepared<AlignedBuf, BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    let mut sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([0u8; 32]));
    let mut sk_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);
    (sk, sk_prepared)
}

fn assert_write_rejects<T: WriterTo>(value: &T) {
    let mut bytes: Vec<u8> = Vec::new();
    let err = value.write_to(&mut bytes).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(bytes.is_empty());
}
