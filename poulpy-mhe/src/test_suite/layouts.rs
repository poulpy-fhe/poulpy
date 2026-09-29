//! PAT layout tests: allocation, serialization, and interop with core
//! compressed encryption and decompression.

use poulpy_core::{
    EncryptionLayout, GGLWECompressedEncryptSk, GGLWEEncryptSk, GLWECompressedEncryptSk,
    layouts::{
        GGLWE, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWESecretPreparedFactory,
        GLWESecretSampling, ModuleCoreAlloc, ModuleCoreCompressedAlloc,
        compressed::{GGLWECompressed, GGLWEDecompress, GLWECompressed, GLWEDecompress},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ReaderFrom, ScratchOwned, WriterTo},
    source::Source,
    test_suite::serialization::test_reader_writer_interface,
};

use super::fixtures::{BASE2K, DNUM, DSIZE, K, RANK, SEED_XE, SEEDS, assert_write_rejects, gglwe_layout, secret};
use crate::layouts::{GGLWEPat, GGLWEPatCompressed, GLWEPatCompressed, MHEModuleAlloc};

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

/// Allocation, serialization and core interop of [`GGLWEPatCompressed`].
pub fn test_gglwe_pat_compressed<BE>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWECompressedEncryptSk<BE>
        + GLWEDecompress<Backend = BE>
        + GGLWEDecompress,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let alloc: GGLWEPatCompressed<AlignedBuf, i64> =
        module.gglwe_pat_compressed_alloc(BASE2K, DNUM, DSIZE, layout.k_aux, RANK, RANK);
    assert_eq!(alloc.gglwe_layout(), layout);
    assert!(alloc.is_canonical());

    let (sk, sk_prepared) = secret(module);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_compressed_encrypt_sk_tmp_bytes(&layout));

    let mut pats: [GGLWEPatCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| module.gglwe_pat_compressed_alloc_from_infos(&layout));
    for (pat, seed) in pats.iter_mut().zip(SEEDS) {
        module.gglwe_compressed_encrypt_sk(
            pat,
            sk.data(),
            &sk_prepared,
            seed,
            &enc_infos,
            &mut Source::new(SEED_XE),
            &mut scratch.borrow(),
        );
    }

    let mut core: GGLWECompressed<AlignedBuf, i64> = module.gglwe_compressed_alloc_from_infos(&layout);
    module.gglwe_compressed_encrypt_sk(
        &mut core,
        sk.data(),
        &sk_prepared,
        SEEDS[0],
        &enc_infos,
        &mut Source::new(SEED_XE),
        &mut scratch.borrow(),
    );
    assert_eq!(pats[0].inner, core);

    let mut have: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    let mut want: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.decompress_gglwe(&mut have, &pats[0]);
    module.decompress_gglwe(&mut want, &core);
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

/// Allocation, serialization and core interop of [`GGLWEPat`].
pub fn test_gglwe_pat<BE>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE> + GGLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let alloc: GGLWEPat<AlignedBuf, i64> = module.gglwe_pat_alloc(BASE2K, DNUM, DSIZE, layout.k_aux, RANK, RANK);
    assert_eq!(alloc.gglwe_layout(), layout);
    assert!(alloc.is_canonical());

    let (sk, sk_prepared) = secret(module);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_encrypt_sk_tmp_bytes(&layout));

    let mut pats: [GGLWEPat<AlignedBuf, i64>; 2] = [(); 2].map(|_| module.gglwe_pat_alloc_from_infos(&layout));
    for (pat, seed) in pats.iter_mut().zip(SEEDS) {
        module.gglwe_encrypt_sk(
            pat,
            sk.data(),
            &sk_prepared,
            &enc_infos,
            &mut Source::new(SEED_XE),
            &mut Source::new(seed),
            &mut scratch.borrow(),
        );
    }

    let mut core: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_encrypt_sk(
        &mut core,
        sk.data(),
        &sk_prepared,
        &enc_infos,
        &mut Source::new(SEED_XE),
        &mut Source::new(SEEDS[0]),
        &mut scratch.borrow(),
    );
    assert_eq!(pats[0].inner, core);

    let mut flagged = pats[0].clone();
    flagged.set_canonical(false);
    assert_write_rejects(&flagged);
    let mut bytes: Vec<u8> = Vec::new();
    pats[0].write_to(&mut bytes).unwrap();
    flagged.read_from(&mut bytes.as_slice()).unwrap();
    assert!(flagged.is_canonical());

    test_reader_writer_interface(pats);
}
