use poulpy_core::{
    GGSWEncryptSk, GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct,
    GLWEKeyswitch, GLWEMulPlain, GLWENormalize, GLWESub, GLWESwitchingKeyEncryptSk, GLWETensoring,
    layouts::{
        GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchOwned},
};

use std::marker::PhantomData;

use criterion::{Criterion, measurement::WallTime};

use crate::{
    BenchOp, bench_ops, bin_fhe_n,
    core::{
        automorphism, decryption, encryption, external_product, glwe_tensor, keyswitch, operations,
        params::{CoreParams, default_bench_params_core},
    },
};

// Op tables for each core capability group. Each function returns the raw
// [`BenchOp`] table for that group only, scoped to the traits its own ops
// need — a backend implementing just a subset of `poulpy-core` can still
// build and run the tables for what it supports. Compose across groups and
// drive [`bench_ops`](crate::bench_ops) directly, or use [`all_ops`]
// to run every group at once.

// ── encryption ───────────────────────────────────────────────────────────────

pub fn encryption_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 3]
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp {
            layer: "core",
            name: "glwe_encrypt_sk",
            runner: encryption::runner_glwe_encrypt_sk::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "ggsw_encrypt_sk",
            runner: encryption::runner_ggsw_encrypt_sk::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_automorphism_key_encrypt_sk",
            runner: encryption::runner_glwe_automorphism_key_encrypt_sk::<BE, _>,
        },
    ]
}

// ── decryption ───────────────────────────────────────────────────────────────

pub fn decryption_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE> + GLWEDecrypt<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp {
        layer: "core",
        name: "glwe_decrypt",
        runner: decryption::runner_glwe_decrypt::<BE, _>,
    }]
}

// ── automorphism ─────────────────────────────────────────────────────────────

pub fn automorphism_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp {
        layer: "core",
        name: "glwe_automorphism",
        runner: automorphism::runner_glwe_automorphism::<BE, _>,
    }]
}

// ── external_product ─────────────────────────────────────────────────────────

pub fn external_product_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 2]
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp {
            layer: "core",
            name: "glwe_external_product",
            runner: external_product::runner_glwe_external_product::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_external_product_assign",
            runner: external_product::runner_glwe_external_product_assign::<BE, _>,
        },
    ]
}

// ── keyswitch ────────────────────────────────────────────────────────────────

pub fn keyswitch_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp {
        layer: "core",
        name: "glwe_keyswitch",
        runner: keyswitch::runner_glwe_keyswitch::<BE, _>,
    }]
}

// ── glwe_tensor ──────────────────────────────────────────────────────────────

pub fn glwe_tensor_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 3]
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    [
        BenchOp {
            layer: "core",
            name: "glwe_tensor_relinearize",
            runner: glwe_tensor::runner_glwe_tensor_relinearize::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_tensor_apply",
            runner: glwe_tensor::runner_glwe_tensor_apply::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_tensor_square_apply",
            runner: glwe_tensor::runner_glwe_tensor_square_apply::<BE, _>,
        },
    ]
}

// ── operations ───────────────────────────────────────────────────────────────

pub fn operations_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 8]
where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE> + GLWESub<BE> + GLWENormalize<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp {
            layer: "core",
            name: "glwe_add_into",
            runner: operations::runner_glwe_add_into::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_add_assign",
            runner: operations::runner_glwe_add_assign::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_sub",
            runner: operations::runner_glwe_sub::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_sub_assign",
            runner: operations::runner_glwe_sub_assign::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_normalize",
            runner: operations::runner_glwe_normalize::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_normalize_assign",
            runner: operations::runner_glwe_normalize_assign::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_mul_plain",
            runner: operations::runner_glwe_mul_plain::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_mul_plain_assign",
            runner: operations::runner_glwe_mul_plain_assign::<BE, _>,
        },
    ]
}

// ── all ──────────────────────────────────────────────────────────────────────

/// Concatenates every core-layer group into a single table. Requires a
/// backend that implements the full `poulpy-core` surface; a backend
/// supporting only part of it should instead compose the `*_ops` tables it
/// needs directly.
pub fn all_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend, M: criterion::measurement::Measurement>()
-> Vec<BenchOp<M, CoreParams>>
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWENormalize<BE>
        + GLWEMulPlain<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let mut ops: Vec<BenchOp<M, CoreParams>> = Vec::new();
    ops.extend(encryption_ops::<BE, _>());
    ops.extend(decryption_ops::<BE, _>());
    ops.extend(automorphism_ops::<BE, _>());
    ops.extend(external_product_ops::<BE, _>());
    ops.extend(keyswitch_ops::<BE, _>());
    ops.extend(glwe_tensor_ops::<BE, _>());
    ops.extend(operations_ops::<BE, _>());
    ops
}

// ── standard ─────────────────────────────────────────────────────────────────

/// A small, representative cross-section of core ops for library-wide
/// regression tracking.
pub fn standard_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend, M: criterion::measurement::Measurement>()
-> Vec<BenchOp<M, CoreParams>>
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GGSWEncryptSk<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    vec![
        BenchOp {
            layer: "core",
            name: "glwe_encrypt_sk",
            runner: encryption::runner_glwe_encrypt_sk::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "ggsw_encrypt_sk",
            runner: encryption::runner_ggsw_encrypt_sk::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_external_product_assign",
            runner: external_product::runner_glwe_external_product_assign::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_automorphism",
            runner: automorphism::runner_glwe_automorphism::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_keyswitch",
            runner: keyswitch::runner_glwe_keyswitch::<BE, _>,
        },
        BenchOp {
            layer: "core",
            name: "glwe_decrypt",
            runner: decryption::runner_glwe_decrypt::<BE, _>,
        },
    ]
}

// ── bench_core (criterion_group targets) ────────────────────────────────────

/// Every core op family (encryption, decryption, keyswitch, automorphism,
/// external product, tensoring, add/sub/normalize/mul-plain), swept at
/// every size matching CKKS (`log_n` 12–16) — the full tier's
/// CKKS/NTT-role sweep. `where` clause matches [`all_ops`]'s own.
pub fn bench_core_ckks<BE>(c: &mut Criterion<WallTime>)
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWENormalize<BE>
        + GLWEMulPlain<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    bench_ops(PhantomData::<BE>, &all_ops::<BE, WallTime>(), default_bench_params_core(), c);
}

/// Every core op family, pinned to the single ring degree bin-fhe's
/// representative params use — the full tier's bin-fhe/FFT-role sweep.
/// Same op table as [`bench_core_ckks`] (comprehensive family coverage),
/// but the FFT-friendly backend is only ever exercised at bin-fhe's one
/// size, so unlike `bench_core_ckks` this doesn't sweep the full `log_n`
/// grid.
pub fn bench_core_binfhe<BE>(c: &mut Criterion<WallTime>)
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWENormalize<BE>
        + GLWEMulPlain<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    bench_ops(
        PhantomData::<BE>,
        &all_ops::<BE, WallTime>(),
        default_bench_params_core().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
}

/// `standard` tier: core ops swept at the sizes matching CKKS (`log_n`
/// 13/14/15), or pinned to bin-fhe's ring degree. `where` clause matches
/// [`standard_ops`]'s own.
pub mod standard {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::layouts::{Backend, HostBackend, Module, ScratchOwned};

    use super::{
        GGSWEncryptSk, GGSWPreparedFactory, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEAutomorphismKeyPreparedFactory,
        GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, GLWEKeyswitch, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKeyEncryptSk, GLWESwitchingKeyPreparedFactory, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow,
        standard_ops,
    };
    use crate::{bench_ops, bin_fhe_n, core::params::default_bench_params_core, is_standard_n};

    /// Core ops swept at the sizes matching CKKS (`log_n` 13/14/15).
    pub fn bench_core_ckks<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        Module<BE>: ModuleNew<BE>
            + GLWEEncryptSk<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWESecretSampling<BE>
            + GGSWEncryptSk<BE>
            + GLWEExternalProduct<BE>
            + GGSWPreparedFactory<BE>
            + GLWEAutomorphism<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWEKeyswitch<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEDecrypt<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
        for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
    {
        bench_ops(
            PhantomData::<BE>,
            &standard_ops::<BE, WallTime>(),
            default_bench_params_core().into_iter().filter(|p| is_standard_n(p.n as u64)),
            c,
        );
    }

    /// Core ops pinned to the single ring degree bin-fhe's representative
    /// params use.
    pub fn bench_core_binfhe<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        Module<BE>: ModuleNew<BE>
            + GLWEEncryptSk<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWESecretSampling<BE>
            + GGSWEncryptSk<BE>
            + GLWEExternalProduct<BE>
            + GGSWPreparedFactory<BE>
            + GLWEAutomorphism<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWEKeyswitch<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEDecrypt<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
        for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
    {
        bench_ops(
            PhantomData::<BE>,
            &standard_ops::<BE, WallTime>(),
            default_bench_params_core().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
    }
}

/// `light` tier: same cross-section as [`standard`], but the CKKS-matched
/// sweep is a single size (`log_n` = 14) instead of {13, 14, 15}.
pub mod light {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::layouts::{Backend, HostBackend, Module, ScratchOwned};

    use super::{
        GGSWEncryptSk, GGSWPreparedFactory, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEAutomorphismKeyPreparedFactory,
        GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, GLWEKeyswitch, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKeyEncryptSk, GLWESwitchingKeyPreparedFactory, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow,
        standard_ops,
    };
    use crate::{bench_ops, bin_fhe_n, core::params::default_bench_params_core, is_light_n};

    /// Core ops swept at the single size matching CKKS (`log_n` = 14).
    pub fn bench_core_ckks<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        Module<BE>: ModuleNew<BE>
            + GLWEEncryptSk<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWESecretSampling<BE>
            + GGSWEncryptSk<BE>
            + GLWEExternalProduct<BE>
            + GGSWPreparedFactory<BE>
            + GLWEAutomorphism<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWEKeyswitch<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEDecrypt<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
        for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
    {
        bench_ops(
            PhantomData::<BE>,
            &standard_ops::<BE, WallTime>(),
            default_bench_params_core().into_iter().filter(|p| is_light_n(p.n as u64)),
            c,
        );
    }

    /// Core ops pinned to the single ring degree bin-fhe's representative
    /// params use.
    pub fn bench_core_binfhe<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        Module<BE>: ModuleNew<BE>
            + GLWEEncryptSk<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWESecretSampling<BE>
            + GGSWEncryptSk<BE>
            + GLWEExternalProduct<BE>
            + GGSWPreparedFactory<BE>
            + GLWEAutomorphism<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWEKeyswitch<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEDecrypt<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
        for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
    {
        bench_ops(
            PhantomData::<BE>,
            &standard_ops::<BE, WallTime>(),
            default_bench_params_core().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
    }
}
