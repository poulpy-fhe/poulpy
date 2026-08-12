use poulpy_core::{
    GGSWEncryptSk, GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct,
    GLWEKeyswitch, GLWEMulPlain, GLWENormalize, GLWESub, GLWESwitchingKeyEncryptSk, GLWETensoring,
    layouts::{
        GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESwitchingKeyPreparedFactory,
        GLWETensorKeyPreparedFactory,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxIdftApplyTmpA,
        VecZnxSubAssignBackend,
    },
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchOwned},
};

use crate::{
    BenchOp,
    core::{automorphism, decryption, encryption, external_product, glwe_tensor, keyswitch, operations},
    params::CoreParams,
};

/// Op tables for each core capability group. Each function returns the raw
/// [`BenchOp`] table for that group only, scoped to the traits its own ops
/// need — a backend implementing just a subset of `poulpy-core` can still
/// build and run the tables for what it supports. Compose across groups and
/// drive [`bench_ops`](crate::bench_ops) directly, or use [`all_ops`]
/// to run every group at once.

// ── encryption ───────────────────────────────────────────────────────────────

pub fn encryption_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>() -> [BenchOp<M, CoreParams>; 3]
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp { name: "glwe_encrypt_sk", runner: encryption::runner_glwe_encrypt_sk::<BE, _> },
        BenchOp { name: "ggsw_encrypt_sk", runner: encryption::runner_ggsw_encrypt_sk::<BE, _> },
        BenchOp { name: "glwe_automorphism_key_encrypt_sk", runner: encryption::runner_glwe_automorphism_key_encrypt_sk::<BE, _> },
    ]
}

// ── decryption ───────────────────────────────────────────────────────────────

pub fn decryption_ops<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE> + GLWEDecrypt<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp { name: "glwe_decrypt", runner: decryption::runner_glwe_decrypt::<BE, _> }]
}

// ── automorphism ─────────────────────────────────────────────────────────────

pub fn automorphism_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>() -> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp { name: "glwe_automorphism", runner: automorphism::runner_glwe_automorphism::<BE, _> }]
}

// ── external_product ─────────────────────────────────────────────────────────

pub fn external_product_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CoreParams>; 2]
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp { name: "glwe_external_product", runner: external_product::runner_glwe_external_product::<BE, _> },
        BenchOp { name: "glwe_external_product_assign", runner: external_product::runner_glwe_external_product_assign::<BE, _> },
    ]
}

// ── keyswitch ────────────────────────────────────────────────────────────────

pub fn keyswitch_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>() -> [BenchOp<M, CoreParams>; 1]
where
    Module<BE>: ModuleNew<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [BenchOp { name: "glwe_keyswitch", runner: keyswitch::runner_glwe_keyswitch::<BE, _> }]
}

// ── glwe_tensor ──────────────────────────────────────────────────────────────

pub fn glwe_tensor_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>() -> [BenchOp<M, CoreParams>; 7]
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxSubAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    [
        BenchOp { name: "glwe_tensor_relinearize", runner: glwe_tensor::runner_glwe_tensor_relinearize::<BE, _> },
        BenchOp { name: "glwe_tensor_apply", runner: glwe_tensor::runner_glwe_tensor_apply::<BE, _> },
        BenchOp { name: "glwe_tensor_prepare_left", runner: glwe_tensor::runner_glwe_tensor_prepare_left::<BE, _> },
        BenchOp { name: "glwe_tensor_prepare_right", runner: glwe_tensor::runner_glwe_tensor_prepare_right::<BE, _> },
        BenchOp { name: "glwe_tensor_diag_lane", runner: glwe_tensor::runner_glwe_tensor_diag_lane::<BE, _> },
        BenchOp { name: "glwe_tensor_pairwise_lane", runner: glwe_tensor::runner_glwe_tensor_pairwise_lane::<BE, _> },
        BenchOp { name: "glwe_tensor_square_apply", runner: glwe_tensor::runner_glwe_tensor_square_apply::<BE, _> },
    ]
}

// ── operations ───────────────────────────────────────────────────────────────

pub fn operations_ops<BE: Backend<OwnedBuf = Vec<u8>>, M: criterion::measurement::Measurement>() -> [BenchOp<M, CoreParams>; 8]
where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE> + GLWESub<BE> + GLWENormalize<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    [
        BenchOp { name: "glwe_add_into", runner: operations::runner_glwe_add_into::<BE, _> },
        BenchOp { name: "glwe_add_assign", runner: operations::runner_glwe_add_assign::<BE, _> },
        BenchOp { name: "glwe_sub", runner: operations::runner_glwe_sub::<BE, _> },
        BenchOp { name: "glwe_sub_assign", runner: operations::runner_glwe_sub_assign::<BE, _> },
        BenchOp { name: "glwe_normalize", runner: operations::runner_glwe_normalize::<BE, _> },
        BenchOp { name: "glwe_normalize_assign", runner: operations::runner_glwe_normalize_assign::<BE, _> },
        BenchOp { name: "glwe_mul_plain", runner: operations::runner_glwe_mul_plain::<BE, _> },
        BenchOp { name: "glwe_mul_plain_assign", runner: operations::runner_glwe_mul_plain_assign::<BE, _> },
    ]
}

// ── all ──────────────────────────────────────────────────────────────────────

/// Concatenates every core-layer group into a single table. Requires a
/// backend that implements the full `poulpy-core` surface; a backend
/// supporting only part of it should instead compose the `*_ops` tables it
/// needs directly.
pub fn all_ops<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, M: criterion::measurement::Measurement>() -> Vec<BenchOp<M, CoreParams>> 
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
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxSubAssignBackend<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWENormalize<BE>
        + GLWEMulPlain<BE>,
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
