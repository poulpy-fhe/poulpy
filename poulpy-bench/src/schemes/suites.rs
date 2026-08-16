use poulpy_bin_fhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, BlindRotationKeyEncryptSk, BlindRotationKeyPreparedFactory, LookupTableFactory,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingExecute, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPreparedFactory,
    },
};
use poulpy_ckks::api::{
    CKKSAddOps, CKKSConjugateOps, CKKSEncodingOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
};
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    layouts::{
        GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWETensorKeyPreparedFactory, LWESecretSampling,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssignBackend},
    layouts::{Backend, HostBackend, Module, ScratchOwned},
};

use crate::{
    BenchOp,
    schemes::{
        bin_fhe, ckks,
        params::{BlindRotateBenchParams, CircuitBootstrappingBenchParam, CkksBenchParams},
    },
};

// Op tables for each CKKS capability group, one per `poulpy-ckks` API trait.
// Each function returns the raw [`BenchOp`] table for that group only,
// scoped to the traits its own ops need — a backend implementing just a
// subset of the CKKS API can still build and run the tables for what it
// supports.

// ── add ──────────────────────────────────────────────────────────────────────

pub fn ckks_add_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 3]
where
    Module<BE>: ModuleNew<BE> + CKKSAddOps<BE>,
{
    [
        BenchOp {
            name: "ckks_add_into",
            runner: ckks::runner_ckks_add_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_add_pt_vec_into",
            runner: ckks::runner_ckks_add_pt_vec_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_add_pt_const_into",
            runner: ckks::runner_ckks_add_pt_const_into::<BE, _>,
        },
    ]
}

// ── sub ──────────────────────────────────────────────────────────────────────

pub fn ckks_sub_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 3]
where
    Module<BE>: ModuleNew<BE> + CKKSSubOps<BE>,
{
    [
        BenchOp {
            name: "ckks_sub_into",
            runner: ckks::runner_ckks_sub_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_sub_pt_vec_into",
            runner: ckks::runner_ckks_sub_pt_vec_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_sub_pt_const_into",
            runner: ckks::runner_ckks_sub_pt_const_into::<BE, _>,
        },
    ]
}

// ── neg ──────────────────────────────────────────────────────────────────────

pub fn ckks_neg_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 1]
where
    Module<BE>: ModuleNew<BE> + CKKSNegOps<BE>,
{
    [BenchOp {
        name: "ckks_neg_into",
        runner: ckks::runner_ckks_neg_into::<BE, _>,
    }]
}

// ── pow2 ─────────────────────────────────────────────────────────────────────

pub fn ckks_pow2_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 2]
where
    Module<BE>: ModuleNew<BE> + CKKSPow2Ops<BE>,
{
    [
        BenchOp {
            name: "ckks_mul_pow2_into",
            runner: ckks::runner_ckks_mul_pow2_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_div_pow2_into",
            runner: ckks::runner_ckks_div_pow2_into::<BE, _>,
        },
    ]
}

// ── mul ──────────────────────────────────────────────────────────────────────

pub fn ckks_mul_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 4]
where
    Module<BE>: ModuleNew<BE> + CKKSMulOps<BE> + GLWETensorKeyPreparedFactory<BE>,
{
    [
        BenchOp {
            name: "ckks_mul_into",
            runner: ckks::runner_ckks_mul_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_square_into",
            runner: ckks::runner_ckks_square_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_mul_pt_vec_into",
            runner: ckks::runner_ckks_mul_pt_vec_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_mul_pt_const_into",
            runner: ckks::runner_ckks_mul_pt_const_into::<BE, _>,
        },
    ]
}

// ── rotate ───────────────────────────────────────────────────────────────────

pub fn ckks_rotate_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 1]
where
    Module<BE>: ModuleNew<BE> + CKKSRotateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    [BenchOp {
        name: "ckks_rotate_into",
        runner: ckks::runner_ckks_rotate_into::<BE, _>,
    }]
}

// ── conjugate ────────────────────────────────────────────────────────────────

pub fn ckks_conjugate_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 1]
where
    Module<BE>: ModuleNew<BE> + CKKSConjugateOps<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    [BenchOp {
        name: "ckks_conjugate_into",
        runner: ckks::runner_ckks_conjugate_into::<BE, _>,
    }]
}

// ── encoding ─────────────────────────────────────────────────────────────────

pub fn ckks_encoding_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> [BenchOp<M, CkksBenchParams>; 4]
where
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, f64>,
{
    [
        BenchOp {
            name: "ckks_encode_slots_assign_into",
            runner: ckks::runner_ckks_encode_slots_assign_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_decode_slots_into",
            runner: ckks::runner_ckks_decode_slots_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_encode_coeffs_into",
            runner: ckks::runner_ckks_encode_coeffs_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_decode_coeffs_into",
            runner: ckks::runner_ckks_decode_coeffs_into::<BE, _>,
        },
    ]
}

// ── all ──────────────────────────────────────────────────────────────────────

/// Concatenates every CKKS-layer group into a single table. Requires a
/// backend that implements the full CKKS API; a backend supporting only part
/// of it should instead compose the `ckks_*_ops` tables it needs directly.
pub fn all_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend, M: criterion::measurement::Measurement>()
-> Vec<BenchOp<M, CkksBenchParams>>
where
    Module<BE>: ModuleNew<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSMulOps<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + CKKSRotateOps<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CKKSConjugateOps<BE>
        + CKKSEncodingOps<BE, f64>,
{
    let mut ops: Vec<BenchOp<M, CkksBenchParams>> = Vec::new();
    ops.extend(ckks_add_ops::<BE, _>());
    ops.extend(ckks_sub_ops::<BE, _>());
    ops.extend(ckks_neg_ops::<BE, _>());
    ops.extend(ckks_pow2_ops::<BE, _>());
    ops.extend(ckks_mul_ops::<BE, _>());
    ops.extend(ckks_rotate_ops::<BE, _>());
    ops.extend(ckks_conjugate_ops::<BE, _>());
    ops.extend(ckks_encoding_ops::<BE, _>());
    ops
}

// ── bin_fhe ──────────────────────────────────────────────────────────────────

pub fn blind_rotate_ops<
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>,
    BRA: BlindRotationAlgo,
    M: criterion::measurement::Measurement,
>() -> [BenchOp<M, BlindRotateBenchParams>; 1]
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory<BE::OwnedBuf, BE::ZnxWord>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    [BenchOp {
        name: "blind_rotate",
        runner: bin_fhe::runner_blind_rotate::<BE, BRA, _>,
    }]
}

pub fn circuit_bootstrapping_ops<
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
    BRA: BlindRotationAlgo,
    M: criterion::measurement::Measurement,
>() -> [BenchOp<M, CircuitBootstrappingBenchParam>; 1]
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CircuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssignBackend<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    [BenchOp {
        name: "circuit_bootstrapping",
        runner: bin_fhe::runner_circuit_bootstrapping::<BE, BRA, _>,
    }]
}

// ── standard ─────────────────────────────────────────────────────────────────

/// A small, representative cross-section of CKKS ops for library-wide
/// regression tracking. Kept independent of [`bin_fhe_standard_ops`] (rather
/// than one combined function) since the two schemes are typically
/// benchmarked against different backends (e.g. an NTT-friendly backend for
/// CKKS, an FFT-friendly one for bin-fhe).
pub fn ckks_standard_ops<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: criterion::measurement::Measurement>()
-> Vec<BenchOp<M, CkksBenchParams>>
where
    Module<BE>: ModuleNew<BE>
        + CKKSAddOps<BE>
        + CKKSMulOps<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + CKKSRotateOps<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CKKSEncodingOps<BE, f64>,
{
    vec![
        BenchOp {
            name: "ckks_add_into",
            runner: ckks::runner_ckks_add_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_mul_into",
            runner: ckks::runner_ckks_mul_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_mul_pt_vec_into",
            runner: ckks::runner_ckks_mul_pt_vec_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_rotate_into",
            runner: ckks::runner_ckks_rotate_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_encode_slots_assign_into",
            runner: ckks::runner_ckks_encode_slots_assign_into::<BE, _>,
        },
        BenchOp {
            name: "ckks_decode_slots_into",
            runner: ckks::runner_ckks_decode_slots_into::<BE, _>,
        },
    ]
}

/// A small, representative cross-section of bin-fhe ops for library-wide
/// regression tracking. Returns two tables since blind rotation and circuit
/// bootstrapping sweep different parameter types. See
/// [`ckks_standard_ops`] for why this is kept separate from the CKKS table.
#[allow(clippy::type_complexity)]
pub fn bin_fhe_standard_ops<
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
    BRA: BlindRotationAlgo,
    M: criterion::measurement::Measurement,
>() -> (
    [BenchOp<M, BlindRotateBenchParams>; 1],
    [BenchOp<M, CircuitBootstrappingBenchParam>; 1],
)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory<BE::OwnedBuf, BE::ZnxWord>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>
        + GLWEExternalProduct<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CircuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let blind_rotate = [BenchOp {
        name: "blind_rotate",
        runner: bin_fhe::runner_blind_rotate::<BE, BRA, _>,
    }];
    let circuit_bootstrapping = [BenchOp {
        name: "circuit_bootstrapping",
        runner: bin_fhe::runner_circuit_bootstrapping::<BE, BRA, _>,
    }];
    (blind_rotate, circuit_bootstrapping)
}
