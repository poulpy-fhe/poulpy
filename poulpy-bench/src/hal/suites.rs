use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDft, SvpApplyDftToDft,
        SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxAddAssignBackend, VecZnxAddIntoBackend, VecZnxAlloc,
        VecZnxAutomorphismAssignBackend, VecZnxAutomorphismAssignTmpBytes, VecZnxAutomorphismBackend, VecZnxBigAddAssign,
        VecZnxBigAddInto, VecZnxBigAddSmallAssign, VecZnxBigAddSmallIntoBackend, VecZnxBigAlloc, VecZnxBigAutomorphism,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigNegate, VecZnxBigNegateAssign,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign,
        VecZnxBigSubSmallABackend, VecZnxBigSubSmallBBackend, VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxIdftApply, VecZnxIdftApplyTmpA,
        VecZnxIdftApplyTmpBytes, VecZnxLshAssignBackend, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxMulXpMinusOneAssignBackend,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxMulXpMinusOneBackend, VecZnxNegateAssignBackend, VecZnxNegateBackend,
        VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VecZnxRotateAssignBackend,
        VecZnxRotateAssignTmpBytes, VecZnxRotateBackend, VecZnxRshAssignBackend, VecZnxRshBackend, VecZnxRshTmpBytes,
        VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend, VmpApplyDft, VmpApplyDftTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned},
};

use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use std::marker::PhantomData;

use criterion::{Criterion, measurement::Measurement, measurement::WallTime};

use crate::{
    BenchOp, bench_ops, bin_fhe_n,
    hal::{
        convolution,
        params::{
            CnvSweepParms, HalSweepParms, ReimSweepParams, VmpSweepParms, default_bench_params_cnv, default_bench_params_hal,
            default_bench_params_vmp,
        },
        reim, svp, vec_znx, vec_znx_big, vec_znx_dft, vmp,
    },
};

// Op tables for each HAL capability group. Each function returns the raw
// [`BenchOp`] table for that group only, scoped to the traits its own ops need —
// a backend implementing just a subset of the HAL can still build and run the
// tables for what it supports.

// ── vec_znx_dft ──────────────────────────────────────────────────────────────

pub fn vec_znx_dft_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, HalSweepParms>; 8]
where
    Module<B>: ModuleNew<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAddInto<B>
        + VecZnxDftAddAssign<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApply<B>
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftSub<B>
        + VecZnxDftSubAssign<B>
        + VecZnxDftSubNegateAssign<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    [
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_add_into",
            runner: vec_znx_dft::runner_vec_znx_dft_add_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_add_assign",
            runner: vec_znx_dft::runner_vec_znx_dft_add_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_apply",
            runner: vec_znx_dft::runner_vec_znx_dft_apply::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_idft_apply",
            runner: vec_znx_dft::runner_vec_znx_idft_apply::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_idft_apply_tmpa",
            runner: vec_znx_dft::runner_vec_znx_idft_apply_tmpa::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_sub",
            runner: vec_znx_dft::runner_vec_znx_dft_sub::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_sub_assign",
            runner: vec_znx_dft::runner_vec_znx_dft_sub_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_sub_negate_assign",
            runner: vec_znx_dft::runner_vec_znx_dft_sub_negate_assign::<B, M>,
        },
    ]
}

// ── vec_znx_big ──────────────────────────────────────────────────────────────

pub fn vec_znx_big_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, HalSweepParms>; 16]
where
    Module<B>: ModuleNew<B>
        + VecZnxAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxBigAddInto<B>
        + VecZnxBigAddAssign<B>
        + VecZnxBigAddSmallIntoBackend<B>
        + VecZnxBigAddSmallAssign<B>
        + VecZnxBigAutomorphism<B>
        + VecZnxBigAutomorphismAssign<B>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNegate<B>
        + VecZnxBigNegateAssign<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxAddAssignBackend<B>
        + VecZnxSubAssignBackend<B>
        + VecZnxBigSub<B>
        + VecZnxBigSubAssign<B>
        + VecZnxBigSubNegateAssign<B>
        + VecZnxBigSubSmallABackend<B>
        + VecZnxBigSubSmallBBackend<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    [
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_add_into",
            runner: vec_znx_big::runner_vec_znx_big_add_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_add_assign",
            runner: vec_znx_big::runner_vec_znx_big_add_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_add_small_into",
            runner: vec_znx_big::runner_vec_znx_big_add_small_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_add_small_assign",
            runner: vec_znx_big::runner_vec_znx_big_add_small_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_automorphism",
            runner: vec_znx_big::runner_vec_znx_big_automorphism::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_automorphism_assign",
            runner: vec_znx_big::runner_vec_znx_big_automorphism_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_negate",
            runner: vec_znx_big::runner_vec_znx_big_negate::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_negate_assign",
            runner: vec_znx_big::runner_vec_znx_big_negate_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_normalize",
            runner: vec_znx_big::runner_vec_znx_big_normalize::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_normalize_add_assign",
            runner: vec_znx_big::runner_vec_znx_big_normalize_add_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_normalize_sub_assign",
            runner: vec_znx_big::runner_vec_znx_big_normalize_sub_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_sub",
            runner: vec_znx_big::runner_vec_znx_big_sub::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_sub_assign",
            runner: vec_znx_big::runner_vec_znx_big_sub_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_sub_negate_assign",
            runner: vec_znx_big::runner_vec_znx_big_sub_negate_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_sub_small_a",
            runner: vec_znx_big::runner_vec_znx_big_sub_small_a::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_sub_small_b",
            runner: vec_znx_big::runner_vec_znx_big_sub_small_b::<B, M>,
        },
    ]
}

// ── svp ──────────────────────────────────────────────────────────────────────

pub fn svp_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, HalSweepParms>; 4]
where
    Module<B>: ModuleNew<B>
        + SvpPPolAlloc<B>
        + VecZnxDftAlloc<B>
        + SvpPrepare<B>
        + SvpApplyDft<B>
        + SvpApplyDftToDft<B>
        + SvpApplyDftToDftAssign<B>,
{
    [
        BenchOp {
            layer: "hal",
            name: "svp_prepare",
            runner: svp::runner_svp_prepare::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "svp_apply_dft",
            runner: svp::runner_svp_apply_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "svp_apply_dft_to_dft",
            runner: svp::runner_svp_apply_dft_to_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "svp_apply_dft_to_dft_assign",
            runner: svp::runner_svp_apply_dft_to_dft_assign::<B, M>,
        },
    ]
}

// ── vec_znx ──────────────────────────────────────────────────────────────────

pub fn vec_znx_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, HalSweepParms>; 19]
where
    Module<B>: ModuleNew<B>
        + VecZnxAlloc<B>
        + VecZnxAddIntoBackend<B>
        + VecZnxAddAssignBackend<B>
        + VecZnxAutomorphismBackend<B>
        + VecZnxAutomorphismAssignBackend<B>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxMulXpMinusOneBackend<B>
        + VecZnxMulXpMinusOneAssignBackend<B>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + VecZnxNegateBackend<B>
        + VecZnxNegateAssignBackend<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeAssignBackend<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxRotateBackend<B>
        + VecZnxRotateAssignBackend<B>
        + VecZnxRotateAssignTmpBytes
        + VecZnxLshBackend<B>
        + VecZnxLshAssignBackend<B>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<B>
        + VecZnxRshAssignBackend<B>
        + VecZnxRshTmpBytes
        + VecZnxSubBackend<B>
        + VecZnxSubAssignBackend<B>
        + VecZnxSubNegateAssignBackend<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    [
        BenchOp {
            layer: "hal",
            name: "vec_znx_add_into",
            runner: vec_znx::runner_vec_znx_add_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_add_assign",
            runner: vec_znx::runner_vec_znx_add_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_automorphism",
            runner: vec_znx::runner_vec_znx_automorphism::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_automorphism_assign",
            runner: vec_znx::runner_vec_znx_automorphism_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_mul_xp_minus_one",
            runner: vec_znx::runner_vec_znx_mul_xp_minus_one::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_mul_xp_minus_one_assign",
            runner: vec_znx::runner_vec_znx_mul_xp_minus_one_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_negate",
            runner: vec_znx::runner_vec_znx_negate::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_negate_assign",
            runner: vec_znx::runner_vec_znx_negate_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_normalize",
            runner: vec_znx::runner_vec_znx_normalize::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_normalize_assign",
            runner: vec_znx::runner_vec_znx_normalize_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_rotate",
            runner: vec_znx::runner_vec_znx_rotate::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_rotate_assign",
            runner: vec_znx::runner_vec_znx_rotate_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_lsh",
            runner: vec_znx::runner_vec_znx_lsh::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_lsh_assign",
            runner: vec_znx::runner_vec_znx_lsh_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_rsh",
            runner: vec_znx::runner_vec_znx_rsh::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_rsh_assign",
            runner: vec_znx::runner_vec_znx_rsh_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_sub",
            runner: vec_znx::runner_vec_znx_sub::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_sub_assign",
            runner: vec_znx::runner_vec_znx_sub_assign::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_sub_negate_assign",
            runner: vec_znx::runner_vec_znx_sub_negate_assign::<B, M>,
        },
    ]
}

// ── vmp ──────────────────────────────────────────────────────────────────────

pub fn vmp_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, VmpSweepParms>; 3]
where
    Module<B>: ModuleNew<B>
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpPrepareTmpBytes
        + VecZnxDftAlloc<B>
        + VmpApplyDft<B>
        + VmpApplyDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    [
        BenchOp {
            layer: "hal",
            name: "vmp_prepare",
            runner: vmp::runner_vmp_prepare::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vmp_apply_dft",
            runner: vmp::runner_vmp_apply_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vmp_apply_dft_to_dft",
            runner: vmp::runner_vmp_apply_dft_to_dft::<B, M>,
        },
    ]
}

// ── convolution ──────────────────────────────────────────────────────────────

pub fn convolution_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> [BenchOp<M, CnvSweepParms>; 6]
where
    Module<B>: ModuleNew<B> + Convolution<B> + CnvPVecAlloc<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    [
        BenchOp {
            layer: "hal",
            name: "cnv_prepare_left",
            runner: convolution::runner_cnv_prepare_left::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "cnv_prepare_right",
            runner: convolution::runner_cnv_prepare_right::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "cnv_apply_dft",
            runner: convolution::runner_cnv_apply_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "cnv_apply_dft_accumulate",
            runner: convolution::runner_cnv_apply_dft_accumulate::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "cnv_pairwise_apply_dft",
            runner: convolution::runner_cnv_pairwise_apply_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "cnv_by_const_apply",
            runner: convolution::runner_cnv_by_const_apply::<B, M>,
        },
    ]
}

// ── reim ─────────────────────────────────────────────────────────────────────

pub fn reim_ops<T: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>, M: Measurement>() -> [BenchOp<M, ReimSweepParams>; 2] {
    [
        BenchOp {
            layer: "hal",
            name: "reim_fft",
            runner: reim::runner_reim_fft::<T, M>,
        },
        BenchOp {
            layer: "hal",
            name: "reim_ifft",
            runner: reim::runner_reim_ifft::<T, M>,
        },
    ]
}

// ── all_vec_znx ──────────────────────────────────────────────────────────────

/// Concatenates the three coefficient/DFT-domain `vec_znx` groups
/// (`vec_znx`, `vec_znx_dft`, `vec_znx_big`) into a single table. Excludes
/// `svp`, `vmp`, and `convolution`, which sweep different parameter types —
/// call [`svp_ops`] separately.
pub fn all_vec_znx_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>() -> Vec<BenchOp<M, HalSweepParms>>
where
    Module<B>: ModuleNew<B>
        + VecZnxAlloc<B>
        + VecZnxAddIntoBackend<B>
        + VecZnxAddAssignBackend<B>
        + VecZnxAutomorphismBackend<B>
        + VecZnxAutomorphismAssignBackend<B>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxMulXpMinusOneBackend<B>
        + VecZnxMulXpMinusOneAssignBackend<B>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + VecZnxNegateBackend<B>
        + VecZnxNegateAssignBackend<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeAssignBackend<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxRotateBackend<B>
        + VecZnxRotateAssignBackend<B>
        + VecZnxRotateAssignTmpBytes
        + VecZnxLshBackend<B>
        + VecZnxLshAssignBackend<B>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<B>
        + VecZnxRshAssignBackend<B>
        + VecZnxRshTmpBytes
        + VecZnxSubBackend<B>
        + VecZnxSubAssignBackend<B>
        + VecZnxSubNegateAssignBackend<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAddInto<B>
        + VecZnxDftAddAssign<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApply<B>
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftSub<B>
        + VecZnxDftSubAssign<B>
        + VecZnxDftSubNegateAssign<B>
        + VecZnxBigAddInto<B>
        + VecZnxBigAddAssign<B>
        + VecZnxBigAddSmallIntoBackend<B>
        + VecZnxBigAddSmallAssign<B>
        + VecZnxBigAutomorphism<B>
        + VecZnxBigAutomorphismAssign<B>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNegate<B>
        + VecZnxBigNegateAssign<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSub<B>
        + VecZnxBigSubAssign<B>
        + VecZnxBigSubNegateAssign<B>
        + VecZnxBigSubSmallABackend<B>
        + VecZnxBigSubSmallBBackend<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let mut ops: Vec<BenchOp<M, HalSweepParms>> = Vec::new();
    ops.extend(vec_znx_ops::<B, M>());
    ops.extend(vec_znx_dft_ops::<B, M>());
    ops.extend(vec_znx_big_ops::<B, M>());
    ops
}

// ── standard ─────────────────────────────────────────────────────────────────

/// A small, representative cross-section of HAL ops for library-wide
/// regression tracking. Returns two tables since `vmp_apply_dft_to_dft`
/// sweeps a different parameter type than the rest.
#[allow(clippy::type_complexity)]
pub fn standard_ops<B: Backend<ZnxWord = i64> + 'static, M: Measurement>()
-> (Vec<BenchOp<M, HalSweepParms>>, [BenchOp<M, VmpSweepParms>; 1])
where
    Module<B>: ModuleNew<B>
        + VecZnxDftApply<B>
        + VecZnxDftAlloc<B>
        + VecZnxIdftApply<B>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAlloc<B>
        + SvpApplyDftToDft<B>
        + SvpPPolAlloc<B>
        + VecZnxAddIntoBackend<B>
        + VecZnxAlloc<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAddInto<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let hal_ops = vec![
        BenchOp {
            layer: "hal",
            name: "vec_znx_dft_apply",
            runner: vec_znx_dft::runner_vec_znx_dft_apply::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_idft_apply",
            runner: vec_znx_dft::runner_vec_znx_idft_apply::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "svp_apply_dft_to_dft",
            runner: svp::runner_svp_apply_dft_to_dft::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_add_into",
            runner: vec_znx::runner_vec_znx_add_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_normalize",
            runner: vec_znx::runner_vec_znx_normalize::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_add_into",
            runner: vec_znx_big::runner_vec_znx_big_add_into::<B, M>,
        },
        BenchOp {
            layer: "hal",
            name: "vec_znx_big_normalize",
            runner: vec_znx_big::runner_vec_znx_big_normalize::<B, M>,
        },
    ];
    let vmp_ops = [BenchOp {
        layer: "hal",
        name: "vmp_apply_dft_to_dft",
        runner: vmp::runner_vmp_apply_dft_to_dft::<B, M>,
    }];
    (hal_ops, vmp_ops)
}

// ── bench_hal (criterion_group targets) ─────────────────────────────────────

/// Every HAL op family (`vec_znx`, `vec_znx_dft`, `vec_znx_big`, `svp`,
/// `vmp`, `convolution`), swept at every size matching CKKS (`log_n`
/// 10–15) — the full tier's CKKS/NTT-role sweep. `where` clause is the
/// union of [`all_vec_znx_ops`], [`svp_ops`], [`vmp_ops`], and
/// [`convolution_ops`]'s own.
pub fn bench_hal_ckks<B>(c: &mut Criterion<WallTime>)
where
    B: Backend<ZnxWord = i64> + 'static,
    Module<B>: ModuleNew<B>
        + VecZnxAlloc<B>
        + VecZnxAddIntoBackend<B>
        + VecZnxAddAssignBackend<B>
        + VecZnxAutomorphismBackend<B>
        + VecZnxAutomorphismAssignBackend<B>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxMulXpMinusOneBackend<B>
        + VecZnxMulXpMinusOneAssignBackend<B>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + VecZnxNegateBackend<B>
        + VecZnxNegateAssignBackend<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeAssignBackend<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxRotateBackend<B>
        + VecZnxRotateAssignBackend<B>
        + VecZnxRotateAssignTmpBytes
        + VecZnxLshBackend<B>
        + VecZnxLshAssignBackend<B>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<B>
        + VecZnxRshAssignBackend<B>
        + VecZnxRshTmpBytes
        + VecZnxSubBackend<B>
        + VecZnxSubAssignBackend<B>
        + VecZnxSubNegateAssignBackend<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAddInto<B>
        + VecZnxDftAddAssign<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApply<B>
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftSub<B>
        + VecZnxDftSubAssign<B>
        + VecZnxDftSubNegateAssign<B>
        + VecZnxBigAddInto<B>
        + VecZnxBigAddAssign<B>
        + VecZnxBigAddSmallIntoBackend<B>
        + VecZnxBigAddSmallAssign<B>
        + VecZnxBigAutomorphism<B>
        + VecZnxBigAutomorphismAssign<B>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNegate<B>
        + VecZnxBigNegateAssign<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSub<B>
        + VecZnxBigSubAssign<B>
        + VecZnxBigSubNegateAssign<B>
        + VecZnxBigSubSmallABackend<B>
        + VecZnxBigSubSmallBBackend<B>
        + SvpPPolAlloc<B>
        + SvpPrepare<B>
        + SvpApplyDft<B>
        + SvpApplyDftToDft<B>
        + SvpApplyDftToDftAssign<B>
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpPrepareTmpBytes
        + VmpApplyDft<B>
        + VmpApplyDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftTmpBytes
        + Convolution<B>
        + CnvPVecAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    bench_ops(
        PhantomData::<B>,
        &all_vec_znx_ops::<B, WallTime>(),
        default_bench_params_hal(),
        c,
    );
    bench_ops(PhantomData::<B>, &svp_ops::<B, WallTime>(), default_bench_params_hal(), c);
    bench_ops(PhantomData::<B>, &vmp_ops::<B, WallTime>(), default_bench_params_vmp(), c);
    bench_ops(
        PhantomData::<B>,
        &convolution_ops::<B, WallTime>(),
        default_bench_params_cnv(),
        c,
    );
}

/// Every HAL op family, pinned to the single ring degree bin-fhe's
/// representative params use — the full tier's bin-fhe/FFT-role sweep.
/// Same op tables as [`bench_hal_ckks`] (comprehensive family coverage),
/// but the FFT-friendly backend is only ever exercised at bin-fhe's one
/// size, so unlike `bench_hal_ckks` this doesn't sweep the full `log_n`
/// grid.
pub fn bench_hal_binfhe<B>(c: &mut Criterion<WallTime>)
where
    B: Backend<ZnxWord = i64> + 'static,
    Module<B>: ModuleNew<B>
        + VecZnxAlloc<B>
        + VecZnxAddIntoBackend<B>
        + VecZnxAddAssignBackend<B>
        + VecZnxAutomorphismBackend<B>
        + VecZnxAutomorphismAssignBackend<B>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxMulXpMinusOneBackend<B>
        + VecZnxMulXpMinusOneAssignBackend<B>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + VecZnxNegateBackend<B>
        + VecZnxNegateAssignBackend<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeAssignBackend<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxRotateBackend<B>
        + VecZnxRotateAssignBackend<B>
        + VecZnxRotateAssignTmpBytes
        + VecZnxLshBackend<B>
        + VecZnxLshAssignBackend<B>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<B>
        + VecZnxRshAssignBackend<B>
        + VecZnxRshTmpBytes
        + VecZnxSubBackend<B>
        + VecZnxSubAssignBackend<B>
        + VecZnxSubNegateAssignBackend<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAddInto<B>
        + VecZnxDftAddAssign<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApply<B>
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftSub<B>
        + VecZnxDftSubAssign<B>
        + VecZnxDftSubNegateAssign<B>
        + VecZnxBigAddInto<B>
        + VecZnxBigAddAssign<B>
        + VecZnxBigAddSmallIntoBackend<B>
        + VecZnxBigAddSmallAssign<B>
        + VecZnxBigAutomorphism<B>
        + VecZnxBigAutomorphismAssign<B>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNegate<B>
        + VecZnxBigNegateAssign<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSub<B>
        + VecZnxBigSubAssign<B>
        + VecZnxBigSubNegateAssign<B>
        + VecZnxBigSubSmallABackend<B>
        + VecZnxBigSubSmallBBackend<B>
        + SvpPPolAlloc<B>
        + SvpPrepare<B>
        + SvpApplyDft<B>
        + SvpApplyDftToDft<B>
        + SvpApplyDftToDftAssign<B>
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpPrepareTmpBytes
        + VmpApplyDft<B>
        + VmpApplyDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftTmpBytes
        + Convolution<B>
        + CnvPVecAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    bench_ops(
        PhantomData::<B>,
        &all_vec_znx_ops::<B, WallTime>(),
        default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
    bench_ops(
        PhantomData::<B>,
        &svp_ops::<B, WallTime>(),
        default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
    bench_ops(
        PhantomData::<B>,
        &vmp_ops::<B, WallTime>(),
        default_bench_params_vmp().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
    bench_ops(
        PhantomData::<B>,
        &convolution_ops::<B, WallTime>(),
        default_bench_params_cnv().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
}

/// `standard` tier: HAL ops swept at the sizes matching CKKS (`log_n`
/// 13/14/15), or pinned to bin-fhe's ring degree. `where` clause matches
/// [`standard_ops`]'s own.
pub mod standard {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::layouts::{Backend, Module, ScratchOwned};

    use super::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpPPolAlloc, VecZnxAddIntoBackend, VecZnxAlloc,
        VecZnxBigAddInto, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, standard_ops,
    };
    use crate::{
        bench_ops, bin_fhe_n,
        hal::params::{default_bench_params_hal, default_bench_params_vmp},
        is_standard_n,
    };

    /// HAL ops swept at the sizes matching CKKS (`log_n` 13/14/15).
    pub fn bench_hal_ckks<B>(c: &mut Criterion<WallTime>)
    where
        B: Backend<ZnxWord = i64> + 'static,
        Module<B>: ModuleNew<B>
            + VecZnxDftApply<B>
            + VecZnxDftAlloc<B>
            + VecZnxIdftApply<B>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigAlloc<B>
            + SvpApplyDftToDft<B>
            + SvpPPolAlloc<B>
            + VecZnxAddIntoBackend<B>
            + VecZnxAlloc<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAddInto<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigNormalizeTmpBytes
            + VmpPMatAlloc<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let (hal_ops, vmp_ops) = standard_ops::<B, WallTime>();
        bench_ops(
            PhantomData::<B>,
            &hal_ops,
            default_bench_params_hal().into_iter().filter(|p| is_standard_n(p.n as u64)),
            c,
        );
        bench_ops(
            PhantomData::<B>,
            &vmp_ops,
            default_bench_params_vmp().into_iter().filter(|p| is_standard_n(p.n as u64)),
            c,
        );
    }

    /// HAL ops pinned to the single ring degree bin-fhe's representative
    /// params use.
    pub fn bench_hal_binfhe<B>(c: &mut Criterion<WallTime>)
    where
        B: Backend<ZnxWord = i64> + 'static,
        Module<B>: ModuleNew<B>
            + VecZnxDftApply<B>
            + VecZnxDftAlloc<B>
            + VecZnxIdftApply<B>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigAlloc<B>
            + SvpApplyDftToDft<B>
            + SvpPPolAlloc<B>
            + VecZnxAddIntoBackend<B>
            + VecZnxAlloc<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAddInto<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigNormalizeTmpBytes
            + VmpPMatAlloc<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let (hal_ops, vmp_ops) = standard_ops::<B, WallTime>();
        bench_ops(
            PhantomData::<B>,
            &hal_ops,
            default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
        bench_ops(
            PhantomData::<B>,
            &vmp_ops,
            default_bench_params_vmp().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
    }
}

/// `light` tier: same cross-section as [`standard`], but the CKKS-matched
/// sweep is a single size (`log_n` = 14) instead of {13, 14, 15}.
pub mod light {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::layouts::{Backend, Module, ScratchOwned};

    use super::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpPPolAlloc, VecZnxAddIntoBackend, VecZnxAlloc,
        VecZnxBigAddInto, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, standard_ops,
    };
    use crate::{
        bench_ops, bin_fhe_n,
        hal::params::{default_bench_params_hal, default_bench_params_vmp},
        is_light_n,
    };

    /// HAL ops swept at the single size matching CKKS (`log_n` = 14).
    pub fn bench_hal_ckks<B>(c: &mut Criterion<WallTime>)
    where
        B: Backend<ZnxWord = i64> + 'static,
        Module<B>: ModuleNew<B>
            + VecZnxDftApply<B>
            + VecZnxDftAlloc<B>
            + VecZnxIdftApply<B>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigAlloc<B>
            + SvpApplyDftToDft<B>
            + SvpPPolAlloc<B>
            + VecZnxAddIntoBackend<B>
            + VecZnxAlloc<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAddInto<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigNormalizeTmpBytes
            + VmpPMatAlloc<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let (hal_ops, vmp_ops) = standard_ops::<B, WallTime>();
        bench_ops(
            PhantomData::<B>,
            &hal_ops,
            default_bench_params_hal().into_iter().filter(|p| is_light_n(p.n as u64)),
            c,
        );
        bench_ops(
            PhantomData::<B>,
            &vmp_ops,
            default_bench_params_vmp().into_iter().filter(|p| is_light_n(p.n as u64)),
            c,
        );
    }

    /// HAL ops pinned to the single ring degree bin-fhe's representative
    /// params use.
    pub fn bench_hal_binfhe<B>(c: &mut Criterion<WallTime>)
    where
        B: Backend<ZnxWord = i64> + 'static,
        Module<B>: ModuleNew<B>
            + VecZnxDftApply<B>
            + VecZnxDftAlloc<B>
            + VecZnxIdftApply<B>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigAlloc<B>
            + SvpApplyDftToDft<B>
            + SvpPPolAlloc<B>
            + VecZnxAddIntoBackend<B>
            + VecZnxAlloc<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAddInto<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigNormalizeTmpBytes
            + VmpPMatAlloc<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let (hal_ops, vmp_ops) = standard_ops::<B, WallTime>();
        bench_ops(
            PhantomData::<B>,
            &hal_ops,
            default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
        bench_ops(
            PhantomData::<B>,
            &vmp_ops,
            default_bench_params_vmp().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
            c,
        );
    }
}
