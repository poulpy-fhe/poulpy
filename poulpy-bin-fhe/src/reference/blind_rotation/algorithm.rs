#![allow(clippy::too_many_arguments)]
use itertools::izip;
use poulpy_hal::layouts::SvpPPolToBackendRef;
use poulpy_hal::layouts::VmpPMatToBackendRef;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDft, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftSubAssign,
        VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxMulXpMinusOneAssignTmpBytes, VecZnxRotate, VecZnxZero,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, SvpPPolOwned, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendRef,
        vec_znx_backend_ref_from_mut, vec_znx_big_backend_ref_from_mut, vec_znx_dft_backend_ref_from_mut,
    },
    oep::HalVecZnxImpl,
};

use poulpy_core::{
    Distribution, GLWEAdd, GLWECopy, GLWEExternalProduct, GLWEMulXpMinusOne, GLWENormalize, GLWEZero, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef, ModuleCoreAlloc},
};

use crate::api::BlindRotationModSwitch;
use crate::blind_rotation::{BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, LookupTable};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;

/// Canonical lower-layer composition for `blind_rotation_execute_tmp_bytes`.
pub fn blind_rotation_execute_tmp_bytes_ref<BE, G, B>(
    module: &Module<BE>,
    block_size: usize,
    extension_factor: usize,
    glwe_infos: &G,
    brk_infos: &B,
) -> usize
where
    G: GLWEInfos,
    B: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    blind_rotation_execute_tmp_bytes_selected::<BE, G, B, false>(module, block_size, extension_factor, glwe_infos, brk_infos)
}

/// Explicit block scheduling helper; selected by backend wiring, never inferred.
pub fn blind_rotation_execute_tmp_bytes_parallel<BE, G, B>(
    module: &Module<BE>,
    block_size: usize,
    extension_factor: usize,
    glwe_infos: &G,
    brk_infos: &B,
) -> usize
where
    G: GLWEInfos,
    B: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    blind_rotation_execute_tmp_bytes_selected::<BE, G, B, true>(module, block_size, extension_factor, glwe_infos, brk_infos)
}

fn blind_rotation_execute_tmp_bytes_selected<BE, G, B, const PARALLEL: bool>(
    module: &Module<BE>,
    block_size: usize,
    extension_factor: usize,
    glwe_infos: &G,
    brk_infos: &B,
) -> usize
where
    G: GLWEInfos,
    B: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    assert!(block_size > 0 && extension_factor.is_power_of_two());
    // The owned output temporary is compact even when the destination retains
    // spare capacity. Copy workspace can depend on either operand's allocation.
    let compact = glwe_infos.glwe_layout();
    let copy_out = if extension_factor == 1 {
        module.glwe_copy_tmp_bytes(glwe_infos, &compact)
    } else {
        0 // Extended execution copies coefficients directly with VecZnxCopy.
    };
    if block_size > 1 || extension_factor > 1 {
        let cols = brk_infos.rank().as_usize() + 1;
        let size = brk_infos.size();
        let dnum = brk_infos.dnum().as_usize();
        let acc_dft = module.bytes_of_vec_znx_dft(module.n(), cols, dnum) * extension_factor;
        let vmp_res = module.bytes_of_vec_znx_dft(module.n(), cols, size) * extension_factor;
        let vmp = module.vmp_apply_dft_to_dft_tmp_bytes(size, dnum, dnum, cols, cols, size);
        let normalize = module.bytes_of_vec_znx_big(module.n(), 1, size)
            + module
                .vec_znx_big_normalize_tmp_bytes()
                .max(module.vec_znx_idft_apply_tmp_bytes());
        if extension_factor == 1 && PARALLEL {
            acc_dft
                + 2 * block_size * vmp_res
                + (block_size * poulpy_hal::execution::worker_scratch_bytes::<BE>(vmp))
                    .max(normalize)
                    .max(copy_out)
        } else {
            let acc = if extension_factor > 1 {
                BE::bytes_of_vec_znx(module.n(), cols, glwe_infos.size()) * extension_factor
            } else {
                0
            };
            acc + acc_dft + 2 * vmp_res + module.bytes_of_vec_znx_dft(module.n(), 1, size) + vmp.max(normalize).max(copy_out)
        }
    } else {
        let copy_in = module.glwe_copy_tmp_bytes(&compact, glwe_infos);
        let acc = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&compact));
        // The initial copy precedes acc_tmp; subsequent operations, including
        // the final copy, execute while that compact scratch allocation is live.
        copy_in.max(
            acc + module
                .glwe_external_product_tmp_bytes(&compact, &compact, brk_infos)
                .max(module.vec_znx_mul_xp_minus_one_assign_tmp_bytes(compact.size()))
                .max(module.glwe_normalize_tmp_bytes())
                .max(copy_out),
        )
    }
}

/// Canonical lower-layer composition for `blind_rotation_execute`.
pub fn blind_rotation_execute_ref<BE, R, L>(
    module: &Module<BE>,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    blind_rotation_execute_selected::<BE, R, L, false>(module, res, lwe, lut, brk, scratch)
}

/// Explicit block scheduling helper; selected by backend wiring, never inferred.
pub fn blind_rotation_execute_parallel<BE, R, L>(
    module: &Module<BE>,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    blind_rotation_execute_selected::<BE, R, L, true>(module, res, lwe, lut, brk, scratch)
}

fn blind_rotation_execute_selected<BE, R, L, const PARALLEL: bool>(
    module: &Module<BE>,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxMulXpMinusOneAssignTmpBytes
        + Sync,
    BE: HalVecZnxImpl,
{
    assert_eq!(module.n(), res.n().as_usize());
    assert_eq!(res.n(), brk.n_glwe());
    assert_eq!(res.rank(), brk.rank());
    assert_eq!(res.base2k(), brk.base2k());
    assert_eq!(res.base2k(), lut.base2k);
    assert_eq!(lwe.n(), brk.n_lwe());
    assert!(lut.extension_factor().is_power_of_two());
    assert!(brk.block_size() > 0);
    assert_eq!(brk.n_lwe().as_usize() % brk.block_size(), 0);
    assert!(lut.data.iter().all(|polynomial| polynomial.n() == res.n()));

    match brk.dist {
        Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {
            if lut.extension_factor() > 1 {
                assert!(
                    matches!(brk.dist, Distribution::BinaryBlock(_)),
                    "extended blind rotation (extension_factor={}) requires a BinaryBlock key distribution, got {:?}",
                    lut.extension_factor(),
                    brk.dist,
                );
                execute_block_binary_extended(module, res, lwe, lut, brk, scratch)
            } else if brk.block_size() > 1 {
                execute_block_binary::<_, _, _, BE, PARALLEL>(module, res, lwe, lut, brk, scratch);
            } else {
                execute_standard(module, res, lwe, lut, brk, scratch);
            }
        }
        _ => panic!("invalid CGGI distribution (have you prepared the key?)"),
    }
}

fn execute_block_binary_extended<R, L, M, BE: Backend<ZnxWord = i64>>(
    module: &M,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    M: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + VecZnxBigBytesOf
        + VecZnxCopy<BE>
        + VecZnxZero<BE>,
{
    let n_glwe: usize = brk.n_glwe().into();
    let extension_factor: usize = lut.extension_factor();
    let base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let dnum: usize = brk.dnum().into();
    let cols: usize = (res.rank() + 1).into();

    let scratch = scratch.borrow();
    let (mut acc, scratch_1) = scratch.take_vec_znx_slice_scratch(extension_factor, n_glwe, cols, res.size());
    let (mut acc_dft, scratch_2) = scratch_1.take_vec_znx_dft_slice_scratch(module.n(), extension_factor, cols, dnum);
    let (mut vmp_res, scratch_3) = scratch_2.take_vec_znx_dft_slice_scratch(module.n(), extension_factor, cols, brk.size());
    let (mut acc_add_dft, scratch_4) = scratch_3.take_vec_znx_dft_slice_scratch(module.n(), extension_factor, cols, brk.size());
    let (mut vmp_xai, mut scratch_5) = scratch_4.take_vec_znx_dft_scratch(module.n(), 1, brk.size());

    for acc_i in &mut acc {
        for col in 0..cols {
            module.vec_znx_zero(acc_i, col);
        }
    }

    let x_pow_a: &Vec<SvpPPolOwned<BE>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).as_usize()]; // TODO: from scratch space

    let two_n: usize = 2 * n_glwe;
    let two_n_ext: usize = 2 * lut.domain_size();

    module.blind_rotation_mod_switch(two_n_ext, &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b_pos: usize = ((lwe_2n[0] + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;

    let b_hi: usize = b_pos / extension_factor;
    let b_lo: usize = b_pos & (extension_factor - 1);

    for (i, j) in (0..b_lo).zip(extension_factor - b_lo..extension_factor) {
        let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                lut.data[j].data(),
            );
        module.vec_znx_rotate(b_hi as i64 + 1, &mut acc[i], 0, &lut_ref, 0);
    }
    for (i, j) in (b_lo..extension_factor).zip(0..extension_factor - b_lo) {
        let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                lut.data[j].data(),
            );
        module.vec_znx_rotate(b_hi as i64, &mut acc[i], 0, &lut_ref, 0);
    }

    let block_size: usize = brk.block_size();

    for (ai, ski) in izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)) {
        for i in 0..extension_factor {
            for j in 0..cols {
                let acc_ref = vec_znx_backend_ref_from_mut::<BE>(&acc[i]);
                module.vec_znx_dft_apply(1, 0, &mut acc_dft[i], j, &acc_ref, j);
                module.vec_znx_dft_zero(&mut acc_add_dft[i], j)
            }
        }

        // TODO: first & last iterations can be optimized
        for (aii, skii) in izip!(ai.iter(), ski.iter()) {
            let ai_pos: usize = ((aii + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;
            let ai_hi: usize = ai_pos / extension_factor;
            let ai_lo: usize = ai_pos & (extension_factor - 1);

            // vmp_res = DFT(acc) * BRK[i]
            for i in 0..extension_factor {
                let skii_ref = skii.data().to_backend_ref();
                scratch_5.scope(|mut scratch_local| {
                    module.vmp_apply_dft_to_dft(
                        &mut vmp_res[i],
                        &vec_znx_dft_backend_ref_from_mut::<BE>(&acc_dft[i]),
                        &skii_ref,
                        0,
                        &mut scratch_local,
                    );
                });
            }

            // Trivial case: no rotation between polynomials, we can directly multiply with (X^{-ai} - 1)
            if ai_lo == 0 {
                // Sets acc_add_dft[i] = (acc[i] * sk) * X^{-ai} - (acc[i] * sk)
                if ai_hi != 0 {
                    // DFT X^{-ai}
                    for j in 0..extension_factor {
                        for i in 0..cols {
                            let x_pow_a_ref = x_pow_a[ai_hi].to_backend_ref();
                            let vmp_res_j_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[j]);
                            {
                                let mut vmp_xai_backend = vmp_xai.to_backend_mut();
                                module.svp_apply_dft_to_dft(&mut vmp_xai_backend, 0, &x_pow_a_ref, 0, &vmp_res_j_ref, i);
                            }
                            let vmp_res_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[j]);
                            let vmp_xai_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_xai);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[j], i, &vmp_xai_ref, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[j], i, &vmp_res_ref, i);
                        }
                    }
                }

            // Non trivial case: rotation between polynomials
            // In this case we can't directly multiply with (X^{-ai} - 1) because of the
            // ring homomorphism R^{N} -> prod R^{N/extension_factor}, so we split the
            // computation in two steps: acc_add_dft = (acc * sk) * (-1) + (acc * sk) * X^{-ai}
            } else {
                // Sets acc_add_dft[0..ai_lo] += (acc[extension_factor - ai_lo..extension_factor] * sk) * X^{-ai+1}
                if (ai_hi + 1) & (two_n - 1) != 0 {
                    for (i, j) in (0..ai_lo).zip(extension_factor - ai_lo..extension_factor) {
                        for k in 0..cols {
                            let x_pow_a_ref = x_pow_a[ai_hi + 1].to_backend_ref();
                            let vmp_res_j_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[j]);
                            {
                                let mut vmp_xai_backend = vmp_xai.to_backend_mut();
                                module.svp_apply_dft_to_dft(&mut vmp_xai_backend, 0, &x_pow_a_ref, 0, &vmp_res_j_ref, k);
                            }
                            let vmp_xai_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_xai);
                            let vmp_res_i_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[i]);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[i], k, &vmp_xai_ref, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[i], k, &vmp_res_i_ref, k);
                        }
                    }
                }

                // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                if ai_hi != 0 {
                    // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                    for (i, j) in (ai_lo..extension_factor).zip(0..extension_factor - ai_lo) {
                        for k in 0..cols {
                            let x_pow_a_ref = x_pow_a[ai_hi].to_backend_ref();
                            let vmp_res_j_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[j]);
                            {
                                let mut vmp_xai_backend = vmp_xai.to_backend_mut();
                                module.svp_apply_dft_to_dft(&mut vmp_xai_backend, 0, &x_pow_a_ref, 0, &vmp_res_j_ref, k);
                            }
                            let vmp_xai_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_xai);
                            let vmp_res_i_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res[i]);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[i], k, &vmp_xai_ref, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[i], k, &vmp_res_i_ref, k);
                        }
                    }
                }
            }
        }

        scratch_5.scope(|scratch_local| {
            let (mut acc_add_big, mut scratch7) = scratch_local.take_vec_znx_big_scratch(module.n(), 1, brk.size());

            for j in 0..extension_factor {
                for i in 0..cols {
                    let acc_add_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&acc_add_dft[j]);
                    module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft_ref, i, &mut scratch7.borrow());
                    {
                        let acc_ref = vec_znx_backend_ref_from_mut::<BE>(&acc[j]);
                        module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &acc_ref, i);
                    }
                    let acc_add_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&acc_add_big);
                    module.vec_znx_big_normalize(
                        &mut acc[j],
                        base2k,
                        res_k,
                        0,
                        i,
                        &acc_add_big_ref,
                        base2k,
                        0,
                        &mut scratch7.borrow(),
                    );
                }
            }
        });
    }

    let mut res_mut = res.to_backend_mut();
    let acc_ref = vec_znx_backend_ref_from_mut::<BE>(&acc[0]);
    for i in 0..cols {
        module.vec_znx_copy(res_mut.data_mut(), i, &acc_ref, i);
    }
}

fn execute_block_binary<R, L, M, BE: Backend<ZnxWord = i64>, const PARALLEL: bool>(
    module: &M,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    M: BlindRotationModSwitch<BE>
        + VecZnxDftBytesOf
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxRotate<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + Sync,
{
    let n_glwe: usize = brk.n_glwe().into();
    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_tmp: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(res);
    let two_n: usize = n_glwe << 1;
    let base2k: usize = brk.base2k().into();
    let res_k = out_tmp.k().as_usize();
    let dnum: usize = brk.dnum().into();

    let cols: usize = (out_tmp.rank() + 1).into();

    module.blind_rotation_mod_switch(2 * lut.domain_size(), &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    module.glwe_zero(&mut out_tmp);

    // Initialize out to X^{b} * LUT(X)
    let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[0].data());
    {
        let mut out_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
        module.vec_znx_rotate(b, out_backend.data_mut(), 0, &lut_ref, 0);
    }

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let scratch = scratch.borrow();
    let (mut acc_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module.n(), cols, dnum);

    if PARALLEL {
        let (vmp_res, scratch_2) = scratch_1.take_vec_znx_dft_slice_scratch(module.n(), block_size, cols, brk.size());
        let (contributions, mut scratch_3) = scratch_2.take_vec_znx_dft_slice_scratch(module.n(), block_size, cols, brk.size());
        let mut tasks: Vec<_> = vmp_res.into_iter().zip(contributions).collect();
        let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(block_size, block_size);
        let worker_scratch_bytes = poulpy_hal::execution::worker_scratch_bytes::<BE>(module.vmp_apply_dft_to_dft_tmp_bytes(
            brk.size(),
            dnum,
            dnum,
            cols,
            cols,
            brk.size(),
        ));

        let x_pow_a = brk.x_pow_a.as_ref().expect("invalid key: x_pow_a has not been initialized");

        for (ai, ski) in izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)) {
            for j in 0..cols {
                let out_ref = <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                    out_tmp.data(),
                );
                module.vec_znx_dft_apply(1, 0, &mut acc_dft, j, &out_ref, j);
            }

            let (worker_scratch, _) = scratch_3.borrow().split(workers, worker_scratch_bytes);
            let acc_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&acc_dft);
            poulpy_hal::execution::for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(
                &mut tasks,
                0,
                worker_scratch,
                &|index, task, scratch| {
                    let (vmp_res, contribution) = task;
                    let skii_ref = ski[index].data().to_backend_ref();
                    module.vmp_apply_dft_to_dft(vmp_res, &acc_dft_ref, &skii_ref, 0, scratch);

                    let ai_pos = ((ai[index] + two_n as i64) & (two_n - 1) as i64) as usize;
                    let x_pow_a_ref = x_pow_a[ai_pos].to_backend_ref();
                    for col in 0..cols {
                        let vmp_res_ref = vec_znx_dft_backend_ref_from_mut::<BE>(vmp_res);
                        module.svp_apply_dft_to_dft(contribution, col, &x_pow_a_ref, 0, &vmp_res_ref, col);
                        let vmp_res_ref = vec_znx_dft_backend_ref_from_mut::<BE>(vmp_res);
                        module.vec_znx_dft_sub_assign(contribution, col, &vmp_res_ref, col);
                    }
                },
            );

            let (sum, rest) = tasks.split_first_mut().unwrap();
            for (_, contribution) in rest {
                for col in 0..cols {
                    let contribution_ref = vec_znx_dft_backend_ref_from_mut::<BE>(contribution);
                    module.vec_znx_dft_add_assign(&mut sum.1, col, &contribution_ref, col);
                }
            }

            let (mut acc_add_big, mut scratch_4) = scratch_3.borrow().take_vec_znx_big_scratch(module.n(), 1, brk.size());
            for col in 0..cols {
                let contribution_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&sum.1);
                module.vec_znx_idft_apply(&mut acc_add_big, 0, &contribution_ref, col, &mut scratch_4.borrow());
                {
                    let out_ref =
                        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                            out_tmp.data(),
                        );
                    module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &out_ref, col);
                }
                let acc_add_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&acc_add_big);
                let mut out_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
                module.vec_znx_big_normalize(
                    out_backend.data_mut(),
                    base2k,
                    res_k,
                    0,
                    col,
                    &acc_add_big_ref,
                    base2k,
                    0,
                    &mut scratch_4.borrow(),
                );
            }
        }

        module.glwe_copy(res, &out_tmp, &mut scratch_3);
        return;
    }

    let (mut vmp_res, scratch_2) = scratch_1.take_vec_znx_dft_scratch(module.n(), cols, brk.size());
    let (mut acc_add_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, brk.size());
    let (mut vmp_xai, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module.n(), 1, brk.size());

    let x_pow_a: &Vec<SvpPPolOwned<BE>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    for (ai, ski) in izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)) {
        for j in 0..cols {
            let out_ref = <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                out_tmp.data(),
            );
            module.vec_znx_dft_apply(1, 0, &mut acc_dft, j, &out_ref, j);
            module.vec_znx_dft_zero(&mut acc_add_dft, j)
        }

        for (aii, skii) in izip!(ai.iter(), ski.iter()) {
            let ai_pos: usize = ((aii + two_n as i64) & (two_n - 1) as i64) as usize;

            // vmp_res = DFT(acc) * BRK[i]
            let skii_ref = skii.data().to_backend_ref();
            module.vmp_apply_dft_to_dft(&mut vmp_res, &acc_dft.to_backend_ref(), &skii_ref, 0, &mut scratch_4.borrow());

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            for i in 0..cols {
                let x_pow_a_ref = x_pow_a[ai_pos].to_backend_ref();
                let vmp_res_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res);
                {
                    let mut vmp_xai_backend = vmp_xai.to_backend_mut();
                    module.svp_apply_dft_to_dft(&mut vmp_xai_backend, 0, &x_pow_a_ref, 0, &vmp_res_ref, i);
                }
                let vmp_res_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_res);
                let vmp_xai_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&vmp_xai);
                module.vec_znx_dft_add_assign(&mut acc_add_dft, i, &vmp_xai_ref, 0);
                module.vec_znx_dft_sub_assign(&mut acc_add_dft, i, &vmp_res_ref, i);
            }
        }

        {
            let (mut acc_add_big, mut scratch_5) = scratch_4.borrow().take_vec_znx_big_scratch(module.n(), 1, brk.size());

            for i in 0..cols {
                let acc_add_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&acc_add_dft);
                module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft_ref, i, &mut scratch_5.borrow());
                {
                    let out_ref =
                        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(
                            out_tmp.data(),
                        );
                    module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &out_ref, i);
                }
                let acc_add_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&acc_add_big);
                {
                    let mut out_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
                    module.vec_znx_big_normalize(
                        out_backend.data_mut(),
                        base2k,
                        res_k,
                        0,
                        i,
                        &acc_add_big_ref,
                        base2k,
                        0,
                        &mut scratch_5.borrow(),
                    );
                }
            }
        }
    }
    module.glwe_copy(res, &out_tmp, &mut scratch_4);
}

fn execute_standard<R, L, M, BE: Backend<ZnxWord = i64>>(
    module: &M,
    res: &mut R,
    lwe: &L,
    lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    M: BlindRotationModSwitch<BE>
        + VecZnxRotate<BE>
        + GLWEExternalProduct<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), brk.n(), "res.n(): {} != brk.n(): {}", res.n(), brk.n());
        assert_eq!(
            lut.domain_size(),
            brk.n_glwe().as_usize(),
            "lut.n(): {} != brk.n(): {}",
            lut.domain_size(),
            brk.n_glwe().as_usize()
        );
        assert_eq!(
            res.rank(),
            brk.rank(),
            "res.rank(): {} != brk.rank(): {}",
            res.rank(),
            brk.rank()
        );
        assert_eq!(
            lwe.n(),
            brk.n_lwe(),
            "lwe.n(): {} != brk.data.len(): {}",
            lwe.n(),
            brk.n_lwe()
        );
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_tmp: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(res);
    module.glwe_copy(&mut out_tmp, res, scratch);

    module.blind_rotation_mod_switch(2 * lut.domain_size(), &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    module.glwe_zero(&mut out_tmp);

    // Initialize out to X^{b} * LUT(X)
    let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[0].data());
    {
        let mut out_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
        module.vec_znx_rotate(b, out_backend.data_mut(), 0, &lut_ref, 0);
    }

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]
    let scratch = scratch.borrow();
    let (mut acc_tmp, mut scratch_1) = scratch.take_glwe_scratch(&out_tmp);

    // TODO: see if faster by skipping normalization in external product and keeping acc in big coeffs
    // TODO: first iteration can be optimized to be a gglwe product
    for (ai, ski) in izip!(a.iter(), brk.data.iter()) {
        // acc_tmp = sk[i] * acc
        {
            module.glwe_external_product(&mut acc_tmp, &out_tmp, &ski.to_backend_ref(), &mut scratch_1.borrow());
        }

        // acc_tmp = (sk[i] * acc) * (X^{ai} - 1)
        module.glwe_mul_xp_minus_one_assign(*ai, &mut acc_tmp, &mut scratch_1.borrow());

        // acc = acc + (sk[i] * acc) * (X^{ai} - 1)
        module.glwe_add_assign(&mut out_tmp, &acc_tmp);
    }

    // We can normalize only at the end because we add normalized values in [-2^{base2k-1}, 2^{base2k-1}]
    // on top of each others, thus ~ 2^{63-base2k} additions are supported before overflow.
    {
        module.glwe_normalize_assign(&mut out_tmp, &mut scratch_1.borrow());
    }
    module.glwe_copy(res, &out_tmp, &mut scratch_1);
}
