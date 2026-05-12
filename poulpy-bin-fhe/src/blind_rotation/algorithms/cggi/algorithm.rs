use itertools::izip;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDft, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftSubAssign, VecZnxDftZero,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxRotateBackend, VecZnxZeroBackend, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScratchArena, SvpPPolOwned, SvpPPolToBackendRef, VecZnx,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendRef, VmpPMatToBackendRef, ZnxView, ZnxViewMut, ZnxZero,
        vec_znx_backend_ref_from_mut, vec_znx_big_backend_ref_from_mut, vec_znx_dft_backend_ref_from_mut,
    },
    oep::HalVecZnxImpl,
};

use poulpy_core::{
    Distribution, GLWEAdd, GLWECopy, GLWEExternalProduct, GLWEMulXpMinusOne, GLWENormalize, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToBackendMut, LWE, LWEInfos, LWEToBackendRef, ModuleCoreAlloc},
};

use crate::blind_rotation::{
    BlindRotationExecute, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, LookupTable, mod_switch_2n,
};

impl<BE: Backend<OwnedBuf = Vec<u8>> + 'static> BlindRotationExecute<CGGI, BE> for Module<BE>
where
    Self: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotateBackend<BE>
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
        + GLWENormalize<BE>
        + VecZnxZeroBackend<BE>,
    // TODO(device): CGGI blind rotation still contains host-visible sub-steps
    // (notably LUT/accumulator staging and coefficient-level glue). Keep the
    // public blind-rotation API backend-generic, but leave this implementation
    // host-backed until those helpers are migrated.
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::OwnedBuf: HostDataRef,
    BE: HalVecZnxImpl<BE>,
{
    fn blind_rotation_execute_tmp_bytes<G, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: GGSWInfos,
    {
        let brk_size: usize = brk_infos.size();

        if block_size > 1 {
            let cols: usize = (brk_infos.rank() + 1).into();
            let dnum: usize = brk_infos.dnum().into();
            let acc_dft: usize = self.bytes_of_vec_znx_dft(cols, dnum) * extension_factor;
            let acc_big: usize = self.bytes_of_vec_znx_big(1, brk_size);
            let vmp_res: usize = self.bytes_of_vec_znx_dft(cols, brk_size) * extension_factor;
            let vmp_xai: usize = self.bytes_of_vec_znx_dft(1, brk_size);
            let acc_dft_add: usize = vmp_res;
            let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(brk_size, dnum, dnum, 2, 2, brk_size); // GGSW product: (1 x 2) x (2 x 2)
            let acc: usize = if extension_factor > 1 {
                VecZnx::bytes_of(self.n(), cols, glwe_infos.size()) * extension_factor
            } else {
                0
            };

            acc + acc_dft
                + acc_dft_add
                + vmp_res
                + vmp_xai
                + (vmp
                    | (acc_big
                        + (self
                            .vec_znx_big_normalize_tmp_bytes()
                            .max(self.vec_znx_idft_apply_tmp_bytes()))))
        } else {
            GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos)
                + self.glwe_external_product_tmp_bytes(glwe_infos, glwe_infos, brk_infos)
        }
    }

    fn blind_rotation_execute<'s, R, DL>(
        &self,
        res: &mut R,
        lwe: &LWE<DL>,
        lut: &LookupTable<BE::OwnedBuf>,
        brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        DL: Data,
        LWE<DL>: LWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        // TODO(device): make the full execute path 100% backend-native. The
        // current implementation still relies on host-visible scratch/result
        // views for parts of CGGI that do not yet have backend-native
        // add/copy helpers.
        match brk.dist {
            Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {
                if lut.extension_factor() > 1 {
                    assert!(
                        matches!(brk.dist, Distribution::BinaryBlock(_)),
                        "extended blind rotation (extension_factor={}) requires a BinaryBlock key distribution, got {:?}",
                        lut.extension_factor(),
                        brk.dist,
                    );
                    execute_block_binary_extended(self, res, lwe, lut, brk, scratch)
                } else if brk.block_size() > 1 {
                    execute_block_binary(self, res, lwe, lut, brk, scratch);
                } else {
                    execute_standard(self, res, lwe, lut, brk, scratch);
                }
            }
            _ => panic!("invalid CGGI distribution (have you prepared the key?)"),
        }
    }
}

fn execute_block_binary_extended<R, DataIn, M, BE: Backend<OwnedBuf = Vec<u8>> + 'static>(
    module: &M,
    res: &mut R,
    lwe: &LWE<DataIn>,
    lut: &LookupTable<BE::OwnedBuf>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    DataIn: Data,
    LWE<DataIn>: LWEToBackendRef<BE>,
    M: VecZnxDftBytesOf
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + VecZnxRotateBackend<BE>
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
        + VecZnxBigBytesOf,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    // TODO(device): this extended block-binary path still performs host-side
    // coefficient orchestration over temporary accumulators.
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::OwnedBuf: HostDataRef,
{
    let n_glwe: usize = brk.n_glwe().into();
    let extension_factor: usize = lut.extension_factor();
    let base2k: usize = res.base2k().into();
    let dnum: usize = brk.dnum().into();
    let cols: usize = (res.rank() + 1).into();

    let scratch = scratch.borrow();
    let (mut acc, scratch_1) = scratch.take_vec_znx_slice_scratch(extension_factor, n_glwe, cols, res.size());
    let (mut acc_dft, scratch_2) = scratch_1.take_vec_znx_dft_slice_scratch(module, extension_factor, cols, dnum);
    let (mut vmp_res, scratch_3) = scratch_2.take_vec_znx_dft_slice_scratch(module, extension_factor, cols, brk.size());
    let (mut acc_add_dft, scratch_4) = scratch_3.take_vec_znx_dft_slice_scratch(module, extension_factor, cols, brk.size());
    let (mut vmp_xai, mut scratch_5) = scratch_4.take_vec_znx_dft_scratch(module, 1, brk.size());

    (0..extension_factor).for_each(|i| {
        acc[i].zero();
    });

    let x_pow_a: &Vec<SvpPPolOwned<BE>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).as_usize()]; // TODO: from scratch space

    let two_n: usize = 2 * n_glwe;
    let two_n_ext: usize = 2 * lut.domain_size();

    mod_switch_2n::<BE, _>(two_n_ext, &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b_pos: usize = ((lwe_2n[0] + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;

    let b_hi: usize = b_pos / extension_factor;
    let b_lo: usize = b_pos & (extension_factor - 1);

    for (i, j) in (0..b_lo).zip(extension_factor - b_lo..extension_factor) {
        let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[j].data());
        module.vec_znx_rotate_backend(b_hi as i64 + 1, &mut acc[i], 0, &lut_ref, 0);
    }
    for (i, j) in (b_lo..extension_factor).zip(0..extension_factor - b_lo) {
        let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[j].data());
        module.vec_znx_rotate_backend(b_hi as i64, &mut acc[i], 0, &lut_ref, 0);
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
            let (mut acc_add_big, mut scratch7) = scratch_local.take_vec_znx_big_scratch(module, 1, brk.size());

            for j in 0..extension_factor {
                for i in 0..cols {
                    let acc_add_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&acc_add_dft[j]);
                    module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft_ref, i, &mut scratch7.borrow());
                    {
                        let acc_ref = vec_znx_backend_ref_from_mut::<BE>(&acc[j]);
                        module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &acc_ref, i);
                    }
                    let acc_add_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&acc_add_big);
                    module.vec_znx_big_normalize(&mut acc[j], base2k, 0, i, &acc_add_big_ref, base2k, 0, &mut scratch7.borrow());
                }
            }
        });
    }

    let mut res_mut = res.to_backend_mut();
    for i in 0..cols {
        let res_data = res_mut.data_mut();
        let min_size = res_data.size().min(acc[0].size());
        for j in 0..min_size {
            res_data.at_mut(i, j).copy_from_slice(acc[0].at(i, j));
        }
        for j in min_size..res_data.size() {
            res_data.at_mut(i, j).fill(0);
        }
    }
}

fn execute_block_binary<R, DataIn, M, BE: Backend<OwnedBuf = Vec<u8>> + 'static>(
    module: &M,
    res: &mut R,
    lwe: &LWE<DataIn>,
    lut: &LookupTable<BE::OwnedBuf>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    DataIn: Data,
    LWE<DataIn>: LWEToBackendRef<BE>,
    M: VecZnxDftBytesOf
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + VecZnxRotateBackend<BE>
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
        + VecZnxBigBytesOf,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    // TODO(device): this block-binary path still assumes host-visible
    // temporary accumulators and LUT buffers.
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::OwnedBuf: HostDataRef,
{
    let n_glwe: usize = brk.n_glwe().into();
    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_tmp: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(res);
    let two_n: usize = n_glwe << 1;
    let base2k: usize = brk.base2k().into();
    let dnum: usize = brk.dnum().into();

    let cols: usize = (out_tmp.rank() + 1).into();

    mod_switch_2n::<BE, _>(2 * lut.domain_size(), &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_tmp.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[0].data());
    {
        let mut out_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
        module.vec_znx_rotate_backend(b, out_backend.data_mut(), 0, &lut_ref, 0);
    }

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let scratch = scratch.borrow();
    let (mut acc_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, cols, dnum);
    let (mut vmp_res, scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, cols, brk.size());
    let (mut acc_add_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module, cols, brk.size());
    let (mut vmp_xai, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module, 1, brk.size());

    let x_pow_a: &Vec<SvpPPolOwned<BE>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    for (ai, ski) in izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)) {
        for j in 0..cols {
            let out_ref = <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(out_tmp.data());
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
            let (mut acc_add_big, mut scratch_5) = scratch_4.borrow().take_vec_znx_big_scratch(module, 1, brk.size());

            for i in 0..cols {
                let acc_add_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&acc_add_dft);
                module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft_ref, i, &mut scratch_5.borrow());
                {
                    let out_ref =
                        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(out_tmp.data());
                    module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &out_ref, i);
                }
                let acc_add_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&acc_add_big);
                {
                    let mut out_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
                    module.vec_znx_big_normalize(
                        out_backend.data_mut(),
                        base2k,
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
    module.glwe_copy(res, &out_tmp);
}

fn execute_standard<R, DataIn, M, BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &M,
    res: &mut R,
    lwe: &LWE<DataIn>,
    lut: &LookupTable<BE::OwnedBuf>,
    brk: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    DataIn: Data,
    LWE<DataIn>: LWEToBackendRef<BE>,
    M: VecZnxRotateBackend<BE>
        + GLWEExternalProduct<BE>
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    // TODO(device): the standard CGGI path still uses host-visible
    // coefficient staging for the accumulator.
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::OwnedBuf: HostDataRef,
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
    let mut out_tmp: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(res);
    module.glwe_copy(&mut out_tmp, res);

    mod_switch_2n::<BE, _>(2 * lut.domain_size(), &mut lwe_2n, lwe, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_tmp.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    let lut_ref: poulpy_hal::layouts::VecZnxBackendRef<'_, BE> =
        <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(lut.data[0].data());
    {
        let mut out_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut out_tmp);
        module.vec_znx_rotate_backend(b, out_backend.data_mut(), 0, &lut_ref, 0);
    }

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]
    let scratch = scratch.borrow();
    let (mut acc_tmp, mut scratch_1) = scratch.take_glwe_scratch(&out_tmp);

    // TODO: see if faster by skipping normalization in external product and keeping acc in big coeffs
    // TODO: first iteration can be optimized to be a gglwe product
    for (ai, ski) in izip!(a.iter(), brk.data.iter()) {
        // acc_tmp = sk[i] * acc
        {
            module.glwe_external_product(&mut acc_tmp, &out_tmp, ski, ski.size(), &mut scratch_1.borrow());
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
    module.glwe_copy(res, &out_tmp);
}
