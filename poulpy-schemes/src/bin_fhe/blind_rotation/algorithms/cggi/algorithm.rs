use itertools::izip;
use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftSubAssign,
        VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxRotate, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, SvpPPolOwned, VecZnx, ZnxZero},
};

use poulpy_core::{
    Distribution, GLWEAdd, GLWEExternalProduct, GLWEMulXpMinusOne, GLWENormalize, ScratchTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToMut, LWE, LWEInfos, LWEToRef},
};

use crate::bin_fhe::blind_rotation::{
    BlindRotationExecute, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, LookupTable, mod_switch_2n,
};

impl<BE: Backend> BlindRotationExecute<CGGI, BE> for Module<BE>
where
    Self: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEExternalProduct<BE>
        + ModuleN
        + VecZnxRotate
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy
        + GLWEMulXpMinusOne<BE>
        + GLWEAdd
        + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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

    fn blind_rotation_execute<DR, DL, DB>(
        &self,
        res: &mut GLWE<DR>,
        lwe: &LWE<DL>,
        lut: &LookupTable,
        brk: &BlindRotationKeyPrepared<DB, CGGI, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DR: DataMut,
        DL: DataRef,
        DB: DataRef,
    {
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

fn execute_block_binary_extended<DataRes, DataIn, DataBrk, M, BE: Backend>(
    module: &M,
    res: &mut GLWE<DataRes>,
    lwe: &LWE<DataIn>,
    lut: &LookupTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, BE>,
    scratch: &mut Scratch<BE>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    M: VecZnxDftBytesOf
        + ModuleN
        + VecZnxRotate
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy
        + VecZnxBigBytesOf,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n_glwe: usize = brk.n_glwe().into();
    let extension_factor: usize = lut.extension_factor();
    let base2k: usize = res.base2k().into();
    let dnum: usize = brk.dnum().into();
    let cols: usize = (res.rank() + 1).into();

    let (mut acc, scratch_1) = scratch.take_vec_znx_slice(extension_factor, n_glwe, cols, res.size());
    let (mut acc_dft, scratch_2) = scratch_1.take_vec_znx_dft_slice(module, extension_factor, cols, dnum);
    let (mut vmp_res, scratch_3) = scratch_2.take_vec_znx_dft_slice(module, extension_factor, cols, brk.size());
    let (mut acc_add_dft, scratch_4) = scratch_3.take_vec_znx_dft_slice(module, extension_factor, cols, brk.size());
    let (mut vmp_xai, scratch_5) = scratch_4.take_vec_znx_dft(module, 1, brk.size());

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
    let lwe_ref: LWE<&[u8]> = lwe.to_ref();

    let two_n: usize = 2 * n_glwe;
    let two_n_ext: usize = 2 * lut.domain_size();

    mod_switch_2n(two_n_ext, &mut lwe_2n, &lwe_ref, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b_pos: usize = ((lwe_2n[0] + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;

    let b_hi: usize = b_pos / extension_factor;
    let b_lo: usize = b_pos & (extension_factor - 1);

    for (i, j) in (0..b_lo).zip(extension_factor - b_lo..extension_factor) {
        module.vec_znx_rotate(b_hi as i64 + 1, &mut acc[i], 0, &lut.data[j], 0);
    }
    for (i, j) in (b_lo..extension_factor).zip(0..extension_factor - b_lo) {
        module.vec_znx_rotate(b_hi as i64, &mut acc[i], 0, &lut.data[j], 0);
    }

    let block_size: usize = brk.block_size();

    izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)).for_each(|(ai, ski)| {
        for i in 0..extension_factor {
            for j in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut acc_dft[i], j, &acc[i], j);
                module.vec_znx_dft_zero(&mut acc_add_dft[i], j)
            }
        }

        // TODO: first & last iterations can be optimized
        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;
            let ai_hi: usize = ai_pos / extension_factor;
            let ai_lo: usize = ai_pos & (extension_factor - 1);

            // vmp_res = DFT(acc) * BRK[i]
            (0..extension_factor).for_each(|i| {
                module.vmp_apply_dft_to_dft(&mut vmp_res[i], &acc_dft[i], skii.data(), 0, scratch_5);
            });

            // Trivial case: no rotation between polynomials, we can directly multiply with (X^{-ai} - 1)
            if ai_lo == 0 {
                // Sets acc_add_dft[i] = (acc[i] * sk) * X^{-ai} - (acc[i] * sk)
                if ai_hi != 0 {
                    // DFT X^{-ai}
                    (0..extension_factor).for_each(|j| {
                        (0..cols).for_each(|i| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], i);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[j], i, &vmp_xai, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[j], i, &vmp_res[j], i);
                        });
                    });
                }

            // Non trivial case: rotation between polynomials
            // In this case we can't directly multiply with (X^{-ai} - 1) because of the
            // ring homomorphism R^{N} -> prod R^{N/extension_factor}, so we split the
            // computation in two steps: acc_add_dft = (acc * sk) * (-1) + (acc * sk) * X^{-ai}
            } else {
                // Sets acc_add_dft[0..ai_lo] += (acc[extension_factor - ai_lo..extension_factor] * sk) * X^{-ai+1}
                if (ai_hi + 1) & (two_n - 1) != 0 {
                    for (i, j) in (0..ai_lo).zip(extension_factor - ai_lo..extension_factor) {
                        (0..cols).for_each(|k| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi + 1], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }

                // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                if ai_hi != 0 {
                    // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                    for (i, j) in (ai_lo..extension_factor).zip(0..extension_factor - ai_lo) {
                        (0..cols).for_each(|k| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_assign(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_assign(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }
            }
        });

        {
            let (mut acc_add_big, scratch7) = scratch_5.take_vec_znx_big(module, 1, brk.size());

            (0..extension_factor).for_each(|j| {
                (0..cols).for_each(|i| {
                    module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft[j], i, scratch7);
                    module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, &acc[j], i);
                    module.vec_znx_big_normalize(&mut acc[j], base2k, 0, i, &acc_add_big, base2k, 0, scratch7);
                });
            });
        }
    });

    (0..cols).for_each(|i| {
        module.vec_znx_copy(res.data_mut(), i, &acc[0], i);
    });
}

fn execute_block_binary<DataRes, DataIn, DataBrk, M, BE: Backend>(
    module: &M,
    res: &mut GLWE<DataRes>,
    lwe: &LWE<DataIn>,
    lut: &LookupTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, BE>,
    scratch: &mut Scratch<BE>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    M: VecZnxDftBytesOf
        + ModuleN
        + VecZnxRotate
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftSubAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy
        + VecZnxBigBytesOf,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n_glwe: usize = brk.n_glwe().into();
    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_mut: GLWE<&mut [u8]> = res.to_mut();
    let lwe_ref: LWE<&[u8]> = lwe.to_ref();
    let two_n: usize = n_glwe << 1;
    let base2k: usize = brk.base2k().into();
    let dnum: usize = brk.dnum().into();

    let cols: usize = (out_mut.rank() + 1).into();

    mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe_ref, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, out_mut.data_mut(), 0, &lut.data[0], 0);

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let (mut acc_dft, scratch_1) = scratch.take_vec_znx_dft(module, cols, dnum);
    let (mut vmp_res, scratch_2) = scratch_1.take_vec_znx_dft(module, cols, brk.size());
    let (mut acc_add_dft, scratch_3) = scratch_2.take_vec_znx_dft(module, cols, brk.size());
    let (mut vmp_xai, scratch_4) = scratch_3.take_vec_znx_dft(module, 1, brk.size());

    let x_pow_a: &Vec<SvpPPolOwned<BE>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    izip!(a.chunks_exact(block_size), brk.data.chunks_exact(block_size)).for_each(|(ai, ski)| {
        for j in 0..cols {
            module.vec_znx_dft_apply(1, 0, &mut acc_dft, j, out_mut.data_mut(), j);
            module.vec_znx_dft_zero(&mut acc_add_dft, j)
        }

        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n as i64) & (two_n - 1) as i64) as usize;

            // vmp_res = DFT(acc) * BRK[i]
            module.vmp_apply_dft_to_dft(&mut vmp_res, &acc_dft, skii.data(), 0, scratch_4);

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            (0..cols).for_each(|i| {
                module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_pos], 0, &vmp_res, i);
                module.vec_znx_dft_add_assign(&mut acc_add_dft, i, &vmp_xai, 0);
                module.vec_znx_dft_sub_assign(&mut acc_add_dft, i, &vmp_res, i);
            });
        });

        {
            let (mut acc_add_big, scratch_5) = scratch_4.take_vec_znx_big(module, 1, brk.size());

            (0..cols).for_each(|i| {
                module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft, i, scratch_5);
                module.vec_znx_big_add_small_assign(&mut acc_add_big, 0, out_mut.data_mut(), i);
                module.vec_znx_big_normalize(out_mut.data_mut(), base2k, 0, i, &acc_add_big, base2k, 0, scratch_5);
            });
        }
    });
}

fn execute_standard<DataRes, DataIn, DataBrk, M, BE: Backend>(
    module: &M,
    res: &mut GLWE<DataRes>,
    lwe: &LWE<DataIn>,
    lut: &LookupTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, BE>,
    scratch: &mut Scratch<BE>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    M: VecZnxRotate + GLWEExternalProduct<BE> + GLWEMulXpMinusOne<BE> + GLWEAdd + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
    let mut out_mut: GLWE<&mut [u8]> = res.to_mut();
    let lwe_ref: LWE<&[u8]> = lwe.to_ref();

    mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe_ref, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, out_mut.data_mut(), 0, &lut.data[0], 0);

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]
    let (mut acc_tmp, scratch_1) = scratch.take_glwe(&out_mut);

    // TODO: see if faster by skipping normalization in external product and keeping acc in big coeffs
    // TODO: first iteration can be optimized to be a gglwe product
    izip!(a.iter(), brk.data.iter()).for_each(|(ai, ski)| {
        // acc_tmp = sk[i] * acc
        module.glwe_external_product(&mut acc_tmp, &out_mut, ski, scratch_1);

        // acc_tmp = (sk[i] * acc) * (X^{ai} - 1)
        module.glwe_mul_xp_minus_one_assign(*ai, &mut acc_tmp, scratch_1);

        // acc = acc + (sk[i] * acc) * (X^{ai} - 1)
        module.glwe_add_assign(&mut out_mut, &acc_tmp);
    });

    // We can normalize only at the end because we add normalized values in [-2^{base2k-1}, 2^{base2k-1}]
    // on top of each others, thus ~ 2^{63-base2k} additions are supported before overflow.
    module.glwe_normalize_assign(&mut out_mut, scratch_1);
}
