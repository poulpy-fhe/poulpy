use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGSWEncryptSk, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Dnum, Dsize, GGSW, GGSWInfos, GGSWLayout, GGSWPreparedFactory, GLWEInfos, GLWESecretPrepared,
        GLWESecretPreparedFactory, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssign},
    layouts::{Backend, DeviceBuf, Module, ScalarZnx, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};
use rand::Rng;

use crate::bin_fhe::{
    bdd_arithmetic::{
        FheUintPrepared, GGSWBlindRotation,
        tests::test_suite::{TEST_FHEUINT_BASE2K, TEST_RANK, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_scalar_to_ggsw_blind_rotation<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GGSWBlindRotation<u32, BE>
        + GGSWNoise<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<DeviceBuf<BE>, BE> = &test_context.sk_glwe;

    let base2k: Base2K = TEST_FHEUINT_BASE2K.into();
    let rank: Rank = TEST_RANK.into();
    let k_ggsw_res: TorusPrecision = TorusPrecision(39);
    let k_ggsw_apply: TorusPrecision = TorusPrecision(52);

    let ggsw_res_infos: GGSWLayout = GGSWLayout {
        n: module.n().into(),
        base2k,
        k: k_ggsw_res,
        rank,
        dnum: Dnum(2),
        dsize: Dsize(1),
    };

    let ggsw_k_infos: GGSWLayout = GGSWLayout {
        n: module.n().into(),
        base2k,
        k: k_ggsw_apply,
        rank,
        dnum: Dnum(3),
        dsize: Dsize(1),
    };

    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);
    let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_res_infos);

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), 1);
    scalar.raw_mut().iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let k: u32 = source.next_u32();

    let ggsw_k_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_k_infos).unwrap();

    let mut k_enc_prep: FheUintPrepared<DeviceBuf<BE>, u32, BE> =
        FheUintPrepared::<DeviceBuf<BE>, u32, BE>::alloc_from_infos(module, &ggsw_k_infos);
    k_enc_prep.encrypt_sk(
        module,
        k,
        sk_glwe_prep,
        &ggsw_k_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let base: [usize; 2] = [module.log_n() >> 1, module.log_n() - (module.log_n() >> 1)];

    assert_eq!(base.iter().sum::<usize>(), module.log_n());

    // Starting bit
    let mut bit_start: usize = 0;

    let max_noise = |col_i: usize| {
        let mut noise: f64 = -(ggsw_res_infos.size() as f64 * base2k.as_usize() as f64) + DEFAULT_SIGMA_XE.log2() + 3.0;
        noise += 0.5 * ggsw_res_infos.log_n() as f64;
        if col_i != 0 {
            noise += 0.5 * ggsw_res_infos.log_n() as f64
        }
        noise
    };

    for _ in 0..32_usize.div_ceil(module.log_n()) {
        // By how many bits to left shift
        let mut bit_step: usize = 0;

        for digit in base {
            let mask: u32 = (1 << digit) - 1;

            // How many bits to take
            let bit_size: usize = (32 - bit_start).min(digit);

            module.scalar_to_ggsw_blind_rotation(
                &mut res,
                &scalar,
                &k_enc_prep,
                false,
                bit_start,
                bit_size,
                bit_step,
                scratch.borrow(),
            );

            let rot: i64 = (((k >> bit_start) & mask) << bit_step) as i64;

            let mut scalar_want: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), 1);
            scalar_want.raw_mut().copy_from_slice(scalar.raw());

            module.vec_znx_rotate_assign(-rot, &mut scalar_want.as_vec_znx_mut(), 0, scratch.borrow());

            for row in 0..res.dnum().as_usize() {
                for col in 0..res.rank().as_usize() + 1 {
                    assert!(
                        res.noise(module, row, col, &scalar_want, sk_glwe_prep, scratch.borrow())
                            .std()
                            .log2()
                            <= max_noise(col)
                    )
                }
            }

            bit_step += digit;
            bit_start += digit;

            if bit_start >= 32 {
                break;
            }
        }
    }
}
