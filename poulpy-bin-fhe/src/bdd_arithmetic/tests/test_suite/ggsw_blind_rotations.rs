use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGSWEncryptSk, GGSWNoise, GLWEDecrypt, GLWEEncryptSk,
    layouts::{
        Base2K, Dnum, Dsize, GGSW, GGSWInfos, GGSWLayout, GGSWPreparedFactory, GLWEInfos, GLWESecretPrepared,
        GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssignBackend},
    layouts::{
        Backend, HostBackend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, ZnxView, ZnxViewMut,
    },
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        FheUintPrepared, GGSWBlindRotation,
        tests::test_suite::{TEST_FHEUINT_BASE2K, TEST_RANK, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_scalar_to_ggsw_blind_rotation<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GGSWBlindRotation<u32, BE>
        + GGSWNoise<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssignBackend<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

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
    let mut res: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_res_infos);

    let mut scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    scalar.raw_mut().iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let k: u32 = source.next_u32();

    let ggsw_k_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_k_infos).unwrap();

    let mut k_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_k_infos);
    k_enc_prep.encrypt_sk(
        module,
        k,
        sk_glwe_prep,
        &ggsw_k_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
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
                &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&scalar),
                &k_enc_prep,
                false,
                bit_start,
                bit_size,
                bit_step,
                &mut scratch.borrow(),
            );

            let rot: i64 = (((k >> bit_start) & mask) << bit_step) as i64;

            let mut scalar_want: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
            scalar_want.raw_mut().copy_from_slice(scalar.raw());

            let mut scalar_want_vec: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, 1);
            scalar_want_vec.raw_mut().copy_from_slice(scalar_want.raw());
            {
                let mut scalar_want_backend = <VecZnx<Vec<u8>> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut scalar_want_vec);
                module.vec_znx_rotate_assign_backend(-rot, &mut scalar_want_backend, 0, &mut scratch.borrow());
            }
            scalar_want.raw_mut().copy_from_slice(scalar_want_vec.raw());

            for row in 0..res.dnum().as_usize() {
                for col in 0..res.rank().as_usize() + 1 {
                    assert!(
                        res.noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&scalar_want),
                            sk_glwe_prep,
                            &mut scratch.borrow(),
                        )
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
