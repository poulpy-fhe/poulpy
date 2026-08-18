//! External-product parity: the other consumer of the gadget digit loop.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{FillUniform, HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GLWEExternalProduct,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GGSWPreparedFactory,
    },
    test_suite::parity::{ParityBackend, ref_glwe},
};

/// `glwe_external_product` agrees with the reference backend byte-for-byte.
pub fn test_glwe_external_product_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEExternalProduct<BR> + GGSWPreparedFactory<BR>,
    Module<BT>: GLWEExternalProduct<BT> + GGSWPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([41u8; 32]);

    for rank in 1..3usize {
        for dsize in 1..=k.div_ceil(base2k) {
            let a_infos = GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
            };
            let res_infos = GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision((k + base2k * dsize) as u32),
                rank: Rank(rank as u32),
            };
            let ggsw_infos = GGSWLayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                dnum: Dnum(k.div_ceil(base2k * dsize) as u32),
                dsize: Dsize(dsize as u32),
                k_aux: TorusPrecision((dsize * base2k) as u32),
                rank: Rank(rank as u32),
            };

            let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
            let mut ggsw_ref_coeffs = module_ref.ggsw_alloc_from_infos(&ggsw_infos);
            ggsw_ref_coeffs.fill_uniform(base2k, &mut source);

            let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut a_test);
            let mut ggsw_test_coeffs = module_test.ggsw_alloc_from_infos(&ggsw_infos);
            ggsw_ref_coeffs.transfer_into(&mut ggsw_test_coeffs);

            let mut res_ref = module_ref.glwe_alloc_from_infos(&res_infos);
            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                module_ref
                    .ggsw_prepare_tmp_bytes(&ggsw_infos)
                    .max(module_ref.glwe_external_product_tmp_bytes(&res_infos, &a_infos, &ggsw_infos)),
            );
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                module_test
                    .ggsw_prepare_tmp_bytes(&ggsw_infos)
                    .max(module_test.glwe_external_product_tmp_bytes(&res_infos, &a_infos, &ggsw_infos)),
            );

            let mut ggsw_ref = module_ref.ggsw_prepared_alloc_from_infos(&ggsw_infos);
            module_ref.ggsw_prepare(&mut ggsw_ref, &ggsw_ref_coeffs, &mut scratch_ref.borrow());
            let mut ggsw_test = module_test.ggsw_prepared_alloc_from_infos(&ggsw_infos);
            module_test.ggsw_prepare(&mut ggsw_test, &ggsw_test_coeffs, &mut scratch_test.borrow());

            module_ref.glwe_external_product(&mut res_ref, &a_ref, &ggsw_ref, &mut scratch_ref.borrow());
            module_test.glwe_external_product(&mut res_test, &a_test, &ggsw_test, &mut scratch_test.borrow());

            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "glwe_external_product: rank={rank} dsize={dsize} k={k}");
        }
    }
}
