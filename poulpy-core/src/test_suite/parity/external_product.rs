//! External-product parity: the other consumer of the gadget digit loop.

use super::poisoned_scratch;
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes,
    },
    layouts::{FillUniform, HostDataMut, Module, ScratchOwned, VecZnx, VecZnxDftToBackendMut, VecZnxToBackendMut},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::prepared::GGSWPreparedToBackendRef;
use crate::{
    GLWEExternalProduct, GLWEExternalProductInternal,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GGSWPreparedFactory,
    },
    test_suite::parity::{ParityBackend, ParityShapes, ref_glwe},
};

/// `glwe_external_product` agrees with the reference backend byte-for-byte.
pub fn test_glwe_external_product_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEExternalProduct<BR>
        + GGSWPreparedFactory<BR>
        + GLWEExternalProductInternal<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxIdftNormalizeConsume<BR>
        + VecZnxIdftNormalizeConsumeTmpBytes,
    Module<BT>: GLWEExternalProduct<BT>
        + GGSWPreparedFactory<BT>
        + GLWEExternalProductInternal<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxIdftNormalizeConsume<BT>
        + VecZnxIdftNormalizeConsumeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([41u8; 32]);

    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(k, base2k) {
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

            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);
            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);

            let mut scratch_ref = poisoned_scratch::<BR>(module_ref.ggsw_prepare_tmp_bytes(&ggsw_infos));
            let mut scratch_test = poisoned_scratch::<BT>(module_test.ggsw_prepare_tmp_bytes(&ggsw_infos));

            let mut ggsw_ref = module_ref.ggsw_prepared_alloc_from_infos(&ggsw_infos);
            module_ref.ggsw_prepare(&mut ggsw_ref, &ggsw_ref_coeffs, &mut scratch_ref.borrow());
            let mut ggsw_test = module_test.ggsw_prepared_alloc_from_infos(&ggsw_infos);
            module_test.ggsw_prepare(&mut ggsw_test, &ggsw_test_coeffs, &mut scratch_test.borrow());

            let mut scratch_ref =
                poisoned_scratch::<BR>(module_ref.glwe_external_product_tmp_bytes(&res_infos, &a_infos, &ggsw_infos));
            let mut scratch_test =
                poisoned_scratch::<BT>(module_test.glwe_external_product_tmp_bytes(&res_infos, &a_infos, &ggsw_infos));

            module_ref.glwe_external_product(&mut res_ref, &a_ref, &ggsw_ref.to_backend_ref(), &mut scratch_ref.borrow());
            module_test.glwe_external_product(
                &mut res_test,
                &a_test,
                &ggsw_test.to_backend_ref(),
                &mut scratch_test.borrow(),
            );

            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "glwe_external_product: rank={rank} dsize={dsize} k={k}");

            // Observe the internal DFT contract through canonical integer output,
            // never through backend-specific transform bytes.
            macro_rules! internal {
                ($be:ty, $module:ident, $a:ident, $key:ident) => {{
                    let size=crate::reference::external_product::glwe::glwe_external_product_output_size::<$be,_,_,_>(&res_infos,&a_infos,&ggsw_infos);
                    let mut dft=$module.vec_znx_dft_alloc(n as usize,rank+1,size);
                    let bytes=<$be as poulpy_hal::layouts::Backend>::len_bytes(&dft.data);
                    <$be as poulpy_hal::layouts::Backend>::copy_from_host(&mut dft.data,&vec![0x55;bytes]);
                    $module.glwe_external_product_dft(&mut dft.to_backend_mut(),&$a,&$key.to_backend_ref(),&mut poisoned_scratch::<$be>($module.glwe_external_product_internal_tmp_bytes(&res_infos,&a_infos,&ggsw_infos)).borrow());
                    let mut normalized=$module.glwe_alloc_from_infos(&res_infos);
                    for col in 0..rank+1 {
                        let tmp=$module.vec_znx_idft_normalize_consume_tmp_bytes(normalized.data.size(),size);
                        $module.vec_znx_idft_normalize_consume(&mut <VecZnx<<$be as poulpy_hal::layouts::Backend>::OwnedBuf,i64> as VecZnxToBackendMut<$be>>::to_backend_mut(&mut normalized.data),base2k,res_infos.k.as_usize(),col,&mut dft.to_backend_mut(),col,base2k,None,&mut poisoned_scratch::<$be>(tmp).borrow());
                    }
                    normalized
                }};
            }
            let internal_ref = internal!(BR, module_ref, a_ref, ggsw_ref);
            let internal_test = internal!(BT, module_test, a_test, ggsw_test);
            internal_test.transfer_into(&mut have);
            assert_eq!(internal_ref, have, "external product DFT canonical parity");
            assert_eq!(internal_ref, res_ref, "external product DFT disagrees with public operation");

            // The assign query is measured independently at the operand's own width.
            let mut assigned_ref = module_ref.glwe_alloc_from_infos(&a_infos);
            let mut assigned_test = module_test.glwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut assigned_ref);
            a_ref.transfer_into(&mut assigned_test);
            let mut scratch_ref =
                poisoned_scratch::<BR>(module_ref.glwe_external_product_tmp_bytes(&a_infos, &a_infos, &ggsw_infos));
            let mut scratch_test =
                poisoned_scratch::<BT>(module_test.glwe_external_product_tmp_bytes(&a_infos, &a_infos, &ggsw_infos));
            module_ref.glwe_external_product_assign(&mut assigned_ref, &ggsw_ref.to_backend_ref(), &mut scratch_ref.borrow());
            module_test.glwe_external_product_assign(&mut assigned_test, &ggsw_test.to_backend_ref(), &mut scratch_test.borrow());
            let mut have = module_ref.glwe_alloc_from_infos(&a_infos);
            assigned_test.transfer_into(&mut have);
            assert_eq!(assigned_ref, have, "glwe_external_product_assign: rank={rank} dsize={dsize}");
        }
    }
}
