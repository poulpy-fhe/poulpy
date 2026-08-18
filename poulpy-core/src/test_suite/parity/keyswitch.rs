//! Key-switch parity: the gadget digit loop, compared across backends.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GGLWEKeyswitch, GLWEKeyswitch,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::GGLWEPreparedFactory,
    },
    test_suite::parity::{ParityBackend, ref_gglwe, ref_glwe},
};

/// Key layout for a `rank_in -> rank_out` switch covering `k` bits of input.
fn key_layout(n: u32, base2k: usize, k: usize, dsize: usize, rank_in: usize, rank_out: usize) -> GGLWELayout {
    let dnum = k.div_ceil(base2k * dsize);
    GGLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k as u32),
        dnum: Dnum(dnum as u32),
        dsize: Dsize(dsize as u32),
        k_aux: TorusPrecision((dsize * base2k) as u32),
        rank_in: Rank(rank_in as u32),
        rank_out: Rank(rank_out as u32),
    }
}

/// `glwe_keyswitch` agrees with the reference backend byte-for-byte.
///
/// Sweeps `dsize` and both ranks, so every branch of the digit loop is compared:
/// the `dsize == 1` short circuit, the `di == 0` overwriting pass, and the
/// narrowed accumulating passes above it.
pub fn test_glwe_keyswitch_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k_in = 4 * base2k + 1;
    let mut source = Source::new([7u8; 32]);

    for rank_in in 1..3usize {
        for rank_out in 1..3usize {
            for dsize in 1..=k_in.div_ceil(base2k) {
                let a_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision(k_in as u32),
                    rank: Rank(rank_in as u32),
                };
                let res_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision((k_in + base2k * dsize) as u32),
                    rank: Rank(rank_out as u32),
                };
                let key_infos = key_layout(n, base2k, k_in, dsize, rank_in, rank_out);

                let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
                let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

                let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
                a_ref.transfer_into(&mut a_test);
                let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
                key_ref_coeffs.transfer_into(&mut key_test_coeffs);

                let mut res_ref = module_ref.glwe_alloc_from_infos(&res_infos);
                let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);

                let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                    module_ref
                        .gglwe_prepare_tmp_bytes(&key_infos)
                        .max(module_ref.glwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
                );
                let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                    module_test
                        .gglwe_prepare_tmp_bytes(&key_infos)
                        .max(module_test.glwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
                );

                let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
                module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
                let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
                module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

                module_ref.glwe_keyswitch(&mut res_ref, &a_ref, &key_ref, &mut scratch_ref.borrow());
                module_test.glwe_keyswitch(&mut res_test, &a_test, &key_test, &mut scratch_test.borrow());

                let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
                res_test.transfer_into(&mut have);
                assert_eq!(
                    res_ref, have,
                    "glwe_keyswitch: rank_in={rank_in} rank_out={rank_out} dsize={dsize} k_in={k_in}"
                );
            }
        }
    }
}

/// `glwe_keyswitch_assign` agrees with the reference backend byte-for-byte.
pub fn test_glwe_keyswitch_assign_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([11u8; 32]);

    for rank in 1..3usize {
        for dsize in 1..=k.div_ceil(base2k) {
            let res_infos = GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
            };
            let key_infos = key_layout(n, base2k, k, dsize, rank, rank);

            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);
            let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);
            let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
            key_ref_coeffs.transfer_into(&mut key_test_coeffs);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                module_ref
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_ref.glwe_keyswitch_tmp_bytes(&res_infos, &res_infos, &key_infos)),
            );
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                module_test
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_test.glwe_keyswitch_tmp_bytes(&res_infos, &res_infos, &key_infos)),
            );

            let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
            module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
            let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
            module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

            module_ref.glwe_keyswitch_assign(&mut res_ref, &key_ref, &mut scratch_ref.borrow());
            module_test.glwe_keyswitch_assign(&mut res_test, &key_test, &mut scratch_test.borrow());

            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "glwe_keyswitch_assign: rank={rank} dsize={dsize} k={k}");
        }
    }
}

/// `gglwe_keyswitch` agrees with the reference backend byte-for-byte.
pub fn test_gglwe_keyswitch_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GGLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GGLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([23u8; 32]);

    for rank in 1..3usize {
        for dsize in 1..=k.div_ceil(base2k) {
            let a_infos = key_layout(n, base2k, k, 1, rank, rank);
            let res_infos = key_layout(n, base2k, k, 1, rank, rank);
            let key_infos = key_layout(n, base2k, k, dsize, rank, rank);

            let a_ref = ref_gglwe(module_ref, &a_infos, &mut source);
            let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

            let mut a_test = module_test.gglwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut a_test);
            let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
            key_ref_coeffs.transfer_into(&mut key_test_coeffs);

            let mut res_ref = module_ref.gglwe_alloc_from_infos(&res_infos);
            let mut res_test = module_test.gglwe_alloc_from_infos(&res_infos);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                module_ref
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_ref.gglwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
            );
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                module_test
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_test.gglwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
            );

            let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
            module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
            let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
            module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

            module_ref.gglwe_keyswitch(&mut res_ref, &a_ref, &key_ref, &mut scratch_ref.borrow());
            module_test.gglwe_keyswitch(&mut res_test, &a_test, &key_test, &mut scratch_test.borrow());

            let mut have = module_ref.gglwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "gglwe_keyswitch: rank={rank} dsize={dsize} k={k}");
        }
    }
}
