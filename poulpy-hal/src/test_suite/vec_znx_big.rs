use super::TestParams;
use rand::Rng;

use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddSmallAssign,
        VecZnxBigAddSmallInto, VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismAssign,
        VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigFromSmall, VecZnxBigNegate, VecZnxBigNegateAssign, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxBigSubSmallA,
        VecZnxBigSubSmallAssign, VecZnxBigSubSmallB, VecZnxBigSubSmallNegateAssign,
    },
    layouts::{
        Backend, DataViewMut, DeviceBuf, DigestU64, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig, ZnxView, ZnxViewMut,
    },
    source::Source,
};

type VecZnxBigOwned<BE> = VecZnxBig<DeviceBuf<BE>, BE>;

pub fn test_vec_znx_big_add_into<BR: Backend, BT: Backend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigAddInto<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigAddInto<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest = b.digest_u64();

            let mut b_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref, j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test, j, &b, j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_into(&mut res_big_ref, i, &a_ref, i, &b_ref, i);
                    module_test.vec_znx_big_add_into(&mut res_big_test, i, &a_test, i, &b_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b_ref.digest_u64(), b_ref_digest);
                assert_eq!(b_test.digest_u64(), b_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_assign(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_add_assign(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_add_small_into<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddSmallInto<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallInto<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_small_into(&mut res_big_ref, i, &a_ref, i, &b, i);
                    module_test.vec_znx_big_add_small_into(&mut res_big_test, i, &a_test, i, &b, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_small_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_small_assign(&mut res_big_ref, i, &a, i);
                module_test.vec_znx_big_add_small_assign(&mut res_big_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_automorphism<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAutomorphism<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphism<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for p in [-5, 5] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_automorphism(p, &mut res_big_ref, i, &a_ref, i);
                    module_test.vec_znx_big_automorphism(p, &mut res_big_test, i, &a_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_automorphism_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAutomorphismAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphismAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for p in [-5, 5] {
            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_automorphism_assign(p, &mut res_big_ref, i, scratch_ref.borrow());
                module_test.vec_znx_big_automorphism_assign(p, &mut res_big_test, i, scratch_test.borrow());
            }

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate<BR: Backend, BT: Backend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigNegate<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigNegate<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            // Set res to garbage
            source.fill_bytes(res_big_ref.data_mut().as_mut());
            source.fill_bytes(res_big_test.data_mut().as_mut());

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_big_negate(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_negate(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
            module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
        }

        for i in 0..cols {
            module_ref.vec_znx_big_negate_assign(&mut res_big_ref, i);
            module_test.vec_znx_big_negate_assign(&mut res_big_test, i);
        }

        let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        let res_ref_digest: u64 = res_big_ref.digest_u64();
        let res_test_digest: u64 = res_big_test.digest_u64();

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(
                &mut res_small_ref,
                base2k,
                0,
                j,
                &res_big_ref,
                base2k,
                j,
                scratch_ref.borrow(),
            );
            module_test.vec_znx_big_normalize(
                &mut res_small_test,
                base2k,
                0,
                j,
                &res_big_test,
                base2k,
                j,
                scratch_test.borrow(),
            );
        }

        assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
        assert_eq!(res_big_test.digest_u64(), res_test_digest);

        assert_eq!(res_small_ref, res_small_test);
    }
}

pub fn test_vec_znx_big_normalize<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(63, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for res_offset in -(base2k as i64)..=(base2k as i64) {
                // Set d to garbage
                source.fill_bytes(res_ref.data_mut());
                source.fill_bytes(res_test.data_mut());

                // Reference
                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_ref,
                        base2k,
                        res_offset,
                        j,
                        &a_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_test,
                        base2k,
                        res_offset,
                        j,
                        &a_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_big_normalize_fused<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([1u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(63, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);
        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let mut base_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut base_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                base_ref.fill_uniform(base2k, &mut source);
                base_test.data_mut().copy_from_slice(&base_ref.data);

                let mut normalized_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut normalized_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                module_ref.vec_znx_big_normalize_into(
                    &mut normalized_ref,
                    base2k,
                    res_offset,
                    0,
                    &a_ref,
                    base2k,
                    0,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize_into(
                    &mut normalized_test,
                    base2k,
                    res_offset,
                    0,
                    &a_test,
                    base2k,
                    0,
                    scratch_test.borrow(),
                );
                assert_eq!(normalized_ref, normalized_test);

                let mut add_ref = base_ref.clone();
                let mut add_test = base_test.clone();
                module_ref.vec_znx_big_normalize_add_assign(
                    &mut add_ref,
                    base2k,
                    res_offset,
                    0,
                    &a_ref,
                    base2k,
                    0,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize_add_assign(
                    &mut add_test,
                    base2k,
                    res_offset,
                    0,
                    &a_test,
                    base2k,
                    0,
                    scratch_test.borrow(),
                );
                assert_eq!(add_ref, add_test);

                // Fused-vs-unfused: `_add_assign` must equal `base + normalize(a)` limb-wise.
                let mut expected_add = base_ref.clone();
                for j in 0..res_size {
                    for (e, n) in expected_add.at_mut(0, j).iter_mut().zip(normalized_ref.at(0, j).iter()) {
                        *e = e.wrapping_add(*n);
                    }
                }
                assert_eq!(add_ref, expected_add);

                let mut sub_ref = base_ref.clone();
                let mut sub_test = base_test.clone();
                module_ref.vec_znx_big_normalize_sub_assign(
                    &mut sub_ref,
                    base2k,
                    res_offset,
                    0,
                    &a_ref,
                    base2k,
                    0,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize_sub_assign(
                    &mut sub_test,
                    base2k,
                    res_offset,
                    0,
                    &a_test,
                    base2k,
                    0,
                    scratch_test.borrow(),
                );
                assert_eq!(sub_ref, sub_test);

                // Fused-vs-unfused: `_sub_assign` must equal `base - normalize(a)` limb-wise.
                let mut expected_sub = base_ref.clone();
                for j in 0..res_size {
                    for (e, n) in expected_sub.at_mut(0, j).iter_mut().zip(normalized_ref.at(0, j).iter()) {
                        *e = e.wrapping_sub(*n);
                    }
                }
                assert_eq!(sub_ref, expected_sub);

                let mut neg_ref = base_ref;
                let mut neg_test = base_test;
                module_ref.vec_znx_big_normalize_negate(
                    &mut neg_ref,
                    base2k,
                    res_offset,
                    0,
                    &a_ref,
                    base2k,
                    0,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize_negate(
                    &mut neg_test,
                    base2k,
                    res_offset,
                    0,
                    &a_test,
                    base2k,
                    0,
                    scratch_test.borrow(),
                );
                assert_eq!(neg_ref, neg_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub<BR: Backend, BT: Backend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigSub<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigSub<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);

            let mut b_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref, j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test, j, &b, j);
            }

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub(&mut res_big_ref, i, &a_ref, i, &b_ref, i);
                    module_test.vec_znx_big_sub(&mut res_big_test, i, &a_test, i, &b_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b_ref.digest_u64(), b_ref_digest);
                assert_eq!(b_test.digest_u64(), b_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_assign(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_sub_assign(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_negate_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_negate_assign(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_sub_negate_assign(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_a<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallA<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallA<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_a(&mut res_big_ref, i, &b, i, &a_ref, i);
                    module_test.vec_znx_big_sub_small_a(&mut res_big_test, i, &b, i, &a_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_b<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallB<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallB<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_b(&mut res_big_ref, i, &a_ref, i, &b, i);
                    module_test.vec_znx_big_sub_small_b(&mut res_big_test, i, &a_test, i, &b, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        0,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        0,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_a_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_small_assign(&mut res_big_ref, i, &a, i);
                module_test.vec_znx_big_sub_small_assign(&mut res_big_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut res_small_ref,
                    base2k,
                    0,
                    j,
                    &res_big_ref,
                    base2k,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    &mut res_small_test,
                    base2k,
                    0,
                    j,
                    &res_big_test,
                    base2k,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_b_assign<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                res.fill_uniform(base2k, &mut source);

                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                    module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
                }

                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_negate_assign(&mut res_big_ref, i, &a, i);
                    module_test.vec_znx_big_sub_small_negate_assign(&mut res_big_test, i, &a, i);
                }

                assert_eq!(a.digest_u64(), a_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut res_small_ref,
                        base2k,
                        res_offset,
                        j,
                        &res_big_ref,
                        base2k,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut res_small_test,
                        base2k,
                        res_offset,
                        j,
                        &res_big_test,
                        base2k,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}
