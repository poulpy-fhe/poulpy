use super::{
    TestParams, download_scalar_znx, download_vec_znx, scalar_znx_backend_ref, upload_scalar_znx, upload_vec_znx,
    vec_znx_backend_mut, vec_znx_backend_ref,
};
use crate::layouts::SvpPPolToBackendMut;
use crate::layouts::SvpPPolToBackendRef;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;

use crate::test_suite::{PoisonedArena, poisoned_arena};
use crate::{
    api::{
        SvpApplyDft, SvpApplyDftTmpBytes, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolCopy, SvpPPolPrepare,
        VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxFillUniformSource,
        VecZnxFillUniformSourceAll, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Module, PrepareHint, Ring, ScalarZnxAsVecZnxBackendMut, SvpPPolOwned, VecZnxOwned},
    source::Source,
};

use crate::layouts::VecZnxDftOwned;
use crate::test_suite::vec_znx_dft::idft_into_alloc;

pub fn test_svp_apply_dft<BR: crate::test_suite::TestBackend<Rg>, BT: crate::test_suite::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPPolPrepare<BR, Rg>
        + SvpApplyDft<BR, Rg>
        + SvpApplyDftTmpBytes
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR, Rg>
        + VecZnxIdftApply<BR, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPPolPrepare<BT, Rg>
        + SvpApplyDft<BT, Rg>
        + SvpApplyDftTmpBytes
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT, Rg>
        + VecZnxIdftApply<BT, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: PoisonedArena<BR> = poisoned_arena(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: PoisonedArena<BT> = poisoned_arena(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar_ref_backend = module_ref.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR, Rg>::as_vec_znx_backend_mut(&mut scalar_ref_backend),
            col,
            &mut source,
        );
    }
    let scalar = download_scalar_znx::<BR, Rg>(&scalar_ref_backend);

    let mut svp_ref: SvpPPolOwned<BR, Rg> = module_ref.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let mut svp_test: SvpPPolOwned<BT, Rg> = module_test.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let scalar_test_backend = upload_scalar_znx::<BT, Rg>(&scalar);

    for j in 0..cols {
        module_ref.svp_ppol_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR, Rg>(&scalar_ref_backend),
            j,
        );
        module_test.svp_ppol_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT, Rg>(&scalar_test_backend),
            j,
        );
    }

    for a_size in [1, 2, 3, 4] {
        let mut a_ref_backend = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref_backend, &mut source);
        let a = download_vec_znx::<BR, Rg>(&a_ref_backend);
        let a_test_backend = upload_vec_znx::<BT, Rg>(&a);

        let mut scratch_svp_ref: PoisonedArena<BR> = poisoned_arena(module_ref.svp_apply_dft_tmp_bytes(a_size));
        let mut scratch_svp_test: PoisonedArena<BT> = poisoned_arena(module_test.svp_apply_dft_tmp_bytes(a_size));

        for res_size in [1, 2, 3, 4] {
            let mut res_dft_ref: VecZnxDftOwned<BR, Rg> = module_ref.vec_znx_dft_alloc(params.n, cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT, Rg> = module_test.vec_znx_dft_alloc(params.n, cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BR, Rg>(&a_ref_backend),
                    j,
                    &mut scratch_svp_ref.arena(),
                );
                module_test.svp_apply_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BT, Rg>(&a_test_backend),
                    j,
                    &mut scratch_svp_test.arena(),
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &res_dft_test);

            let res_host_template = VecZnxOwned::<i64, Rg>::alloc(params.n, cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR, Rg>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT, Rg>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR, Rg>(&mut res_ref_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT, Rg>(&mut res_test_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR, Rg>(&res_ref_backend);
            let res_test = download_vec_znx::<BT, Rg>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft<BR: crate::test_suite::TestBackend<Rg>, BT: crate::test_suite::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPPolPrepare<BR, Rg>
        + SvpApplyDftToDft<BR, Rg>
        + SvpPPolAlloc<BR>
        + SvpPPolCopy<BR, Rg>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR, Rg>
        + VecZnxDftApply<BR, Rg>
        + VecZnxIdftApply<BR, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPPolPrepare<BT, Rg>
        + SvpApplyDftToDft<BT, Rg>
        + SvpPPolAlloc<BT>
        + SvpPPolCopy<BT, Rg>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT, Rg>
        + VecZnxDftApply<BT, Rg>
        + VecZnxIdftApply<BT, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: PoisonedArena<BR> = poisoned_arena(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: PoisonedArena<BT> = poisoned_arena(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar_ref_backend = module_ref.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR, Rg>::as_vec_znx_backend_mut(&mut scalar_ref_backend),
            col,
            &mut source,
        );
    }
    let scalar = download_scalar_znx::<BR, Rg>(&scalar_ref_backend);

    let mut svp_ref: SvpPPolOwned<BR, Rg> = module_ref.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let mut svp_test: SvpPPolOwned<BT, Rg> = module_test.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let scalar_test_backend = upload_scalar_znx::<BT, Rg>(&scalar);

    for j in 0..cols {
        module_ref.svp_ppol_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR, Rg>(&scalar_ref_backend),
            j,
        );
        module_test.svp_ppol_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT, Rg>(&scalar_test_backend),
            j,
        );
    }

    // Exercise the backend-private prepared representation and its column
    // stride by copying the two factors in reverse order.
    let mut svp_ref_copy: SvpPPolOwned<BR, Rg> = module_ref.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let mut svp_test_copy: SvpPPolOwned<BT, Rg> = module_test.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    for j in 0..cols {
        module_ref.svp_ppol_copy(&mut svp_ref_copy.to_backend_mut(), cols - 1 - j, &svp_ref.to_backend_ref(), j);
        module_test.svp_ppol_copy(
            &mut svp_test_copy.to_backend_mut(),
            cols - 1 - j,
            &svp_test.to_backend_ref(),
            j,
        );
    }

    // Copying into a destination that names the other representation must be
    // rejected before any byte moves.
    {
        let mut other_hint: SvpPPolOwned<BT, Rg> = module_test.svp_ppol_alloc(params.n, cols, PrepareHint::OneShot);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module_test.svp_ppol_copy(&mut other_hint.to_backend_mut(), 0, &svp_test.to_backend_ref(), 0);
        }));
        std::panic::set_hook(hook);
        assert!(caught.is_err(), "svp_ppol_copy accepted a mismatched PrepareHint");
    }

    for a_size in [3] {
        let mut a_ref_backend = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref_backend, &mut source);
        let a = download_vec_znx::<BR, Rg>(&a_ref_backend);
        let a_test_backend = upload_vec_znx::<BT, Rg>(&a);

        let mut a_dft_ref: VecZnxDftOwned<BR, Rg> = module_ref.vec_znx_dft_alloc(params.n, cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT, Rg> = module_test.vec_znx_dft_alloc(params.n, cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR, Rg>(&a_ref_backend),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT, Rg>(&a_test_backend),
                j,
            );
        }

        for res_size in [3] {
            let mut res_dft_ref: VecZnxDftOwned<BR, Rg> = module_ref.vec_znx_dft_alloc(params.n, cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT, Rg> = module_test.vec_znx_dft_alloc(params.n, cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_dft_to_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref_copy.to_backend_ref(),
                    cols - 1 - j,
                    &a_dft_ref.to_backend_ref(),
                    j,
                );
                module_test.svp_apply_dft_to_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test_copy.to_backend_ref(),
                    cols - 1 - j,
                    &a_dft_test.to_backend_ref(),
                    j,
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &res_dft_test);

            let res_host_template = VecZnxOwned::<i64, Rg>::alloc(params.n, cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR, Rg>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT, Rg>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR, Rg>(&mut res_ref_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT, Rg>(&mut res_test_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR, Rg>(&res_ref_backend);
            let res_test = download_vec_znx::<BT, Rg>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft_assign<
    BR: crate::test_suite::TestBackend<Rg>,
    BT: crate::test_suite::TestBackend<Rg>,
    Rg: Ring,
>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPPolPrepare<BR, Rg>
        + SvpApplyDftToDftAssign<BR, Rg>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR, Rg>
        + VecZnxDftApply<BR, Rg>
        + VecZnxIdftApply<BR, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPPolPrepare<BT, Rg>
        + SvpApplyDftToDftAssign<BT, Rg>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT, Rg>
        + VecZnxDftApply<BT, Rg>
        + VecZnxIdftApply<BT, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: PoisonedArena<BR> = poisoned_arena(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: PoisonedArena<BT> = poisoned_arena(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar_ref_backend = module_ref.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR, Rg>::as_vec_znx_backend_mut(&mut scalar_ref_backend),
            col,
            &mut source,
        );
    }
    let scalar = download_scalar_znx::<BR, Rg>(&scalar_ref_backend);

    let mut svp_ref: SvpPPolOwned<BR, Rg> = module_ref.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let mut svp_test: SvpPPolOwned<BT, Rg> = module_test.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    let scalar_test_backend = upload_scalar_znx::<BT, Rg>(&scalar);

    for j in 0..cols {
        module_ref.svp_ppol_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR, Rg>(&scalar_ref_backend),
            j,
        );
        module_test.svp_ppol_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT, Rg>(&scalar_test_backend),
            j,
        );
    }

    for res_size in [1, 2, 3, 4] {
        let mut res_ref_backend_input = module_ref.vec_znx_alloc(params.n, cols, res_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend_input, &mut source);
        let res = download_vec_znx::<BR, Rg>(&res_ref_backend_input);
        let res_test_backend_input = upload_vec_znx::<BT, Rg>(&res);

        let mut res_dft_ref: VecZnxDftOwned<BR, Rg> = module_ref.vec_znx_dft_alloc(params.n, cols, res_size);
        let mut res_dft_test: VecZnxDftOwned<BT, Rg> = module_test.vec_znx_dft_alloc(params.n, cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR, Rg>(&res_ref_backend_input),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT, Rg>(&res_test_backend_input),
                j,
            );
        }

        for j in 0..cols {
            module_ref.svp_apply_dft_to_dft_assign(&mut res_dft_ref.to_backend_mut(), j, &svp_ref.to_backend_ref(), j);
            module_test.svp_apply_dft_to_dft_assign(&mut res_dft_test.to_backend_mut(), j, &svp_test.to_backend_ref(), j);
        }

        let res_big_ref = idft_into_alloc(module_ref, &res_dft_ref);
        let res_big_test = idft_into_alloc(module_test, &res_dft_test);

        let res_host_template = VecZnxOwned::<i64, Rg>::alloc(params.n, cols, res_size);
        let mut res_ref_backend = upload_vec_znx::<BR, Rg>(&res_host_template);
        let mut res_test_backend = upload_vec_znx::<BT, Rg>(&res_host_template);

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BR, Rg>(&mut res_ref_backend),
                base2k,
                res_size * base2k,
                0,
                j,
                &res_big_ref.to_backend_ref(),
                base2k,
                j,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BT, Rg>(&mut res_test_backend),
                base2k,
                res_size * base2k,
                0,
                j,
                &res_big_test.to_backend_ref(),
                base2k,
                j,
                &mut scratch_test.arena(),
            );
        }

        let res_ref = download_vec_znx::<BR, Rg>(&res_ref_backend);
        let res_test = download_vec_znx::<BT, Rg>(&res_test_backend);
        assert_eq!(res_ref, res_test);
    }
}
