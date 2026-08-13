use super::{
    TestParams, download_vec_znx, scalar_znx_backend_ref, upload_scalar_znx, upload_vec_znx, vec_znx_backend_mut,
    vec_znx_backend_ref,
};
use crate::layouts::SvpPPolToBackendMut;
use crate::layouts::SvpPPolToBackendRef;
use crate::layouts::SvpTPolToBackendMut;
use crate::layouts::SvpTPolToBackendRef;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;

use crate::{
    api::{
        ScratchOwnedAlloc, SvpApplyPPolDftToBig, SvpApplyPPolDftToDft, SvpApplyPPolDftToDftAssign, SvpApplyPPolDftToSmall,
        SvpApplyPPolSmallToBig, SvpApplyPPolSmallToDft, SvpApplyPPolSmallToSmall, SvpApplySmallDftToBig, SvpApplySmallDftToDft,
        SvpApplySmallDftToDftAssign, SvpApplySmallDftToSmall, SvpApplySmallSmallToBig, SvpApplySmallSmallToDft,
        SvpApplySmallSmallToSmall, SvpApplyTPolDftToBig, SvpApplyTPolDftToDft, SvpApplyTPolDftToDftAssign,
        SvpApplyTPolDftToSmall, SvpApplyTPolSmallToBig, SvpApplyTPolSmallToDft, SvpApplyTPolSmallToSmall, SvpApplyToBigTmpBytes,
        SvpApplyToSmallTmpBytes, SvpPPolAlloc, SvpPreparePPol, SvpPrepareTPol, SvpTPolAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, FillUniform, HostBytesBackend, Module, ScratchOwned, SvpPPolOwned, SvpTPolOwned},
    source::Source,
};

use crate::layouts::VecZnxBigOwned;
use crate::layouts::VecZnxDftOwned;
use crate::layouts::ZnxView;

fn idft_into_alloc<BE>(module: &Module<BE>, a: &mut VecZnxDftOwned<BE>) -> VecZnxBigOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApplyTmpA<BE>,
{
    let cols = a.cols();
    let size = a.size();
    let mut res = module.vec_znx_big_alloc(cols, size);
    for j in 0..cols {
        let mut res_backend = res.to_backend_mut();
        let mut a_backend = a.to_backend_mut();
        module.vec_znx_idft_apply_tmpa(&mut res_backend, j, &mut a_backend, j);
    }
    res
}

pub fn test_svp_apply_ppol_small_to_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPreparePPol<BR>
        + SvpApplyPPolSmallToDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPreparePPol<BT>
        + SvpApplyPPolSmallToDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare_ppol(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare_ppol(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_ref_backend = upload_vec_znx::<BR>(&a);
        let a_test_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_ppol_small_to_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BR>(&a_ref_backend),
                    j,
                );
                module_test.svp_apply_ppol_small_to_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BT>(&a_test_backend),
                    j,
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let res_host_template = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            let res_test = download_vec_znx::<BT>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_ppol_dft_to_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPreparePPol<BR>
        + SvpApplyPPolDftToDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPreparePPol<BT>
        + SvpApplyPPolDftToDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare_ppol(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare_ppol(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for a_size in [3] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_ref_backend = upload_vec_znx::<BR>(&a);
        let a_test_backend = upload_vec_znx::<BT>(&a);

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR>(&a_ref_backend),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT>(&a_test_backend),
                j,
            );
        }

        for res_size in [3] {
            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_ppol_dft_to_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref.to_backend_ref(),
                    j,
                    &a_dft_ref.to_backend_ref(),
                    j,
                );
                module_test.svp_apply_ppol_dft_to_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test.to_backend_ref(),
                    j,
                    &a_dft_test.to_backend_ref(),
                    j,
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let res_host_template = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            let res_test = download_vec_znx::<BT>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_ppol_dft_to_dft_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPreparePPol<BR>
        + SvpApplyPPolDftToDftAssign<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPreparePPol<BT>
        + SvpApplyPPolDftToDftAssign<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare_ppol(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare_ppol(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for res_size in [1, 2, 3, 4] {
        let mut res = module_host.vec_znx_alloc(cols, res_size);
        res.fill_uniform(base2k, &mut source);
        let res_ref_backend_input = upload_vec_znx::<BR>(&res);
        let res_test_backend_input = upload_vec_znx::<BT>(&res);

        let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
        let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR>(&res_ref_backend_input),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT>(&res_test_backend_input),
                j,
            );
        }

        for j in 0..cols {
            module_ref.svp_apply_ppol_dft_to_dft_assign(&mut res_dft_ref.to_backend_mut(), j, &svp_ref.to_backend_ref(), j);
            module_test.svp_apply_ppol_dft_to_dft_assign(&mut res_dft_test.to_backend_mut(), j, &svp_test.to_backend_ref(), j);
        }

        let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
        let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

        let res_host_template = module_host.vec_znx_alloc(cols, res_size);
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                base2k,
                0,
                j,
                &res_big_ref.to_backend_ref(),
                base2k,
                j,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                base2k,
                0,
                j,
                &res_big_test.to_backend_ref(),
                base2k,
                j,
                &mut scratch_test.arena(),
            );
        }

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        let res_test = download_vec_znx::<BT>(&res_test_backend);
        assert_eq!(res_ref, res_test);
    }
}

/// Bundle of the SVP capabilities exercised by the domain-matrix tests.
///
/// Written as a supertrait bundle so the two exported tests can name it once
/// instead of repeating twenty bounds each.
pub trait SvpMatrixModule<BE: Backend>:
    SvpPPolAlloc<BE>
    + SvpTPolAlloc<BE>
    + SvpPreparePPol<BE>
    + SvpPrepareTPol<BE>
    + SvpApplySmallSmallToDft<BE>
    + SvpApplySmallDftToDft<BE>
    + SvpApplyTPolSmallToDft<BE>
    + SvpApplyTPolDftToDft<BE>
    + SvpApplyPPolSmallToDft<BE>
    + SvpApplyPPolDftToDft<BE>
    + SvpApplySmallSmallToBig<BE>
    + SvpApplySmallDftToBig<BE>
    + SvpApplyTPolSmallToBig<BE>
    + SvpApplyTPolDftToBig<BE>
    + SvpApplyPPolSmallToBig<BE>
    + SvpApplyPPolDftToBig<BE>
    + SvpApplySmallSmallToSmall<BE>
    + SvpApplySmallDftToSmall<BE>
    + SvpApplyTPolSmallToSmall<BE>
    + SvpApplyTPolDftToSmall<BE>
    + SvpApplyPPolSmallToSmall<BE>
    + SvpApplyPPolDftToSmall<BE>
    + SvpApplySmallDftToDftAssign<BE>
    + SvpApplyTPolDftToDftAssign<BE>
    + SvpApplyPPolDftToDftAssign<BE>
    + SvpApplyToBigTmpBytes
    + SvpApplyToSmallTmpBytes
    + VecZnxDftAlloc<BE>
    + VecZnxBigAlloc<BE>
    + VecZnxDftApply<BE>
    + VecZnxIdftApplyTmpA<BE>
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
{
}

impl<BE: Backend, M> SvpMatrixModule<BE> for M where
    M: SvpPPolAlloc<BE>
        + SvpTPolAlloc<BE>
        + SvpPreparePPol<BE>
        + SvpPrepareTPol<BE>
        + SvpApplySmallSmallToDft<BE>
        + SvpApplySmallDftToDft<BE>
        + SvpApplyTPolSmallToDft<BE>
        + SvpApplyTPolDftToDft<BE>
        + SvpApplyPPolSmallToDft<BE>
        + SvpApplyPPolDftToDft<BE>
        + SvpApplySmallSmallToBig<BE>
        + SvpApplySmallDftToBig<BE>
        + SvpApplyTPolSmallToBig<BE>
        + SvpApplyTPolDftToBig<BE>
        + SvpApplyPPolSmallToBig<BE>
        + SvpApplyPPolDftToBig<BE>
        + SvpApplySmallSmallToSmall<BE>
        + SvpApplySmallDftToSmall<BE>
        + SvpApplyTPolSmallToSmall<BE>
        + SvpApplyTPolDftToSmall<BE>
        + SvpApplyPPolSmallToSmall<BE>
        + SvpApplyPPolDftToSmall<BE>
        + SvpApplySmallDftToDftAssign<BE>
        + SvpApplyTPolDftToDftAssign<BE>
        + SvpApplyPPolDftToDftAssign<BE>
        + SvpApplyToBigTmpBytes
        + SvpApplyToSmallTmpBytes
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
{
}

/// Every `svp_apply_<scalar>_<vector>_to_<out>` variant must compute the same product.
///
/// The `ppol` / `dft` / `to_dft` combination is the reference; the other seventeen
/// are normalized back down to coefficients and compared against it. This is
/// what pins the tiers together: a backend that specializes `svp_prepare_tpol` or
/// any single apply kernel cannot silently disagree with the rest.
fn check_svp_domain_matrix<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
) where
    Module<BE>: SvpMatrixModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let base2k = params.base2k;
    let cols: usize = 2;
    let size: usize = 3;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_backend = upload_scalar_znx::<BE>(&scalar);

    let mut a = module_host.vec_znx_alloc(cols, size);
    a.fill_uniform(base2k, &mut source);
    let a_backend = upload_vec_znx::<BE>(&a);

    let mut ppol: SvpPPolOwned<BE> = module.svp_ppol_alloc(cols);
    let mut tpol: SvpTPolOwned<BE> = module.svp_tpol_alloc(cols);
    for j in 0..cols {
        module.svp_prepare_ppol(
            &mut ppol.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BE>(&scalar_backend),
            j,
        );
        module.svp_prepare_tpol(
            &mut tpol.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BE>(&scalar_backend),
            j,
        );
    }

    let mut a_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
    for j in 0..cols {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut a_dft.to_backend_mut(),
            j,
            &vec_znx_backend_ref::<BE>(&a_backend),
            j,
        );
    }

    let scratch_bytes = module
        .svp_apply_to_big_tmp_bytes(size)
        .max(module.svp_apply_to_small_tmp_bytes(size))
        .max(module.vec_znx_big_normalize_tmp_bytes());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    let host_template = module_host.vec_znx_alloc(cols, size);

    // Normalizes a DFT-domain result down to coefficients and downloads it.
    let settle_dft = |res_dft: &mut VecZnxDftOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        let res_big = idft_into_alloc(module, res_dft);
        let mut out = upload_vec_znx::<BE>(&host_template);
        for j in 0..cols {
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut out),
                base2k,
                0,
                j,
                &res_big.to_backend_ref(),
                base2k,
                j,
                &mut scratch.arena(),
            );
        }
        download_vec_znx::<BE>(&out)
    };

    // Reference: cold-prepared scalar, DFT input, DFT output.
    let want = {
        let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
        for j in 0..cols {
            module.svp_apply_ppol_dft_to_dft(
                &mut res_dft.to_backend_mut(),
                j,
                &ppol.to_backend_ref(),
                j,
                &a_dft.to_backend_ref(),
                j,
            );
        }
        settle_dft(&mut res_dft, &mut scratch)
    };

    // Guards against a vacuous pass: every variant matching an all-zero
    // reference would prove nothing.
    assert!(
        (0..cols).any(|j| (0..size).any(|k| want.at(j, k).iter().any(|&c| c != 0))),
        "reference product is all zero"
    );

    // `_to_dft`, remaining five rhs/input domain pairs.
    macro_rules! check_to_dft {
        ($label:literal, $method:ident, $rhs:expr, $input:expr) => {{
            let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
            for j in 0..cols {
                module.$method(&mut res_dft.to_backend_mut(), j, &$rhs, j, &$input, j);
            }
            assert_eq!(settle_dft(&mut res_dft, &mut scratch), want, "{} disagrees", $label);
        }};
    }

    check_to_dft!(
        "small_small_to_dft",
        svp_apply_small_small_to_dft,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_dft!(
        "small_dft_to_dft",
        svp_apply_small_dft_to_dft,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        a_dft.to_backend_ref()
    );
    check_to_dft!(
        "t_small_to_dft",
        svp_apply_tpol_small_to_dft,
        tpol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_dft!(
        "t_dft_to_dft",
        svp_apply_tpol_dft_to_dft,
        tpol.to_backend_ref(),
        a_dft.to_backend_ref()
    );
    check_to_dft!(
        "p_small_to_dft",
        svp_apply_ppol_small_to_dft,
        ppol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );

    // `_assign`: seed the accumulator with the input, then multiply in place.
    macro_rules! check_assign {
        ($label:literal, $method:ident, $rhs:expr) => {{
            let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
            for j in 0..cols {
                module.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    j,
                );
            }
            for j in 0..cols {
                module.$method(&mut res_dft.to_backend_mut(), j, &$rhs, j);
            }
            assert_eq!(settle_dft(&mut res_dft, &mut scratch), want, "{} disagrees", $label);
        }};
    }

    check_assign!(
        "small_dft_to_dft_assign",
        svp_apply_small_dft_to_dft_assign,
        scalar_znx_backend_ref::<BE>(&scalar_backend)
    );
    check_assign!("t_dft_to_dft_assign", svp_apply_tpol_dft_to_dft_assign, tpol.to_backend_ref());
    check_assign!("p_dft_to_dft_assign", svp_apply_ppol_dft_to_dft_assign, ppol.to_backend_ref());

    // `_to_big`: same product, IDFT folded in.
    macro_rules! check_to_big {
        ($label:literal, $method:ident, $rhs:expr, $input:expr) => {{
            let mut res_big = module.vec_znx_big_alloc(cols, size);
            for j in 0..cols {
                module.$method(&mut res_big.to_backend_mut(), j, &$rhs, j, &$input, j, &mut scratch.arena());
            }
            let mut out = upload_vec_znx::<BE>(&host_template);
            for j in 0..cols {
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut out),
                    base2k,
                    0,
                    j,
                    &res_big.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch.arena(),
                );
            }
            assert_eq!(download_vec_znx::<BE>(&out), want, "{} disagrees", $label);
        }};
    }

    check_to_big!(
        "small_small_to_big",
        svp_apply_small_small_to_big,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_big!(
        "small_dft_to_big",
        svp_apply_small_dft_to_big,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        a_dft.to_backend_ref()
    );
    check_to_big!(
        "t_small_to_big",
        svp_apply_tpol_small_to_big,
        tpol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_big!(
        "t_dft_to_big",
        svp_apply_tpol_dft_to_big,
        tpol.to_backend_ref(),
        a_dft.to_backend_ref()
    );
    check_to_big!(
        "p_small_to_big",
        svp_apply_ppol_small_to_big,
        ppol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_big!(
        "p_dft_to_big",
        svp_apply_ppol_dft_to_big,
        ppol.to_backend_ref(),
        a_dft.to_backend_ref()
    );

    // `_to_small`: IDFT and normalization both folded in.
    macro_rules! check_to_small {
        ($label:literal, $method:ident, $rhs:expr, $input:expr) => {{
            let mut out = upload_vec_znx::<BE>(&host_template);
            for j in 0..cols {
                module.$method(
                    &mut vec_znx_backend_mut::<BE>(&mut out),
                    base2k,
                    0,
                    j,
                    &$rhs,
                    j,
                    &$input,
                    base2k,
                    j,
                    &mut scratch.arena(),
                );
            }
            assert_eq!(download_vec_znx::<BE>(&out), want, "{} disagrees", $label);
        }};
    }

    check_to_small!(
        "small_small_to_small",
        svp_apply_small_small_to_small,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_small!(
        "small_dft_to_small",
        svp_apply_small_dft_to_small,
        scalar_znx_backend_ref::<BE>(&scalar_backend),
        a_dft.to_backend_ref()
    );
    check_to_small!(
        "t_small_to_small",
        svp_apply_tpol_small_to_small,
        tpol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_small!(
        "t_dft_to_small",
        svp_apply_tpol_dft_to_small,
        tpol.to_backend_ref(),
        a_dft.to_backend_ref()
    );
    check_to_small!(
        "p_small_to_small",
        svp_apply_ppol_small_to_small,
        ppol.to_backend_ref(),
        vec_znx_backend_ref::<BE>(&a_backend)
    );
    check_to_small!(
        "p_dft_to_small",
        svp_apply_ppol_dft_to_small,
        ppol.to_backend_ref(),
        a_dft.to_backend_ref()
    );
}

/// Runs the domain-matrix consistency check on both registered backends.
pub fn test_svp_apply_domain_matrix<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpMatrixModule<BR>,
    Module<BT>: SvpMatrixModule<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    check_svp_domain_matrix(params, module_host, module_ref);
    check_svp_domain_matrix(params, module_host, module_test);
}
