//! Pairwise backend parity for the digit-product OEP.
use super::{ParityBackend, ParityShapes, poisoned_scratch};
use crate::{
    layouts::{Base2K, gadget_product_limbs},
    oep::GGLWEProductDigitsStridedImpl,
};
use poulpy_hal::{
    AlignedBuf,
    api::{
        VecZnxAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxFillUniformSource,
        VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftAddTmpBytes, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        MatZnx, MatZnxAtBackendMut, MatZnxToBackendRef, Module, PrepareHint, VecZnx, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxViewMut,
    },
    source::Source,
    test_suite::{TestParams, download_mat_znx, download_vec_znx, upload_mat_znx, upload_vec_znx, vec_znx_backend_mut},
};

/// HAL operations needed to construct and canonically observe a digit product.
/// This keeps the test independent of host pointers or transform representations.
pub trait DigitParityModule<BE: ParityBackend>:
    VecZnxAlloc<BE>
    + VecZnxDftAlloc<BE>
    + VecZnxDftApply<BE>
    + VecZnxDftBytesOf
    + VecZnxDftCopy<BE>
    + VecZnxDftZero<BE>
    + VecZnxFillUniformSource<BE>
    + VecZnxIdftNormalizeConsume<BE>
    + VecZnxIdftNormalizeConsumeTmpBytes
    + VmpPMatAlloc<BE>
    + VmpPrepare<BE>
    + VmpPrepareTmpBytes
    + VmpApplyDftToDft<BE>
    + VmpApplyDftToDftAdd<BE>
    + VmpApplyDftToDftTmpBytes
    + VmpApplyDftToDftAddTmpBytes
{
}
impl<BE: ParityBackend, M> DigitParityModule<BE> for M where
    M: VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxIdftNormalizeConsume<BE>
        + VecZnxIdftNormalizeConsumeTmpBytes
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAddTmpBytes
{
}

fn product<BE>(
    module: &Module<BE>,
    a: &VecZnx<AlignedBuf, i64>,
    mat: &MatZnx<AlignedBuf, i64>,
    base2k: usize,
    dsize: usize,
) -> VecZnx<AlignedBuf, i64>
where
    BE: ParityBackend + GGLWEProductDigitsStridedImpl,
    Module<BE>: DigitParityModule<BE>,
{
    let (cols_in, cols_out, rows, size) = (mat.cols_in(), mat.cols_out(), mat.rows(), mat.size());
    let input = upload_vec_znx::<BE>(a);
    let mut input_dft = module.vec_znx_dft_alloc(module.n(), cols_in, a.size());
    for col in 0..cols_in {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut input_dft.to_backend_mut(),
            col,
            &<VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendRef<BE>>::to_backend_ref(&input),
            col,
        );
    }
    let mat = upload_mat_znx::<BE>(mat);
    let mut key = module.vmp_pmat_alloc(module.n(), rows, cols_in, cols_out, size, PrepareHint::Reuse);
    module.vmp_prepare(
        &mut key.to_backend_mut(),
        &<MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&mat),
        &mut poisoned_scratch::<BE>(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)).arena(),
    );
    let mut result = module.vec_znx_dft_alloc(module.n(), cols_out, size);
    let bytes = BE::len_bytes(&result.data);
    BE::copy_from_host(&mut result.data, &vec![0x55; bytes]);
    let terms = module.n() * rows * dsize * cols_in;
    let limbs = gadget_product_limbs(Base2K(base2k as u32), terms);
    let tmp = BE::gglwe_product_digits_strided_tmp_bytes(module, size, cols_in, a.size(), dsize, rows, cols_in, cols_out, size);
    BE::gglwe_product_digits_strided(
        module,
        &mut result.to_backend_mut(),
        &input_dft.to_backend_ref(),
        dsize,
        limbs,
        &key.to_backend_ref(),
        &mut poisoned_scratch::<BE>(tmp).arena(),
    );
    let mut output = module.vec_znx_alloc(module.n(), cols_out, size);
    for col in 0..cols_out {
        module.vec_znx_idft_normalize_consume(
            &mut <VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut output),
            base2k,
            size * base2k,
            col,
            &mut result.to_backend_mut(),
            col,
            base2k,
            None,
            &mut poisoned_scratch::<BE>(module.vec_znx_idft_normalize_consume_tmp_bytes(size, size)).arena(),
        );
    }
    download_vec_znx::<BE>(&output)
}

/// Each side prepares the coefficient input and matrix independently; comparison
/// happens only after inverse transformation and canonical normalization.
pub fn test_gglwe_product_digits_strided_parity<BR, BT>(
    params: &TestParams,
    _shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    BR: ParityBackend + GGLWEProductDigitsStridedImpl,
    BT: ParityBackend + GGLWEProductDigitsStridedImpl,
    Module<BR>: DigitParityModule<BR>,
    Module<BT>: DigitParityModule<BT>,
{
    let base2k = params.base2k;
    let mut source = Source::new([127; 32]);
    for (dsize, cols_in, cols_out, size) in [(1usize, 1, 1, 1), (2, 1, 2, 5), (3, 2, 1, 8), (7, 1, 2, 15)] {
        for sparse in [false, true] {
            let mut a = r.vec_znx_alloc(r.n(), cols_in, size);
            for col in 0..cols_in {
                r.vec_znx_fill_uniform_source(
                    base2k,
                    size * base2k,
                    &mut vec_znx_backend_mut::<BR>(&mut a),
                    col,
                    &mut source,
                );
            }
            let mut a = download_vec_znx::<BR>(&a);
            if sparse {
                for col in 0..cols_in {
                    for limb in 0..size - 1 {
                        a.at_mut(col, limb).fill(0);
                    }
                }
            }
            let mut matrix = r.mat_znx_alloc(r.n(), size.div_ceil(dsize), cols_in, cols_out, size);
            for row in 0..size.div_ceil(dsize) {
                for col in 0..cols_in {
                    let mut view = MatZnxAtBackendMut::<BR>::at_backend_mut(&mut matrix, row, col);
                    for out in 0..cols_out {
                        r.vec_znx_fill_uniform_source(base2k, size * base2k, &mut view, out, &mut source);
                    }
                }
            }
            let matrix = download_mat_znx::<BR>(&matrix);
            let want = product(r, &a, &matrix, params.base2k, dsize);
            let have = product(t, &a, &matrix, params.base2k, dsize);
            assert_eq!(
                want, have,
                "digit product dsize={dsize} input={cols_in} output={cols_out} size={size} sparse={sparse}"
            );
        }
    }
}
