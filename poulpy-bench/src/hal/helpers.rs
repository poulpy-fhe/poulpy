//! Host-buffer allocation, upload, and backend-view helpers used only by the
//! HAL runner functions in this module tree — not part of the crate's public
//! API.

use poulpy_hal::layouts::{
    Backend, CnvPVecLOwned, CnvPVecROwned, DataView, MatZnx, MatZnxBackendRef, MatZnxToBackendRef, ScalarZnx,
    ScalarZnxBackendRef, ScalarZnxToBackendRef, SvpPPol, SvpPPolOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
    VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VecZnxDftOwned, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
    VmpPMat, VmpPMatOwned,
};
use poulpy_hal::source::Source;
use rand::Rng;

fn random_aligned_host_bytes(len: usize, source: &mut Source) -> Vec<u8> {
    let mut bytes = poulpy_hal::alloc_aligned_custom::<u8>(len, poulpy_hal::DEFAULTALIGN);
    source.fill_bytes(&mut bytes);
    bytes
}

pub fn upload_host_vec_znx<BE: Backend<ZnxWord = i64>>(src: &VecZnx<Vec<u8>, i64>) -> VecZnx<BE::OwnedBuf, BE::ZnxWord> {
    VecZnx::from_data(BE::from_host_bytes(src.data()), src.n(), src.cols(), src.size())
}

pub fn upload_host_scalar_znx<BE: Backend<ZnxWord = i64>>(src: &ScalarZnx<Vec<u8>, i64>) -> ScalarZnx<BE::OwnedBuf, BE::ZnxWord> {
    ScalarZnx::from_data(BE::from_host_bytes(src.data()), src.n(), src.cols())
}

pub fn upload_host_mat_znx<BE: Backend<ZnxWord = i64>>(src: &MatZnx<Vec<u8>, i64>) -> MatZnx<BE::OwnedBuf, BE::ZnxWord> {
    MatZnx::from_data(
        BE::from_host_bytes(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

pub fn random_host_scalar_znx(n: usize, cols: usize, source: &mut Source) -> ScalarZnx<Vec<u8>, i64> {
    let bytes = random_aligned_host_bytes(ScalarZnx::<Vec<u8>, i64>::bytes_of(n, cols), source);
    ScalarZnx::from_bytes(n, cols, bytes)
}

pub fn random_host_vec_znx(n: usize, cols: usize, size: usize, source: &mut Source) -> VecZnx<Vec<u8>, i64> {
    let bytes = random_aligned_host_bytes(VecZnx::<Vec<u8>, i64>::bytes_of(n, cols, size), source);
    VecZnx::from_bytes(n, cols, size, bytes)
}

pub fn random_host_mat_znx(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> MatZnx<Vec<u8>, i64> {
    let bytes = random_aligned_host_bytes(MatZnx::<Vec<u8>, i64>::bytes_of(n, rows, cols_in, cols_out, size), source);
    MatZnx::from_bytes(n, rows, cols_in, cols_out, size, bytes)
}

pub fn random_backend_vec_znx_dft<BE: Backend<ZnxWord = i64>>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> VecZnxDftOwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vec_znx_dft(n, cols, size)];
    source.fill_bytes(&mut bytes);
    VecZnxDftOwned::<BE>::from_bytes(n, cols, size, bytes)
}

#[allow(dead_code)]
pub fn random_backend_vec_znx_big<BE: Backend<ZnxWord = i64>>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> VecZnxBigOwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vec_znx_big(n, cols, size)];
    source.fill_bytes(&mut bytes);
    VecZnxBigOwned::<BE>::from_bytes(n, cols, size, bytes)
}

pub fn random_backend_svp_ppol<BE: Backend<ZnxWord = i64>>(n: usize, cols: usize, source: &mut Source) -> SvpPPolOwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_svp_ppol(n, cols)];
    source.fill_bytes(&mut bytes);
    SvpPPol::from_data(BE::from_host_bytes(&bytes), n, cols)
}

pub fn random_backend_vmp_pmat<BE: Backend<ZnxWord = i64>>(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> VmpPMatOwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size)];
    source.fill_bytes(&mut bytes);
    VmpPMat::from_data(BE::from_host_bytes(&bytes), n, rows, cols_in, cols_out, size)
}

pub fn random_backend_cnv_pvec_left<BE: Backend<ZnxWord = i64>>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecLOwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_left(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecLOwned::<BE>::from_bytes(n, cols, size, bytes)
}

pub fn random_backend_cnv_pvec_right<BE: Backend<ZnxWord = i64>>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecROwned<BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_right(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecROwned::<BE>::from_bytes(n, cols, size, bytes)
}

pub fn scalar_znx_backend_ref<'a, BE: Backend<ZnxWord = i64>>(
    src: &'a ScalarZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> ScalarZnxBackendRef<'a, BE> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_backend_ref<'a, BE: Backend<ZnxWord = i64>>(
    src: &'a VecZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> VecZnxBackendRef<'a, BE> {
    <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_backend_mut<'a, BE: Backend<ZnxWord = i64>>(
    src: &'a mut VecZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> VecZnxBackendMut<'a, BE> {
    <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(src)
}

pub fn mat_znx_backend_ref<'a, BE: Backend<ZnxWord = i64>>(
    src: &'a MatZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> MatZnxBackendRef<'a, BE> {
    <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub fn vec_znx_dft_backend_ref<'a, BE: Backend<ZnxWord = i64>>(src: &'a VecZnxDftOwned<BE>) -> VecZnxDftBackendRef<'a, BE> {
    src.to_backend_ref()
}

pub fn vec_znx_dft_backend_mut<'a, BE: Backend<ZnxWord = i64>>(src: &'a mut VecZnxDftOwned<BE>) -> VecZnxDftBackendMut<'a, BE> {
    src.to_backend_mut()
}

#[allow(dead_code)]
pub fn vec_znx_big_backend_ref<'a, BE: Backend<ZnxWord = i64>>(src: &'a VecZnxBigOwned<BE>) -> VecZnxBigBackendRef<'a, BE> {
    src.to_backend_ref()
}

pub fn vec_znx_big_backend_mut<'a, BE: Backend<ZnxWord = i64>>(src: &'a mut VecZnxBigOwned<BE>) -> VecZnxBigBackendMut<'a, BE> {
    src.to_backend_mut()
}

// pub fn upload_host_glwe<BE>(module: &Module<BE>, src: &GLWE<Vec<u8>, i64>) -> GLWE<BE::OwnedBuf, BE::ZnxWord>
// where
//     BE: Backend<ZnxWord = i64> + TransferFrom<BenchHostBackend>,
//     Module<BE>: ModuleTransfer<BE>,
// {
//     module.upload_glwe::<BenchHostBackend>(src)
// }

// pub fn upload_host_lwe<BE>(module: &Module<BE>, src: &LWE<Vec<u8>, i64>) -> LWE<BE::OwnedBuf, BE::ZnxWord>
// where
//     BE: Backend<ZnxWord = i64> + TransferFrom<BenchHostBackend>,
//     Module<BE>: ModuleTransfer<BE>,
// {
//     module.upload_lwe::<BenchHostBackend>(src)
// }

// pub fn upload_host_lwe_secret<BE>(module: &Module<BE>, src: &LWESecret<Vec<u8>, i64>) -> LWESecret<BE::OwnedBuf, BE::ZnxWord>
// where
//     BE: Backend<ZnxWord = i64> + TransferFrom<BenchHostBackend>,
//     Module<BE>: ModuleTransfer<BE>,
// {
//     module.upload_lwe_secret::<BenchHostBackend>(src)
// }

// pub fn upload_host_glwe_plaintext<BE>(
//     module: &Module<BE>,
//     src: &GLWEPlaintext<Vec<u8>, i64>,
// ) -> GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>
// where
//     BE: Backend<ZnxWord = i64> + TransferFrom<BenchHostBackend>,
//     Module<BE>: ModuleTransfer<BE>,
// {
//     module.upload_glwe_plaintext::<BenchHostBackend>(src)
// }

// pub fn upload_host_ggsw<BE>(module: &Module<BE>, src: &GGSW<Vec<u8>, i64>) -> GGSW<BE::OwnedBuf, BE::ZnxWord>
// where
//     BE: Backend<ZnxWord = i64> + TransferFrom<BenchHostBackend>,
//     Module<BE>: ModuleTransfer<BE>,
// {
//     module.upload_ggsw::<BenchHostBackend>(src)
// }
