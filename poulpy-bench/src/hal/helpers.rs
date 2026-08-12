//! Host-buffer allocation, upload, and backend-view helpers used only by the
//! HAL runner functions in this module tree — not part of the crate's public
//! API.

use poulpy_hal::layouts::{
    Backend, CnvPVecL, CnvPVecR, DataView, MatZnx, MatZnxBackendRef, MatZnxToBackendRef, ScalarZnx, ScalarZnxBackendRef,
    ScalarZnxToBackendRef, SvpPPol, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
    VecZnxBigToBackendMut, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    VecZnxToBackendMut, VecZnxToBackendRef, VmpPMat,
};
use poulpy_hal::source::Source;
use rand::Rng;

fn random_aligned_host_bytes(len: usize, source: &mut Source) -> Vec<u8> {
    let mut bytes = poulpy_hal::alloc_aligned_custom::<u8>(len, poulpy_hal::DEFAULTALIGN);
    source.fill_bytes(&mut bytes);
    bytes
}

pub(super) fn upload_host_vec_znx<BE: Backend>(src: &VecZnx<Vec<u8>>) -> VecZnx<BE::OwnedBuf> {
    VecZnx::from_data_with_max_size(
        BE::from_host_bytes(src.data()),
        src.n(),
        src.cols(),
        src.size(),
        src.max_size(),
    )
}

pub(super) fn upload_host_scalar_znx<BE: Backend>(src: &ScalarZnx<Vec<u8>>) -> ScalarZnx<BE::OwnedBuf> {
    ScalarZnx::from_data(BE::from_host_bytes(src.data()), src.n(), src.cols())
}

pub(super) fn upload_host_mat_znx<BE: Backend>(src: &MatZnx<Vec<u8>>) -> MatZnx<BE::OwnedBuf> {
    MatZnx::from_data(
        BE::from_host_bytes(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

pub(super) fn random_host_scalar_znx(n: usize, cols: usize, source: &mut Source) -> ScalarZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n, cols), source);
    ScalarZnx::from_bytes(n, cols, bytes)
}

pub(super) fn random_host_vec_znx(n: usize, cols: usize, size: usize, source: &mut Source) -> VecZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size), source);
    VecZnx::from_bytes(n, cols, size, bytes)
}

pub(super) fn random_host_mat_znx(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> MatZnx<Vec<u8>> {
    let bytes = random_aligned_host_bytes(MatZnx::<Vec<u8>>::bytes_of(n, rows, cols_in, cols_out, size), source);
    MatZnx::from_bytes(n, rows, cols_in, cols_out, size, bytes)
}

pub(super) fn random_backend_vec_znx_dft<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> VecZnxDft<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vec_znx_dft(n, cols, size)];
    source.fill_bytes(&mut bytes);
    VecZnxDft::from_bytes(n, cols, size, bytes)
}

pub(super) fn random_backend_svp_ppol<BE: Backend>(n: usize, cols: usize, source: &mut Source) -> SvpPPol<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_svp_ppol(n, cols)];
    source.fill_bytes(&mut bytes);
    SvpPPol::from_data(BE::from_host_bytes(&bytes), n, cols)
}

pub(super) fn random_backend_vmp_pmat<BE: Backend>(
    n: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    source: &mut Source,
) -> VmpPMat<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size)];
    source.fill_bytes(&mut bytes);
    VmpPMat::from_data(BE::from_host_bytes(&bytes), n, rows, cols_in, cols_out, size)
}

pub(super) fn random_backend_cnv_pvec_left<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecL<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_left(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecL::from_bytes(n, cols, size, bytes)
}

pub(super) fn random_backend_cnv_pvec_right<BE: Backend>(
    n: usize,
    cols: usize,
    size: usize,
    source: &mut Source,
) -> CnvPVecR<BE::OwnedBuf, BE> {
    let mut bytes = vec![0u8; BE::bytes_of_cnv_pvec_right(n, cols, size)];
    source.fill_bytes(&mut bytes);
    CnvPVecR::from_bytes(n, cols, size, bytes)
}

pub(super) fn scalar_znx_backend_ref<'a, BE: Backend>(src: &'a ScalarZnx<BE::OwnedBuf>) -> ScalarZnxBackendRef<'a, BE> {
    <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub(super) fn vec_znx_backend_ref<'a, BE: Backend>(src: &'a VecZnx<BE::OwnedBuf>) -> VecZnxBackendRef<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub(super) fn vec_znx_backend_mut<'a, BE: Backend>(src: &'a mut VecZnx<BE::OwnedBuf>) -> VecZnxBackendMut<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(src)
}

pub(super) fn mat_znx_backend_ref<'a, BE: Backend>(src: &'a MatZnx<BE::OwnedBuf>) -> MatZnxBackendRef<'a, BE> {
    <MatZnx<BE::OwnedBuf> as MatZnxToBackendRef<BE>>::to_backend_ref(src)
}

pub(super) fn vec_znx_dft_backend_ref<'a, BE: Backend>(src: &'a VecZnxDft<BE::OwnedBuf, BE>) -> VecZnxDftBackendRef<'a, BE> {
    <VecZnxDft<BE::OwnedBuf, BE> as VecZnxDftToBackendRef<BE>>::to_backend_ref(src)
}

pub(super) fn vec_znx_dft_backend_mut<'a, BE: Backend>(src: &'a mut VecZnxDft<BE::OwnedBuf, BE>) -> VecZnxDftBackendMut<'a, BE> {
    <VecZnxDft<BE::OwnedBuf, BE> as VecZnxDftToBackendMut<BE>>::to_backend_mut(src)
}

pub(super) fn vec_znx_big_backend_mut<'a, BE: Backend>(src: &'a mut VecZnxBig<BE::OwnedBuf, BE>) -> VecZnxBigBackendMut<'a, BE> {
    <VecZnxBig<BE::OwnedBuf, BE> as VecZnxBigToBackendMut<BE>>::to_backend_mut(src)
}
