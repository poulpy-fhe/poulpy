use crate::layouts::{
    Backend, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned, VecZnxBackendRef, VecZnxDftBackendMut,
    VecZnxDftBackendRef,
};

/// Allocates as [crate::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
pub trait SvpPrepare<B: Backend> {
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one prepared scalar polynomial column into another.
pub trait SvpPPolCopyBackend<B: Backend> {
    fn svp_ppol_copy_backend(
        &self,
        res: &mut SvpPPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDft<B: Backend> {
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDftToDft<B: Backend> {
    fn svp_apply_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    );
}

/// Apply a scalar-vector product between `res[res_col]` and `a[a_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDftToDftAssign<B: Backend> {
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
