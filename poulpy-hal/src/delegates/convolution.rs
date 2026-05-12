use crate::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution},
    layouts::{
        Backend, CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef, Module,
        ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    },
    oep::HalConvolutionImpl,
};

macro_rules! impl_convolution_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE: Backend> $trait for Module<BE>
        where
            BE: HalConvolutionImpl<BE>,
        {
            $($body)+
        }
    };
}

impl<BE: Backend> CnvPVecAlloc<BE> for Module<BE> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecL<BE::OwnedBuf, BE> {
        CnvPVecL::alloc(self.n(), cols, size)
    }

    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecR<BE::OwnedBuf, BE> {
        CnvPVecR::alloc(self.n(), cols, size)
    }
}

impl<BE: Backend> CnvPVecBytesOf for Module<BE> {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_left(self.n(), cols, size)
    }

    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_right(self.n(), cols, size)
    }
}

impl_convolution_delegate!(
    Convolution<BE>,
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_left_tmp_bytes(self, res_size, a_size)
    },
    fn cnv_prepare_left<'s, 'r>(
        &self,
        res: &mut CnvPVecLBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_left(self, res, a, mask, scratch);
    },
    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_right_tmp_bytes(self, res_size, a_size)
    },
    fn cnv_prepare_right<'s, 'r>(
        &self,
        res: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_right(self, res, a, mask, scratch);
    },
    fn cnv_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_apply_dft_tmp_bytes(self, cnv_offset, res_size, a_size, b_size)
    },
    fn cnv_by_const_apply_tmp_bytes(&self, res_size: usize, cnv_offset: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_by_const_apply_tmp_bytes(self, res_size, cnv_offset, a_size, b_size)
    },
    fn cnv_by_const_apply<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_by_const_apply(self, cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    },
    fn cnv_apply_dft<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_apply_dft(self, cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    },
    fn cnv_pairwise_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_pairwise_apply_dft_tmp_bytes(self, cnv_offset, res_size, a_size, b_size)
    },
    fn cnv_pairwise_apply_dft<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_pairwise_apply_dft(self, cnv_offset, res, res_col, a, b, i, j, scratch)
    },
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_self_tmp_bytes(self, res_size, a_size)
    },
    fn cnv_prepare_self<'s, 'l, 'r>(
        &self,
        left: &mut CnvPVecLBackendMut<'l, BE>,
        right: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_prepare_self(self, left, right, a, mask, scratch)
    }
);
