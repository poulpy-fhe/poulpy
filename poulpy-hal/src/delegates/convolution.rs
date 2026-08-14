use crate::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, CnvTVecAlloc, CnvTVecBytesOf, Convolution},
    layouts::{
        Backend, CnvDftAccTermPvec, CnvDftAccTermTvec, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecRBackendMut,
        CnvPVecRBackendRef, CnvPVecROwned, CnvTVecLBackendMut, CnvTVecLBackendRef, CnvTVecLOwned, CnvTVecRBackendMut,
        CnvTVecRBackendRef, CnvTVecROwned, Module, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    },
    oep::{HalConvolutionImpl, HalVecZnxDftImpl},
};

impl<BE: Backend> CnvPVecAlloc<BE> for Module<BE> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecLOwned<BE> {
        CnvPVecLOwned::<BE>::alloc(self.n(), cols, size)
    }

    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecROwned<BE> {
        CnvPVecROwned::<BE>::alloc(self.n(), cols, size)
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

impl<BE: Backend> CnvTVecAlloc<BE> for Module<BE> {
    fn cnv_tvec_left_alloc(&self, cols: usize, size: usize) -> CnvTVecLOwned<BE> {
        CnvTVecLOwned::<BE>::alloc(self.n(), cols, size)
    }

    fn cnv_tvec_right_alloc(&self, cols: usize, size: usize) -> CnvTVecROwned<BE> {
        CnvTVecROwned::<BE>::alloc(self.n(), cols, size)
    }
}

impl<BE: Backend> CnvTVecBytesOf for Module<BE> {
    fn bytes_of_cnv_tvec_left(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_tvec_left(self.n(), cols, size)
    }

    fn bytes_of_cnv_tvec_right(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_tvec_right(self.n(), cols, size)
    }
}

/// Forwards the six apply forms of one operand pair to the backend's OEP.
macro_rules! cnv_tier_delegate {
    ($tier:ident, $Tier:ident, $L:ty, $R:ty) => {
        paste::paste! {
            fn [<cnv_apply_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_apply_ $tier _to_dft_tmp_bytes>](self, cnv_offset, res_size, a_size, b_size)
            }

            fn [<cnv_apply_ $tier _to_dft>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                a_col: usize,
                b: &$R,
                b_col: usize,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_apply_ $tier _to_dft>](
                    self, cnv_offset, res, res_col, a, a_col, b, b_col, scratch,
                );
            }

            fn [<cnv_apply_ $tier _to_dft_accumulate_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_apply_ $tier _to_dft_accumulate_tmp_bytes>](
                    self, cnv_offset, res_size, a_size, b_size,
                )
            }

            fn [<cnv_apply_ $tier _to_dft_accumulate>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                a_col: usize,
                b: &$R,
                b_col: usize,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_apply_ $tier _to_dft_accumulate>](
                    self, cnv_offset, res, res_col, a, a_col, b, b_col, scratch,
                );
            }

            fn [<cnv_accumulate_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_accumulate_ $tier _to_dft_tmp_bytes>](
                    self, cnv_offset, res_size, a_size, b_size,
                )
            }

            fn [<cnv_accumulate_ $tier _to_dft>]<'a>(
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                terms: &[[<CnvDftAccTerm $Tier>]<'a, BE>],
                scratch: &mut ScratchArena<'_, BE>,
            ) where
                BE: 'a,
            {
                <BE as HalConvolutionImpl<BE>>::[<cnv_accumulate_ $tier _to_dft>](self, cnv_offset, res, res_col, terms, scratch);
            }

            fn [<cnv_pairwise_apply_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_pairwise_apply_ $tier _to_dft_tmp_bytes>](
                    self, cnv_offset, res_size, a_size, b_size,
                )
            }

            fn [<cnv_pairwise_apply_ $tier _to_dft>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                b: &$R,
                i: usize,
                j: usize,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_pairwise_apply_ $tier _to_dft>](
                    self, cnv_offset, res, res_col, a, b, i, j, scratch,
                );
            }
        }
    };
}

/// Forwards one tier's prepare family to the backend's OEP.
macro_rules! cnv_prepare_delegate {
    ($tier:ident, $L:ty, $R:ty) => {
        paste::paste! {
            fn [<cnv_prepare_left_ $tier _tmp_bytes>](&self, res_size: usize, a_size: usize) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_left_ $tier _tmp_bytes>](self, res_size, a_size)
            }

            fn [<cnv_prepare_left_ $tier>](
                &self,
                res: &mut $L,
                a: &VecZnxBackendRef<'_, BE>,
                mask: i64,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_left_ $tier>](self, res, a, mask, scratch);
            }

            fn [<cnv_prepare_right_ $tier _tmp_bytes>](&self, res_size: usize, a_size: usize) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_right_ $tier _tmp_bytes>](self, res_size, a_size)
            }

            fn [<cnv_prepare_right_ $tier>](
                &self,
                res: &mut $R,
                a: &VecZnxBackendRef<'_, BE>,
                mask: i64,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_right_ $tier>](self, res, a, mask, scratch);
            }

            fn [<cnv_prepare_self_ $tier _tmp_bytes>](&self, res_size: usize, a_size: usize) -> usize {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_self_ $tier _tmp_bytes>](self, res_size, a_size)
            }

            fn [<cnv_prepare_self_ $tier>](
                &self,
                left: &mut $L,
                right: &mut $R,
                a: &VecZnxBackendRef<'_, BE>,
                mask: i64,
                scratch: &mut ScratchArena<'_, BE>,
            ) {
                <BE as HalConvolutionImpl<BE>>::[<cnv_prepare_self_ $tier>](self, left, right, a, mask, scratch);
            }
        }
    };
}

impl<BE: Backend<ZnxWord = i64>> Convolution<BE> for Module<BE>
where
    BE: HalConvolutionImpl<BE> + HalVecZnxDftImpl<BE>,
{
    cnv_prepare_delegate!(pvec, CnvPVecLBackendMut<'_, BE>, CnvPVecRBackendMut<'_, BE>);
    cnv_prepare_delegate!(tvec, CnvTVecLBackendMut<'_, BE>, CnvTVecRBackendMut<'_, BE>);

    fn cnv_by_const_apply_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalConvolutionImpl<BE>>::cnv_by_const_apply_tmp_bytes(self, cnv_offset, res_size, a_size, b_size)
    }

    fn cnv_by_const_apply(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        <BE as HalConvolutionImpl<BE>>::cnv_by_const_apply(self, cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch);
    }

    cnv_tier_delegate!(pvec, Pvec, CnvPVecLBackendRef<'_, BE>, CnvPVecRBackendRef<'_, BE>);
    cnv_tier_delegate!(tvec, Tvec, CnvTVecLBackendRef<'_, BE>, CnvTVecRBackendRef<'_, BE>);
}
