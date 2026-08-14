use std::marker::PhantomData;

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, HostDataRef, VecZnxInfos, ZnxInfos, ZnxView};

cnv_vec_family!(
    CnvPVec,
    cnv_pvec,
    "Packed (cold-prep) left operand for bivariate convolution.\n\n\
     Holds a polynomial vector in the prepared representation named by the\n\
     `DftWord` type `W`. More expensive to build than [`CnvTVecL`](crate::layouts::CnvTVecL),\n\
     and optimized for amortized repeated apply. Created via\n\
     [`Convolution::cnv_prepare_left_pvec`](crate::api::Convolution::cnv_prepare_left_pvec).",
    "Packed (cold-prep) right operand for bivariate convolution.\n\n\
     Holds a polynomial vector in the prepared representation named by the\n\
     `DftWord` type `W`. More expensive to build than [`CnvTVecR`](crate::layouts::CnvTVecR),\n\
     and optimized for amortized repeated apply. Created via\n\
     [`Convolution::cnv_prepare_right_pvec`](crate::api::Convolution::cnv_prepare_right_pvec)."
);

cnv_vec_family!(
    CnvTVec,
    cnv_tvec,
    "Transformed (hot-prep) left operand for bivariate convolution.\n\n\
     The cheap-to-build prepared form, meant for short reuse or one-shot use;\n\
     [`CnvPVecL`](crate::layouts::CnvPVecL) is the packed form. The two are distinct\n\
     types even where a backend gives them the same physical storage shape.\n\
     Created via [`Convolution::cnv_prepare_left_tvec`](crate::api::Convolution::cnv_prepare_left_tvec).",
    "Transformed (hot-prep) right operand for bivariate convolution.\n\n\
     The cheap-to-build prepared form, meant for short reuse or one-shot use;\n\
     [`CnvPVecR`](crate::layouts::CnvPVecR) is the packed form. The two are distinct\n\
     types even where a backend gives them the same physical storage shape.\n\
     Created via [`Convolution::cnv_prepare_right_tvec`](crate::api::Convolution::cnv_prepare_right_tvec)."
);

/// One `(left, right)` operand pair of a fused convolution accumulation.
///
/// Consumed by `cnv_accumulate_<tier>_to_dft`, which overwrites the destination
/// column with the sum of the bivariate convolutions of all terms. Both operands
/// are of the same prep tier.
pub struct CnvDftAccTerm<'a, BE: Backend + 'a, L, R> {
    /// Left operand.
    pub a: L,
    /// Column of `a` to convolve.
    pub a_col: usize,
    /// Right operand.
    pub b: R,
    /// Column of `b` to convolve.
    pub b_col: usize,
    _phantom: PhantomData<&'a BE>,
}

impl<'a, BE: Backend + 'a, L, R> CnvDftAccTerm<'a, BE, L, R> {
    pub fn new(a: L, a_col: usize, b: R, b_col: usize) -> Self {
        Self {
            a,
            a_col,
            b,
            b_col,
            _phantom: PhantomData,
        }
    }
}

/// Accumulation term for the packed (cold-prep) tier.
pub type CnvDftAccTermPvec<'a, BE> = CnvDftAccTerm<'a, BE, CnvPVecLBackendRef<'a, BE>, CnvPVecRBackendRef<'a, BE>>;
/// Accumulation term for the transformed (hot-prep) tier.
pub type CnvDftAccTermTvec<'a, BE> = CnvDftAccTerm<'a, BE, CnvTVecLBackendRef<'a, BE>, CnvTVecRBackendRef<'a, BE>>;
