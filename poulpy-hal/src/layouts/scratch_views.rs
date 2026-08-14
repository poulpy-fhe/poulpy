use std::ops::{Deref, DerefMut};

use crate::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLReborrowBackendMut, CnvPVecLReborrowBackendRef,
    CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CnvPVecRReborrowBackendMut,
    CnvPVecRReborrowBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, MatZnxBackendMut, MatZnxBackendRef, MatZnxInfos,
    MatZnxToBackendMut, MatZnxToBackendRef, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScalarZnxToBackendMut,
    ScalarZnxToBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef,
    SvpPPolToBackendMut, SvpPPolToBackendRef, SvpTPolBackendMut, SvpTPolBackendRef, SvpTPolReborrowBackendMut,
    SvpTPolReborrowBackendRef, SvpTPolToBackendMut, SvpTPolToBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
    VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxBigToBackendMut,
    VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendMut, VecZnxDftReborrowBackendRef,
    VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef,
    VecZnxToBackendMut, VecZnxToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatReborrowBackendMut,
    VmpPMatReborrowBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef, VmpTMatBackendMut, VmpTMatBackendRef,
    VmpTMatReborrowBackendMut, VmpTMatReborrowBackendRef, VmpTMatToBackendMut, VmpTMatToBackendRef, ZnxInfos,
    mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut,
};

macro_rules! view_wrapper {
    ($name:ident, $inner:ty) => {
        pub struct $name<'a, B: Backend + 'a> {
            inner: $inner,
        }

        impl<'a, B: Backend + 'a> $name<'a, B> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, B: Backend + 'a> Deref for $name<'a, B> {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, B: Backend + 'a> DerefMut for $name<'a, B> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, B: Backend + 'a> ZnxInfos for $name<'a, B> {
            fn n(&self) -> usize {
                self.inner.n()
            }

            fn size(&self) -> usize {
                self.inner.size()
            }

            fn poly_count(&self) -> usize {
                self.inner.poly_count()
            }
        }
    };
}

/// Wrapper over a vector-shaped layout: forwards [`VecZnxInfos`] on top of the
/// common shape.
macro_rules! vec_view_wrapper {
    ($name:ident, $inner:ty) => {
        view_wrapper!($name, $inner);

        impl<'a, B: Backend + 'a> VecZnxInfos for $name<'a, B> {
            fn cols(&self) -> usize {
                self.inner.cols()
            }
        }
    };
}

/// Wrapper over a matrix-shaped layout: forwards [`MatZnxInfos`] on top of the
/// common shape.
macro_rules! mat_view_wrapper {
    ($name:ident, $inner:ty) => {
        view_wrapper!($name, $inner);

        impl<'a, B: Backend + 'a> MatZnxInfos for $name<'a, B> {
            fn rows(&self) -> usize {
                self.inner.rows()
            }

            fn cols_in(&self) -> usize {
                self.inner.cols_in()
            }

            fn cols_out(&self) -> usize {
                self.inner.cols_out()
            }
        }
    };
}

vec_view_wrapper!(CnvPVecLViewMut, CnvPVecLBackendMut<'a, B>);
vec_view_wrapper!(CnvPVecRViewMut, CnvPVecRBackendMut<'a, B>);
mat_view_wrapper!(MatZnxViewMut, MatZnxBackendMut<'a, B>);
vec_view_wrapper!(ScalarZnxViewMut, ScalarZnxBackendMut<'a, B>);
vec_view_wrapper!(SvpPPolViewMut, SvpPPolBackendMut<'a, B>);
vec_view_wrapper!(SvpTPolViewMut, SvpTPolBackendMut<'a, B>);
vec_view_wrapper!(VecZnxViewMut, VecZnxBackendMut<'a, B>);
vec_view_wrapper!(VecZnxBigViewMut, VecZnxBigBackendMut<'a, B>);
vec_view_wrapper!(VecZnxDftViewMut, VecZnxDftBackendMut<'a, B>);
mat_view_wrapper!(VmpPMatViewMut, VmpPMatBackendMut<'a, B>);
mat_view_wrapper!(VmpTMatViewMut, VmpTMatBackendMut<'a, B>);

impl<'a, B: Backend + 'a> CnvPVecLToBackendRef<B> for CnvPVecLViewMut<'a, B> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> CnvPVecLToBackendMut<B> for CnvPVecLViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> CnvPVecRToBackendRef<B> for CnvPVecRViewMut<'a, B> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> CnvPVecRToBackendMut<B> for CnvPVecRViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> MatZnxToBackendRef<B> for MatZnxViewMut<'a, B> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_mut::<B>(&self.inner)
    }
}

impl<'a, B: Backend + 'a> MatZnxToBackendMut<B> for MatZnxViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        mat_znx_backend_mut_from_mut::<B>(&mut self.inner)
    }
}

impl<'a, B: Backend + 'a> ScalarZnxToBackendRef<B> for ScalarZnxViewMut<'a, B> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx::from_data(B::view_ref_mut(&self.inner.data), self.inner.n(), self.inner.cols())
    }
}

impl<'a, B: Backend + 'a> ScalarZnxToBackendMut<B> for ScalarZnxViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        let n = self.inner.n();
        let cols = self.inner.cols();
        ScalarZnx::from_data(B::view_mut_ref(&mut self.inner.data), n, cols)
    }
}

impl<'a, B: Backend + 'a> SvpPPolToBackendRef<B> for SvpPPolViewMut<'a, B> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> SvpPPolToBackendMut<B> for SvpPPolViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> SvpTPolToBackendRef<B> for SvpTPolViewMut<'a, B> {
    fn to_backend_ref(&self) -> SvpTPolBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> SvpTPolToBackendMut<B> for SvpTPolViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VecZnxToBackendRef<B> for VecZnxViewMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        <VecZnx<B::BufMut<'a>, B::ZnxWord> as VecZnxReborrowBackendRef<B>>::reborrow_backend_ref(&self.inner)
    }
}

impl<'a, B: Backend + 'a> VecZnxToBackendMut<B> for VecZnxViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        <VecZnx<B::BufMut<'a>, B::ZnxWord> as VecZnxReborrowBackendMut<B>>::reborrow_backend_mut(&mut self.inner)
    }
}

impl<'a, B: Backend + 'a> VecZnxBigToBackendRef<B> for VecZnxBigViewMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VecZnxBigToBackendMut<B> for VecZnxBigViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VecZnxDftToBackendRef<B> for VecZnxDftViewMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VecZnxDftToBackendMut<B> for VecZnxDftViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VmpPMatToBackendRef<B> for VmpPMatViewMut<'a, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VmpPMatToBackendMut<B> for VmpPMatViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VmpTMatToBackendRef<B> for VmpTMatViewMut<'a, B> {
    fn to_backend_ref(&self) -> VmpTMatBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VmpTMatToBackendMut<B> for VmpTMatViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}
