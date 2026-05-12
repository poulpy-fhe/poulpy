use std::ops::{Deref, DerefMut};

use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLReborrowBackendMut, CnvPVecLReborrowBackendRef,
    CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef, CnvPVecRReborrowBackendMut,
    CnvPVecRReborrowBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, MatZnx, MatZnxBackendMut, MatZnxBackendRef,
    MatZnxToBackendMut, MatZnxToBackendRef, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScalarZnxToBackendMut,
    ScalarZnxToBackendRef, SvpPPol, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef,
    SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
    VecZnxBigBackendRef, VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
    VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendMut, VecZnxDftReborrowBackendRef,
    VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef, VecZnxToBackendMut,
    VecZnxToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatReborrowBackendMut, VmpPMatReborrowBackendRef,
    VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxInfos, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut,
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
            fn cols(&self) -> usize {
                self.inner.cols()
            }

            fn rows(&self) -> usize {
                self.inner.rows()
            }

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

view_wrapper!(CnvPVecLViewMut, CnvPVecL<B::BufMut<'a>, B>);
view_wrapper!(CnvPVecRViewMut, CnvPVecR<B::BufMut<'a>, B>);
view_wrapper!(MatZnxViewMut, MatZnx<B::BufMut<'a>>);
view_wrapper!(ScalarZnxViewMut, ScalarZnx<B::BufMut<'a>>);
view_wrapper!(SvpPPolViewMut, SvpPPol<B::BufMut<'a>, B>);
view_wrapper!(VecZnxViewMut, VecZnx<B::BufMut<'a>>);
view_wrapper!(VecZnxBigViewMut, VecZnxBig<B::BufMut<'a>, B>);
view_wrapper!(VecZnxDftViewMut, VecZnxDft<B::BufMut<'a>, B>);
view_wrapper!(VmpPMatViewMut, VmpPMat<B::BufMut<'a>, B>);

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

impl<'a, B: Backend + 'a> VecZnxToBackendRef<B> for VecZnxViewMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        <VecZnx<B::BufMut<'a>> as VecZnxReborrowBackendRef<B>>::reborrow_backend_ref(&self.inner)
    }
}

impl<'a, B: Backend + 'a> VecZnxToBackendMut<B> for VecZnxViewMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        <VecZnx<B::BufMut<'a>> as VecZnxReborrowBackendMut<B>>::reborrow_backend_mut(&mut self.inner)
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
