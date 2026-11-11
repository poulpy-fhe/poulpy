use std::ops::{Deref, DerefMut};

use crate::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLReborrowBackendMut, CnvPVecLReborrowBackendRef,
    CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CnvPVecRReborrowBackendMut,
    CnvPVecRReborrowBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, MatZnxBackendMut, MatZnxBackendRef, MatZnxInfos,
    MatZnxToBackendMut, MatZnxToBackendRef, Ring, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScalarZnxToBackendMut,
    ScalarZnxToBackendRef, Standard, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef,
    SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
    VecZnxBigBackendRef, VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
    VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendMut, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut,
    VecZnxDftToBackendRef, VecZnxInfos, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef, VecZnxToBackendMut,
    VecZnxToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatReborrowBackendMut, VmpPMatReborrowBackendRef,
    VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxInfos, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut,
};

macro_rules! view_wrapper {
    ($name:ident, $inner:ty) => {
        pub struct $name<'a, B: Backend + 'a, Rg: Ring = Standard> {
            inner: $inner,
        }

        impl<'a, B: Backend + 'a, Rg: Ring> $name<'a, B, Rg> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, B: Backend + 'a, Rg: Ring> Deref for $name<'a, B, Rg> {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, B: Backend + 'a, Rg: Ring> DerefMut for $name<'a, B, Rg> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, B: Backend + 'a, Rg: Ring> ZnxInfos for $name<'a, B, Rg> {
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

        impl<'a, B: Backend + 'a, Rg: Ring> VecZnxInfos for $name<'a, B, Rg> {
            fn cols(&self) -> usize {
                self.inner.cols()
            }
            fn n_full(&self) -> usize {
                self.inner.n_full()
            }
            fn coeff_offset(&self) -> usize {
                self.inner.coeff_offset()
            }
            fn limb_offset(&self) -> usize {
                self.inner.limb_offset()
            }
            fn limb_step(&self) -> usize {
                self.inner.limb_step()
            }
        }
    };
}

/// Wrapper over a matrix-shaped layout: forwards [`MatZnxInfos`] on top of the
/// common shape.
macro_rules! mat_view_wrapper {
    ($name:ident, $inner:ty) => {
        view_wrapper!($name, $inner);

        impl<'a, B: Backend + 'a, Rg: Ring> MatZnxInfos for $name<'a, B, Rg> {
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

vec_view_wrapper!(CnvPVecLViewMut, CnvPVecLBackendMut<'a, B, Rg>);
vec_view_wrapper!(CnvPVecRViewMut, CnvPVecRBackendMut<'a, B, Rg>);
mat_view_wrapper!(MatZnxViewMut, MatZnxBackendMut<'a, B, Rg>);
vec_view_wrapper!(ScalarZnxViewMut, ScalarZnxBackendMut<'a, B, Rg>);
vec_view_wrapper!(SvpPPolViewMut, SvpPPolBackendMut<'a, B, Rg>);
vec_view_wrapper!(VecZnxViewMut, VecZnxBackendMut<'a, B, Rg>);
vec_view_wrapper!(VecZnxBigViewMut, VecZnxBigBackendMut<'a, B, Rg>);
vec_view_wrapper!(VecZnxDftViewMut, VecZnxDftBackendMut<'a, B, Rg>);
mat_view_wrapper!(VmpPMatViewMut, VmpPMatBackendMut<'a, B, Rg>);

impl<B: Backend, Rg: Ring> CnvPVecLToBackendRef<B, Rg> for CnvPVecLViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> CnvPVecLToBackendMut<B, Rg> for CnvPVecLViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}

impl<B: Backend, Rg: Ring> CnvPVecRToBackendRef<B, Rg> for CnvPVecRViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> CnvPVecRToBackendMut<B, Rg> for CnvPVecRViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}

impl<B: Backend, Rg: Ring> MatZnxToBackendRef<B, Rg> for MatZnxViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B, Rg> {
        mat_znx_backend_ref_from_mut::<B, Rg>(&self.inner)
    }
}

impl<B: Backend, Rg: Ring> MatZnxToBackendMut<B, Rg> for MatZnxViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B, Rg> {
        mat_znx_backend_mut_from_mut::<B, Rg>(&mut self.inner)
    }
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendRef<B, Rg> for ScalarZnxViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B, Rg> {
        ScalarZnx::from_data(B::view_ref_mut(&self.inner.data), self.inner.n(), self.inner.cols())
    }
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendMut<B, Rg> for ScalarZnxViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B, Rg> {
        let n = self.inner.n();
        let cols = self.inner.cols();
        ScalarZnx::from_data(B::view_mut_ref(&mut self.inner.data), n, cols)
    }
}

impl<B: Backend, Rg: Ring> SvpPPolToBackendRef<B, Rg> for SvpPPolViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> SvpPPolToBackendMut<B, Rg> for SvpPPolViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a, Rg: Ring> VecZnxToBackendRef<B, Rg> for VecZnxViewMut<'a, B, Rg> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B, Rg> {
        <VecZnx<B::BufMut<'a>, B::ZnxWord, Rg> as VecZnxReborrowBackendRef<B, Rg>>::reborrow_backend_ref(&self.inner)
    }
}

impl<'a, B: Backend + 'a, Rg: Ring> VecZnxToBackendMut<B, Rg> for VecZnxViewMut<'a, B, Rg> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, Rg> {
        <VecZnx<B::BufMut<'a>, B::ZnxWord, Rg> as VecZnxReborrowBackendMut<B, Rg>>::reborrow_backend_mut(&mut self.inner)
    }
}

impl<B: Backend, Rg: Ring> VecZnxBigToBackendRef<B, Rg> for VecZnxBigViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> VecZnxBigToBackendMut<B, Rg> for VecZnxBigViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}

impl<B: Backend, Rg: Ring> VecZnxDftToBackendRef<B, Rg> for VecZnxDftViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> VecZnxDftToBackendMut<B, Rg> for VecZnxDftViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendRef<B, Rg> for VmpPMatViewMut<'_, B, Rg> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg> {
        self.inner.reborrow_backend_ref()
    }
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendMut<B, Rg> for VmpPMatViewMut<'_, B, Rg> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg> {
        self.inner.reborrow_backend_mut()
    }
}
