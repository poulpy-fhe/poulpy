use poulpy_hal::layouts::{
    Backend, ScalarZnx, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VmpPMatReborrowBackendMut,
    VmpPMatReborrowBackendRef, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut, vec_znx_backend_mut_from_mut,
    vec_znx_backend_ref_from_mut, vec_znx_backend_ref_from_ref,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, GGLWE, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut,
        GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GGLWEToBackendMut, GGLWEToBackendRef,
        GGSW, GGSWBackendMut, GGSWBackendRef, GGSWInfos, GGSWPrepared, GGSWPreparedBackendMut, GGSWPreparedBackendRef,
        GGSWPreparedToBackendMut, GGSWPreparedToBackendRef, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEBackendMut,
        GLWEBackendRef, GLWEPlaintext, GLWESecret, GLWESecretBackendMut, GLWESecretBackendRef, GLWESecretPrepared,
        GLWESecretPreparedBackendMut, GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendMut,
        GLWESecretPreparedToBackendRef, GLWESecretTensor, GLWESecretTensorBackendMut, GLWESecretTensorBackendRef,
        GLWESecretTensorToBackendMut, GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, GLWETensor,
        GLWEToBackendMut, GLWEToBackendRef, LWE, LWEBackendMut, LWEBackendRef, LWEPlaintext, LWEPlaintextBackendMut,
        LWEPlaintextBackendRef, LWEPlaintextToBackendMut, LWEPlaintextToBackendRef, LWEToBackendMut, LWEToBackendRef, Rank,
        SetBase2k, SetGGLWEInfos, SetK, TorusPrecision,
    },
};

/// Defines a nominal mutable scratch view over a backend-borrowed layout.
///
/// The wrapper gives projected backend buffer types a distinct identity for
/// trait coherence while forwarding the common layout metadata and deref
/// surface to the wrapped value.
#[macro_export]
macro_rules! view_wrapper {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        pub struct $name<'a, BE: ::poulpy_hal::layouts::Backend + 'a> {
            inner: $inner,
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $name<'a, BE> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::core::ops::Deref for $name<'a, BE> {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::core::ops::DerefMut for $name<'a, BE> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::layouts::LWEInfos for $name<'a, BE> {
            fn base2k(&self) -> $crate::layouts::Base2K {
                $crate::layouts::LWEInfos::base2k(&self.inner)
            }

            fn n(&self) -> $crate::layouts::Degree {
                $crate::layouts::LWEInfos::n(&self.inner)
            }

            fn max_size(&self) -> usize {
                $crate::layouts::LWEInfos::max_size(&self.inner)
            }

            fn size(&self) -> usize {
                $crate::layouts::LWEInfos::size(&self.inner)
            }

            fn k(&self) -> $crate::layouts::TorusPrecision {
                $crate::layouts::LWEInfos::k(&self.inner)
            }
        }
    };
}

view_wrapper!(LWEViewMut, LWE<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(LWEPlaintextViewMut, LWEPlaintext<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWEViewRef, GLWE<BE::BufRef<'a>, BE::ZnxWord>);
view_wrapper!(GLWEViewMut, GLWE<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWEPlaintextViewMut, GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWETensorViewMut, GLWETensor<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWESecretViewMut, GLWESecret<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWESecretTensorViewMut, GLWESecretTensor<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GLWESecretPreparedViewMut, GLWESecretPrepared<BE::BufMut<'a>, BE>);
view_wrapper!(GGLWEViewMut, GGLWE<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GGLWEPreparedViewMut, GGLWEPrepared<BE::BufMut<'a>, BE>);
view_wrapper!(GGSWViewMut, GGSW<BE::BufMut<'a>, BE::ZnxWord>);
view_wrapper!(GGSWPreparedViewMut, GGSWPrepared<BE::BufMut<'a>, BE>);

impl<BE: Backend> GGLWEViewMut<'_, BE> {
    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(crate::layouts::gglwe_at_backend_ref_from_mut::<BE>(&self.inner, row, col))
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

macro_rules! impl_set_lwe_infos {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> SetBase2k for $name<'a, BE> {
            fn set_base2k(&mut self, base2k: Base2K) {
                self.inner.set_base2k(base2k);
            }
        }
    };
}

impl_set_lwe_infos!(LWEViewMut);
impl_set_lwe_infos!(GLWEViewMut);
impl_set_lwe_infos!(GLWEPlaintextViewMut);

impl<BE: Backend> crate::layouts::IntPolyInfos for GLWEPlaintextViewMut<'_, BE> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<BE: Backend> crate::layouts::IntPolyInfos for LWEPlaintextViewMut<'_, BE> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<BE: Backend> SetK for GLWEViewMut<'_, BE> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.inner.set_k(k);
    }
}

impl<BE: Backend> SetBase2k for LWEPlaintextViewMut<'_, BE> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.base2k = base2k;
    }
}

/// Forwards [`GLWEInfos`](crate::layouts::GLWEInfos) through a nominal
/// backend view wrapper generated by [`view_wrapper!`](crate::view_wrapper).
#[macro_export]
macro_rules! impl_glwe_infos {
    ($name:ident) => {
        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::layouts::GLWEInfos for $name<'a, BE> {
            fn rank(&self) -> $crate::layouts::Rank {
                $crate::layouts::GLWEInfos::rank(&self.inner)
            }
        }
    };
}

impl_glwe_infos!(GLWEViewMut);
impl_glwe_infos!(GLWEViewRef);
impl_glwe_infos!(GLWEPlaintextViewMut);
impl_glwe_infos!(GLWETensorViewMut);
impl_glwe_infos!(GLWESecretViewMut);
impl_glwe_infos!(GLWESecretTensorViewMut);
impl_glwe_infos!(GLWESecretPreparedViewMut);
impl_glwe_infos!(GGLWEViewMut);
impl_glwe_infos!(GGLWEPreparedViewMut);
impl_glwe_infos!(GGSWViewMut);
impl_glwe_infos!(GGSWPreparedViewMut);

macro_rules! impl_dist {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> GetDistribution for $name<'a, BE> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }

        impl<'a, BE: Backend + 'a> GetDistributionMut for $name<'a, BE> {
            fn dist_mut(&mut self) -> &mut Distribution {
                self.inner.dist_mut()
            }
        }
    };
}

impl_dist!(GLWESecretTensorViewMut);
impl_dist!(GLWESecretPreparedViewMut);

impl<BE: Backend> GetDistribution for GLWESecretViewMut<'_, BE> {
    fn dist(&self) -> &Distribution {
        self.inner.dist()
    }
}

impl<BE: Backend> GGLWEInfos for GGLWEViewMut<'_, BE> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend> GGLWEInfos for GGLWEPreparedViewMut<'_, BE> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend> SetGGLWEInfos for GGLWEViewMut<'_, BE> {
    fn set_dsize(&mut self, dsize: usize) {
        self.inner.dsize = dsize.into();
    }
}

impl<BE: Backend> GGSWInfos for GGSWViewMut<'_, BE> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }
}

impl<BE: Backend> GGSWInfos for GGSWPreparedViewMut<'_, BE> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }
}

impl<BE: Backend> LWEToBackendRef<BE> for LWEViewMut<'_, BE> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE> {
        LWE {
            base2k: self.inner.base2k,
            k: self.inner.k,
            body: vec_znx_backend_ref_from_mut::<BE>(&self.inner.body),
            mask: vec_znx_backend_ref_from_mut::<BE>(&self.inner.mask),
        }
    }
}

impl<BE: Backend> LWEToBackendMut<BE> for LWEViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE> {
        let base2k = self.inner.base2k;
        let k = self.inner.k;
        let body = vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.body);
        let mask = vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.mask);
        LWE { base2k, k, body, mask }
    }
}

impl<BE: Backend> LWEPlaintextToBackendRef<BE> for LWEPlaintextViewMut<'_, BE> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            k: self.inner.k,
            data: vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

impl<BE: Backend> LWEPlaintextToBackendMut<BE> for LWEPlaintextViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            k: self.inner.k,
            data: vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        }
    }
}

macro_rules! impl_glwe_to_backend {
    ($name:ident, |$this:ident| $canonical:expr, |$this_mut:ident, $flag:ident| $set_canonical:expr) => {
        impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for $name<'a, BE> {
            fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
                let $this = self;
                GLWE {
                    base2k: self.inner.base2k,
                    k: self.inner.k,
                    canonical: $canonical,
                    data: vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
                }
            }
        }

        impl<'a, BE: Backend + 'a> GLWEToBackendMut<BE> for $name<'a, BE> {
            fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
                let $this = &*self;
                let canonical = $canonical;
                GLWE {
                    base2k: self.inner.base2k,
                    k: self.inner.k,
                    canonical,
                    data: vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
                }
            }

            fn set_canonical(&mut self, $flag: bool) {
                let $this_mut = self;
                $set_canonical
            }
        }
    };
}

impl_glwe_to_backend!(GLWEViewMut, |this| this.inner.canonical, |this, canonical| this
    .inner
    .canonical =
    canonical);
impl_glwe_to_backend!(GLWEPlaintextViewMut, |_this| true, |_this, _canonical| ());
impl_glwe_to_backend!(GLWETensorViewMut, |_this| true, |_this, _canonical| ());

impl<BE: Backend> GLWEToBackendRef<BE> for GLWEViewRef<'_, BE> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.inner.base2k,
            k: self.inner.k,
            canonical: self.inner.canonical,
            data: vec_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        }
    }
}

impl<BE: Backend> GLWESecretToBackendRef<BE> for GLWESecretViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(
                BE::view_ref_mut(&self.inner.data.data),
                self.inner.data.n(),
                self.inner.data.cols(),
            ),
        }
    }
}

impl<BE: Backend> GLWESecretToBackendMut<BE> for GLWESecretViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<BE: Backend> GLWESecretTensorToBackendRef<BE> for GLWESecretTensorViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE> {
        GLWESecretTensor {
            dist: self.inner.dist,
            rank: self.inner.rank,
            data: ScalarZnx::from_data(
                BE::view_ref_mut(&self.inner.data.data),
                self.inner.data.n(),
                self.inner.data.cols(),
            ),
        }
    }
}

impl<BE: Backend> GLWESecretTensorToBackendMut<BE> for GLWESecretTensorViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecretTensor {
            dist: self.inner.dist,
            rank: self.inner.rank,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<BE: Backend> GLWESecretPreparedToBackendRef<BE> for GLWESecretPreparedViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWESecretPreparedBackendRef<'_, BE> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend> GLWESecretPreparedToBackendMut<BE> for GLWESecretPreparedViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretPreparedBackendMut<'_, BE> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for GGLWEViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGLWEToBackendMut<BE> for GGLWEViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<BE: Backend> GGLWEPreparedToBackendRef<BE> for GGLWEPreparedViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, BE> {
        GGLWEPrepared {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            dnum: self.inner.dnum,
            stride: self.inner.stride,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend> GGLWEPreparedToBackendMut<BE> for GGLWEPreparedViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, BE> {
        GGLWEPrepared {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            dnum: self.inner.dnum,
            stride: self.inner.stride,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<BE: Backend> GGSWToBackendRef<BE> for GGSWViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWToBackendMut<BE> for GGSWViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWPreparedToBackendRef<BE> for GGSWPreparedViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, BE> {
        GGSWPrepared {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend> GGSWPreparedToBackendMut<BE> for GGSWPreparedViewMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, BE> {
        GGSWPrepared {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}
