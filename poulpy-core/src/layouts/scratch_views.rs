use poulpy_hal::layouts::{
    Backend, Ring, ScalarZnx, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VmpPMatReborrowBackendMut,
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
        GLWEBackendRef, GLWEPlaintext, GLWEPlaintextInfos, GLWEPlaintextMeta, GLWESecret, GLWESecretBackendMut,
        GLWESecretBackendRef, GLWESecretPrepared, GLWESecretPreparedBackendMut, GLWESecretPreparedBackendRef,
        GLWESecretPreparedToBackendMut, GLWESecretPreparedToBackendRef, GLWESecretTensor, GLWESecretTensorBackendMut,
        GLWESecretTensorBackendRef, GLWESecretTensorToBackendMut, GLWESecretTensorToBackendRef, GLWESecretToBackendMut,
        GLWESecretToBackendRef, GLWETensor, GLWEToBackendMut, GLWEToBackendRef, LWE, LWEBackendMut, LWEBackendRef, LWEPlaintext,
        LWEPlaintextBackendMut, LWEPlaintextBackendRef, LWEPlaintextToBackendMut, LWEPlaintextToBackendRef, LWEToBackendMut,
        LWEToBackendRef, Rank, SetBase2k, SetGGLWEInfos, SetGLWEPlaintextInfos, SetK, TorusPrecision,
    },
};

/// Defines a nominal mutable scratch view over a backend-borrowed layout.
///
/// The wrapper gives projected backend buffer types a distinct identity for
/// trait coherence while forwarding the common layout metadata and deref
/// surface to the wrapped value.
///
/// The generated type is `$name<'a, BE: Backend + 'a, Rg: Ring = Standard>`, generic over the
/// operand ring. `$inner` names the wrapped layout through these parameters and must use `Rg`,
/// as in `GLWE<BE::BufMut<'a>, BE::ZnxWord, Rg>`.
#[macro_export]
macro_rules! view_wrapper {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        pub struct $name<
            'a,
            BE: ::poulpy_hal::layouts::Backend + 'a,
            Rg: ::poulpy_hal::layouts::Ring = ::poulpy_hal::layouts::Standard,
        > {
            inner: $inner,
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a, Rg: ::poulpy_hal::layouts::Ring> $name<'a, BE, Rg> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a, Rg: ::poulpy_hal::layouts::Ring> ::core::ops::Deref
            for $name<'a, BE, Rg>
        {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a, Rg: ::poulpy_hal::layouts::Ring> ::core::ops::DerefMut
            for $name<'a, BE, Rg>
        {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a, Rg: ::poulpy_hal::layouts::Ring> $crate::layouts::LWEInfos
            for $name<'a, BE, Rg>
        {
            fn noise(&self) -> Option<$crate::ComponentNoise> { $crate::layouts::LWEInfos::noise(&self.inner) }

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

view_wrapper!(LWEViewMut, LWE<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(LWEPlaintextViewMut, LWEPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWEViewRef, GLWE<BE::BufRef<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWEViewMut, GLWE<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWEPlaintextViewMut, GLWEPlaintext<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWETensorViewMut, GLWETensor<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWESecretViewMut, GLWESecret<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWESecretTensorViewMut, GLWESecretTensor<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GLWESecretPreparedViewMut, GLWESecretPrepared<BE::BufMut<'a>, BE, Rg>);
view_wrapper!(GGLWEViewMut, GGLWE<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GGLWEPreparedViewMut, GGLWEPrepared<BE::BufMut<'a>, BE, Rg>);
view_wrapper!(GGSWViewMut, GGSW<BE::BufMut<'a>, BE::ZnxWord, Rg>);
view_wrapper!(GGSWPreparedViewMut, GGSWPrepared<BE::BufMut<'a>, BE, Rg>);

impl<BE: Backend, Rg: Ring> GGLWEViewMut<'_, BE, Rg> {
    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE, Rg> {
        GLWEViewRef::from_inner(crate::layouts::gglwe_at_backend_ref_from_mut::<BE, Rg>(&self.inner, row, col))
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE, Rg> {
        GLWEViewMut::from_inner(crate::layouts::gglwe_at_backend_mut_from_mut::<BE, Rg>(
            &mut self.inner,
            row,
            col,
        ))
    }
}

macro_rules! impl_set_lwe_infos {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a, Rg: Ring> SetBase2k for $name<'a, BE, Rg> {
            fn set_base2k(&mut self, base2k: Base2K) {
                self.inner.set_base2k(base2k);
            }
        }
    };
}

impl_set_lwe_infos!(LWEViewMut);
impl_set_lwe_infos!(GLWEViewMut);
impl_set_lwe_infos!(GLWEPlaintextViewMut);

impl<BE: Backend, Rg: Ring> crate::layouts::IntPolyInfos for GLWEPlaintextViewMut<'_, BE, Rg> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<BE: Backend, Rg: Ring> crate::layouts::IntPolyInfos for LWEPlaintextViewMut<'_, BE, Rg> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.inner.encoded_k()
    }
}

impl<BE: Backend, Rg: Ring> SetK for GLWEViewMut<'_, BE, Rg> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.inner.set_k(k);
    }
}

impl<BE: Backend, Rg: Ring> SetBase2k for LWEPlaintextViewMut<'_, BE, Rg> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.base2k = base2k;
    }
}

/// Forwards [`GLWEInfos`](crate::layouts::GLWEInfos) through a nominal
/// backend view wrapper generated by [`view_wrapper!`](crate::view_wrapper),
/// implementing it on `$name<'a, BE, Rg>` for every ring `Rg`.
#[macro_export]
macro_rules! impl_glwe_infos {
    ($name:ident) => {
        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a, Rg: ::poulpy_hal::layouts::Ring> $crate::layouts::GLWEInfos
            for $name<'a, BE, Rg>
        {
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
        impl<'a, BE: Backend + 'a, Rg: Ring> GetDistribution for $name<'a, BE, Rg> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }

        impl<'a, BE: Backend + 'a, Rg: Ring> GetDistributionMut for $name<'a, BE, Rg> {
            fn dist_mut(&mut self) -> &mut Distribution {
                self.inner.dist_mut()
            }
        }
    };
}

impl_dist!(GLWESecretTensorViewMut);
impl_dist!(GLWESecretPreparedViewMut);
impl_dist!(GLWESecretViewMut);

impl<BE: Backend, Rg: Ring> GGLWEInfos for GGLWEViewMut<'_, BE, Rg> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_digit()
    }

    fn k_offset(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend, Rg: Ring> GGLWEInfos for GGLWEPreparedViewMut<'_, BE, Rg> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_digit()
    }

    fn k_offset(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<BE: Backend, Rg: Ring> SetGGLWEInfos for GGLWEViewMut<'_, BE, Rg> {
    fn set_k_digit(&mut self, k_digit: TorusPrecision) {
        self.inner.k_digit = k_digit;
    }
}

impl<BE: Backend, Rg: Ring> GGSWInfos for GGSWViewMut<'_, BE, Rg> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend, Rg: Ring> GGSWInfos for GGSWPreparedViewMut<'_, BE, Rg> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> crate::layouts::TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend, Rg: Ring> LWEToBackendRef<BE, Rg> for LWEViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE, Rg> {
        LWE {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k: self.inner.k,
            body: vec_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.body),
            mask: vec_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.mask),
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEToBackendMut<BE, Rg> for LWEViewMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, self.inner.mask.n() + 1);
    }

    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE, Rg> {
        let base2k = self.inner.base2k;
        let k = self.inner.k;
        let body = vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.body);
        let mask = vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.mask);
        self.inner.noise = None;
        LWE {
            noise: None,
            base2k,
            k,
            body,
            mask,
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendRef<BE, Rg> for LWEPlaintextViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE, Rg> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            k: self.inner.k,
            data: vec_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendMut<BE, Rg> for LWEPlaintextViewMut<'_, BE, Rg> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE, Rg> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            k: self.inner.k,
            data: vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
        }
    }
}

macro_rules! impl_glwe_to_backend {
    ($name:ident, |$this:ident| ($canonical:expr, $plaintext_meta:expr), |$this_mut:ident, $flag:ident| $set_canonical:expr) => {
        impl<'a, BE: Backend + 'a, Rg: Ring> GLWEToBackendRef<BE, Rg> for $name<'a, BE, Rg> {
            fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Rg> {
                let $this = self;
                GLWE {
                    noise: crate::layouts::LWEInfos::noise(&self.inner),
                    base2k: self.inner.base2k,
                    k: self.inner.k,
                    canonical: $canonical,
                    plaintext_meta: $plaintext_meta,
                    data: vec_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
                }
            }
        }

        impl<'a, BE: Backend + 'a, Rg: Ring> GLWEToBackendMut<BE, Rg> for $name<'a, BE, Rg> {
            fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
                self.inner.record_noise(metadata);
            }

            fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, Rg> {
                let $this = &*self;
                let (canonical, plaintext_meta) = ($canonical, $plaintext_meta);
                self.inner.record_noise(None);
                GLWE {
                    noise: None,
                    base2k: self.inner.base2k,
                    k: self.inner.k,
                    canonical,
                    plaintext_meta,
                    data: vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
                }
            }

            fn set_canonical(&mut self, $flag: bool) {
                let $this_mut = self;
                $set_canonical
            }
        }
    };
}

impl_glwe_to_backend!(
    GLWEViewMut,
    |this| (this.inner.canonical, this.inner.plaintext_meta),
    |this, canonical| this.inner.canonical = canonical
);
impl_glwe_to_backend!(
    GLWEPlaintextViewMut,
    |this| (true, this.inner.plaintext_meta),
    |_this, _canonical| ()
);
impl_glwe_to_backend!(GLWETensorViewMut, |_this| (true, Default::default()), |_this, _canonical| ());

macro_rules! impl_plaintext_infos {
    ($($name:ident),*) => {$(
        impl<'a, BE: Backend + 'a, Rg: Ring> GLWEPlaintextInfos for $name<'a, BE, Rg> {
            fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
                self.inner.plaintext_meta
            }
        }
    )*};
}

impl_plaintext_infos!(GLWEViewMut, GLWEViewRef, GLWEPlaintextViewMut);

impl<BE: Backend, Rg: Ring> SetGLWEPlaintextInfos for GLWEViewMut<'_, BE, Rg> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.inner.plaintext_meta = meta
    }
}

impl<BE: Backend, Rg: Ring> SetGLWEPlaintextInfos for GLWEPlaintextViewMut<'_, BE, Rg> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.inner.plaintext_meta = meta
    }
}

impl<BE: Backend, Rg: Ring> GLWEToBackendRef<BE, Rg> for GLWEViewRef<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Rg> {
        GLWE {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k: self.inner.k,
            canonical: self.inner.canonical,
            plaintext_meta: self.inner.plaintext_meta,
            data: vec_znx_backend_ref_from_ref::<BE, Rg>(&self.inner.data),
        }
    }
}

impl<BE: Backend, Rg: Ring> GLWESecretToBackendRef<BE, Rg> for GLWESecretViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE, Rg> {
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

impl<BE: Backend, Rg: Ring> GLWESecretToBackendMut<BE, Rg> for GLWESecretViewMut<'_, BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE, Rg> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<BE: Backend, Rg: Ring> GLWESecretTensorToBackendRef<BE, Rg> for GLWESecretTensorViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE, Rg> {
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

impl<BE: Backend, Rg: Ring> GLWESecretTensorToBackendMut<BE, Rg> for GLWESecretTensorViewMut<'_, BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE, Rg> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecretTensor {
            dist: self.inner.dist,
            rank: self.inner.rank,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<BE: Backend, Rg: Ring> GLWESecretPreparedToBackendRef<BE, Rg> for GLWESecretPreparedViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWESecretPreparedBackendRef<'_, BE, Rg> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend, Rg: Ring> GLWESecretPreparedToBackendMut<BE, Rg> for GLWESecretPreparedViewMut<'_, BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretPreparedBackendMut<'_, BE, Rg> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<BE: Backend, Rg: Ring> GGLWEToBackendRef<BE, Rg> for GGLWEViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE, Rg> {
        GGLWEBackendRef::from_inner(GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            data: mat_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGLWEToBackendMut<BE, Rg> for GGLWEViewMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, self.inner.data.cols_out());
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GGLWEBackendMut::from_inner(GGLWE {
            noise: None,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            data: mat_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGLWEPreparedToBackendRef<BE, Rg> for GGLWEPreparedViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, BE, Rg> {
        GGLWEPrepared {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            dnum: self.inner.dnum,
            stride: self.inner.stride,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend, Rg: Ring> GGLWEPreparedToBackendMut<BE, Rg> for GGLWEPreparedViewMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, self.inner.data.cols_out());
    }

    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GGLWEPrepared {
            noise: None,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            dnum: self.inner.dnum,
            stride: self.inner.stride,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<BE: Backend, Rg: Ring> GGSWToBackendRef<BE, Rg> for GGSWViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE, Rg> {
        GGSWBackendRef::from_inner(GGSW {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            data: mat_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGSWToBackendMut<BE, Rg> for GGSWViewMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, self.inner.data.cols_out());
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GGSWBackendMut::from_inner(GGSW {
            noise: None,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            data: mat_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGSWPreparedToBackendRef<BE, Rg> for GGSWPreparedViewMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, BE, Rg> {
        GGSWPrepared {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<BE: Backend, Rg: Ring> GGSWPreparedToBackendMut<BE, Rg> for GGSWPreparedViewMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, self.inner.data.cols_out());
    }

    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GGSWPrepared {
            noise: None,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}
