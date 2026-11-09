macro_rules! impl_gglwe_to_backend_for_field {
    ($ty:ty, $field:tt, $inner:ty) => {
        impl<BE: Backend, D: Data, Rg: ::poulpy_hal::layouts::Ring> GGLWEToBackendRef<BE, Rg> for $ty
        where
            $inner: GGLWEToBackendRef<BE, Rg>,
        {
            fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE, Rg> {
                self.$field.to_backend_ref()
            }
        }

        impl<BE: Backend, D: Data, Rg: ::poulpy_hal::layouts::Ring> GGLWEToBackendMut<BE, Rg> for $ty
        where
            $inner: GGLWEToBackendMut<BE, Rg>,
        {
            fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
                <_ as GGLWEToBackendMut<BE, Rg>>::set_noise(&mut self.$field, metadata);
            }

            fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE, Rg> {
                self.$field.to_backend_mut()
            }
        }
    };
}

macro_rules! impl_gglwe_at_view_for_field {
    ($ty:ty; $($field:tt)+) => {
        impl<BE: Backend, Rg: ::poulpy_hal::layouts::Ring> GGLWEAtViewRef<BE, Rg> for $ty {
            fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE, Rg> {
                self.$($field)+.at_view(row, col)
            }
        }

        impl<BE: Backend, Rg: ::poulpy_hal::layouts::Ring> GGLWEAtViewMut<BE, Rg> for $ty {
            fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE, Rg> {
                self.$($field)+.at_view_mut(row, col)
            }
        }
    };
}

macro_rules! impl_glwe_host_at_for_field {
    ($ty:ty; $($field:tt)+) => {
        impl<D: HostDataRef, W: ZnxWord, Rg: ::poulpy_hal::layouts::Ring> $ty {
            pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8], W, Rg> {
                self.$($field)+.at(row, col)
            }
        }

        impl<D: HostDataMut, W: ZnxWord, Rg: ::poulpy_hal::layouts::Ring> $ty {
            pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8], W, Rg> {
                self.$($field)+.at_mut(row, col)
            }
        }
    };
}

macro_rules! impl_gglwe_infos_for_inner {
    ($ty:ty, [$($gen:tt)*]; $($field:tt)+) => {
        impl<$($gen)*> LWEInfos for $ty {
            fn noise(&self) -> Option<crate::ComponentNoise> { self.$($field)+.noise() }

            fn n(&self) -> Degree {
                self.$($field)+.n()
            }

            fn base2k(&self) -> Base2K {
                self.$($field)+.base2k()
            }

            fn max_size(&self) -> usize {
                self.$($field)+.max_size()
            }

            fn size(&self) -> usize {
                self.$($field)+.size()
            }

            fn k(&self) -> TorusPrecision{
                self.$($field)+.k()
            }
        }

        impl<$($gen)*> GLWEInfos for $ty {
            fn rank(&self) -> Rank {
                self.$($field)+.rank()
            }
        }

        impl<$($gen)*> GGLWEInfos for $ty {
            fn k_aux(&self) -> TorusPrecision {
                self.$($field)+.k_aux()
            }

            fn rank_in(&self) -> Rank {
                self.$($field)+.rank_in()
            }

            fn rank_out(&self) -> Rank {
                self.$($field)+.rank_out()
            }

            fn k_digit(&self) -> TorusPrecision {
                self.$($field)+.k_digit()
            }

            fn k_offset(&self) -> TorusPrecision {
                self.$($field)+.k_offset()
            }

            fn dnum(&self) -> Dnum {
                self.$($field)+.dnum()
            }
        }
    };
}
