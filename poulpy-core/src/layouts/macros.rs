macro_rules! impl_gglwe_to_backend_for_field {
    ($ty:ty, $field:tt, $inner:ty) => {
        impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for $ty
        where
            $inner: GGLWEToBackendRef<BE>,
        {
            fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
                self.$field.to_backend_ref()
            }
        }

        impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for $ty
        where
            $inner: GGLWEToBackendMut<BE>,
        {
            fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
                self.$field.to_backend_mut()
            }
        }
    };
}

macro_rules! impl_gglwe_at_view_for_field {
    ($ty:ty; $($field:tt)+) => {
        impl<BE: Backend> GGLWEAtViewRef<BE> for $ty {
            fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
                self.$($field)+.at_view(row, col)
            }
        }

        impl<BE: Backend> GGLWEAtViewMut<BE> for $ty {
            fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
                self.$($field)+.at_view_mut(row, col)
            }
        }
    };
}

macro_rules! impl_glwe_host_at_for_field {
    ($ty:ty; $($field:tt)+) => {
        impl<D: HostDataRef> $ty {
            pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
                self.$($field)+.at(row, col)
            }
        }

        impl<D: HostDataMut> $ty {
            pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
                self.$($field)+.at_mut(row, col)
            }
        }
    };
}

macro_rules! impl_gglwe_infos_for_inner {
    ($ty:ty, [$($gen:tt)*]; $($field:tt)+) => {
        impl<$($gen)*> LWEInfos for $ty {
            fn n(&self) -> Degree {
                self.$($field)+.n()
            }

            fn base2k(&self) -> Base2K {
                self.$($field)+.base2k()
            }

            fn size(&self) -> usize {
                self.$($field)+.size()
            }
        }

        impl<$($gen)*> GLWEInfos for $ty {
            fn rank(&self) -> Rank {
                self.$($field)+.rank()
            }
        }

        impl<$($gen)*> GGLWEInfos for $ty {
            fn rank_in(&self) -> Rank {
                self.$($field)+.rank_in()
            }

            fn rank_out(&self) -> Rank {
                self.$($field)+.rank_out()
            }

            fn dsize(&self) -> Dsize {
                self.$($field)+.dsize()
            }

            fn dnum(&self) -> Dnum {
                self.$($field)+.dnum()
            }
        }
    };
}
