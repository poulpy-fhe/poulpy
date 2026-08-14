//! Container family shared by the two CNV operand prep tiers.
//!
//! Each tier supplies a left and a right container ([`CnvPVecL`]/[`CnvPVecR`],
//! [`CnvTVecL`]/[`CnvTVecR`]) with independent `bytes_of` bindings and one
//! layout-compat marker per tier. The container plumbing does not vary, so
//! [`cnv_vec_family!`] emits all four from one body.
//!
//! [`CnvPVecL`]: crate::layouts::CnvPVecL
//! [`CnvPVecR`]: crate::layouts::CnvPVecR
//! [`CnvTVecL`]: crate::layouts::CnvTVecL
//! [`CnvTVecR`]: crate::layouts::CnvTVecR

/// Emits one CNV operand container (`L` or `R` side of one tier).
///
/// `$Vec` is the tier stem (`CnvPVec`), `$side` the side suffix (`L`), and
/// `$bytes_of` the `Backend` method giving that container's buffer size.
macro_rules! cnv_vec_side {
    ($Vec:ident, $side:ident, $bytes_of:ident, $doc:expr) => {
        paste::paste! {
        #[doc = $doc]
        pub struct [<$Vec $side>]<D: Data, W: DftWord, B: Backend<DftWord = W>> {
            data: D,
            shape: [<$Vec Shape>],
            _phantom: PhantomData<(W, B)>,
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for [<$Vec $side>]<D, W, B> {
            fn n(&self) -> usize {
                self.shape.n()
            }

            fn size(&self) -> usize {
                self.shape.size()
            }

            fn poly_count(&self) -> usize {
                crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxInfos for [<$Vec $side>]<D, W, B> {
            fn cols(&self) -> usize {
                self.shape.cols()
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for [<$Vec $side>]<D, W, B> {
            type D = D;
            fn data(&self) -> &Self::D {
                &self.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for [<$Vec $side>]<D, W, B> {
            fn data_mut(&mut self) -> &mut Self::D {
                &mut self.data
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for [<$Vec $side>]<D, W, B> {
            type Scalar = W;
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> [<$Vec $side>]<D, W, B> {
            pub fn shape(&self) -> [<$Vec Shape>] {
                self.shape
            }

            pub fn n(&self) -> usize {
                self.shape.n()
            }

            pub fn cols(&self) -> usize {
                self.shape.cols()
            }

            pub fn size(&self) -> usize {
                self.shape.size()
            }

            pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
                Self {
                    data,
                    shape: [<$Vec Shape>]::new(n, cols, size),
                    _phantom: PhantomData,
                }
            }

            #[doc = concat!("Allocates a zero-initialized backend-owned `", stringify!([<$Vec $side>]), "`.")]
            pub fn alloc(n: usize, cols: usize, size: usize) -> [<$Vec $side>]<B::OwnedBuf, W, B>
            where
                B: Backend<OwnedBuf = D>,
            {
                let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::$bytes_of(n, cols, size));
                [<$Vec $side>] {
                    data,
                    shape: [<$Vec Shape>]::new(n, cols, size),
                    _phantom: PhantomData,
                }
            }

            #[doc = concat!("Uploads a host byte buffer into a backend-owned `", stringify!([<$Vec $side>]), "`.")]
            ///
            /// # Panics
            ///
            /// Panics if the buffer length does not match the backend's byte size
            /// for this container.
            pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> [<$Vec $side>]<B::OwnedBuf, W, B>
            where
                B: Backend<OwnedBuf = D>,
            {
                let data: Vec<u8> = bytes.into();
                assert!(data.len() == B::$bytes_of(n, cols, size));
                let data: B::OwnedBuf = B::from_host_bytes(&data);
                [<$Vec $side>] {
                    data,
                    shape: [<$Vec Shape>]::new(n, cols, size),
                    _phantom: PhantomData,
                }
            }

            /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
            ///
            /// The buffer moves as-is; only the type tag changes. Requires the
            /// tier's layout-compat marker declared by the backend pair. `D` is
            /// kept, so for further backend-native use `B2`'s buffer types must
            /// match `D` (true for all current CPU backends).
            pub fn into_backend<B2>(self) -> [<$Vec $side>]<D, W, B2>
            where
                B2: Backend<DftWord = W>,
                B: crate::layouts::[<$Vec LayoutCompatible>]<B2>,
            {
                let shape = self.shape;
                assert_eq!(
                    B::$bytes_of(shape.n(), shape.cols(), shape.size()),
                    B2::$bytes_of(shape.n(), shape.cols(), shape.size()),
                    "into_backend: byte sizes diverge despite declared layout compatibility"
                );
                [<$Vec $side>] {
                    data: self.data,
                    shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Owned `", stringify!([<$Vec $side>]), "` backed by a backend-owned buffer.")]
        pub type [<$Vec $side Owned>]<B> = [<$Vec $side>]<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
        #[doc = concat!("Shared backend-native borrow of a `", stringify!([<$Vec $side>]), "`.")]
        pub type [<$Vec $side BackendRef>]<'a, B> = [<$Vec $side>]<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
        #[doc = concat!("Mutable backend-native borrow of a `", stringify!([<$Vec $side>]), "`.")]
        pub type [<$Vec $side BackendMut>]<'a, B> = [<$Vec $side>]<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

        #[doc = concat!("Borrow a backend-owned `", stringify!([<$Vec $side>]), "` using the backend's native view type.")]
        pub trait [<$Vec $side ToBackendRef>]<BE: Backend> {
            fn to_backend_ref(&self) -> [<$Vec $side BackendRef>]<'_, BE>;
        }

        impl<BE: Backend> [<$Vec $side ToBackendRef>]<BE> for [<$Vec $side>]<BE::OwnedBuf, BE::DftWord, BE> {
            fn to_backend_ref(&self) -> [<$Vec $side BackendRef>]<'_, BE> {
                [<$Vec $side>] {
                    data: BE::view(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!([<$Vec $side>]), "` as a shared backend-native view.")]
        pub trait [<$Vec $side ReborrowBackendRef>]<BE: Backend> {
            fn reborrow_backend_ref(&self) -> [<$Vec $side BackendRef>]<'_, BE>;
        }

        impl<'b, BE: Backend + 'b> [<$Vec $side ReborrowBackendRef>]<BE> for [<$Vec $side>]<BE::BufMut<'b>, BE::DftWord, BE> {
            fn reborrow_backend_ref(&self) -> [<$Vec $side BackendRef>]<'_, BE> {
                [<$Vec $side>] {
                    data: BE::view_ref_mut(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Mutably borrow a backend-owned `", stringify!([<$Vec $side>]), "` using the backend's native view type.")]
        pub trait [<$Vec $side ToBackendMut>]<BE: Backend> {
            fn to_backend_mut(&mut self) -> [<$Vec $side BackendMut>]<'_, BE>;
        }

        impl<BE: Backend> [<$Vec $side ToBackendMut>]<BE> for [<$Vec $side>]<BE::OwnedBuf, BE::DftWord, BE> {
            fn to_backend_mut(&mut self) -> [<$Vec $side BackendMut>]<'_, BE> {
                [<$Vec $side>] {
                    data: BE::view_mut(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!([<$Vec $side>]), "` as a mutable backend-native view.")]
        pub trait [<$Vec $side ReborrowBackendMut>]<BE: Backend> {
            fn reborrow_backend_mut(&mut self) -> [<$Vec $side BackendMut>]<'_, BE>;
        }

        impl<'b, BE: Backend + 'b> [<$Vec $side ReborrowBackendMut>]<BE> for [<$Vec $side>]<BE::BufMut<'b>, BE::DftWord, BE> {
            fn reborrow_backend_mut(&mut self) -> [<$Vec $side BackendMut>]<'_, BE> {
                [<$Vec $side>] {
                    data: BE::view_mut_ref(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }
        }
    };
}

/// Emits one complete CNV operand tier: its shape type plus the left and right
/// containers.
macro_rules! cnv_vec_family {
    ($Vec:ident, $stem:ident, $doc_l:expr, $doc_r:expr) => {
        paste::paste! {
        #[repr(C)]
        #[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
        pub struct [<$Vec Shape>] {
            n: usize,
            size: usize,
            cols: usize,
        }

        impl [<$Vec Shape>] {
            pub const fn new(n: usize, cols: usize, size: usize) -> Self {
                Self { n, size, cols }
            }

            pub const fn n(self) -> usize {
                self.n
            }

            pub const fn size(self) -> usize {
                self.size
            }

            pub const fn cols(self) -> usize {
                self.cols
            }
        }

        cnv_vec_side!($Vec, L, [<bytes_of_ $stem _left>], $doc_l);
        cnv_vec_side!($Vec, R, [<bytes_of_ $stem _right>], $doc_r);
        }
    };
}
