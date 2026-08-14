//! Container family shared by the two VMP matrix prep tiers.
//!
//! [`VmpPMat`](crate::layouts::VmpPMat) and [`VmpTMat`](crate::layouts::VmpTMat)
//! are distinct nominal types with independent `bytes_of` and layout-compat
//! markers, but their container plumbing (shape accessors, borrow/reborrow
//! views, backend re-tag) does not vary by tier. [`vmp_mat_family!`] emits it so
//! a tier cannot silently drift from the other.

/// Emits one complete VMP matrix container family.
///
/// `$Mat` is the struct name and `$stem` its snake-case counterpart, from which
/// the shape type, view aliases, borrow helpers, `bytes_of_*` binding and
/// layout-compat marker names are derived.
macro_rules! vmp_mat_family {
    (
        $(#[$meta:meta])*
        $Mat:ident, $stem:ident
    ) => {
        paste::paste! {

        #[repr(C)]
        #[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
        pub struct [<$Mat Shape>] {
            n: usize,
            size: usize,
            rows: usize,
            cols_in: usize,
            cols_out: usize,
        }

        impl [<$Mat Shape>] {
            pub const fn new(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
                Self { n, size, rows, cols_in, cols_out }
            }

            pub const fn n(self) -> usize {
                self.n
            }

            pub const fn size(self) -> usize {
                self.size
            }

            pub const fn rows(self) -> usize {
                self.rows
            }

            pub const fn cols_in(self) -> usize {
                self.cols_in
            }

            pub const fn cols_out(self) -> usize {
                self.cols_out
            }
        }

        $(#[$meta])*
        #[repr(C)]
        pub struct $Mat<D: Data, W: DftWord, B: Backend<DftWord = W>> {
            data: D,
            shape: [<$Mat Shape>],
            _phantom: PhantomData<(W, B)>,
        }

        // Equality (and hashing, where provided) is defined directly on the
        // representation: same shape, same buffer bytes. No `W`/`B` value is ever
        // compared, so no bound on them is needed — in particular `Eq` holds even
        // for non-`Eq` words like `f64` (byte equality is a total equivalence).
        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> PartialEq for $Mat<D, W, B> {
            fn eq(&self, other: &Self) -> bool {
                self.shape == other.shape && self.data == other.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for $Mat<D, W, B> {}

        impl<D: Data + std::hash::Hash, W: DftWord, B: Backend<DftWord = W>> std::hash::Hash for $Mat<D, W, B> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.shape.hash(state);
                self.data.hash(state);
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for $Mat<D, W, B> {
            fn digest_u64(&self) -> u64 {
                let mut h: DefaultHasher = DefaultHasher::new();
                h.write(self.data.as_ref());
                h.write_usize(self.n());
                h.write_usize(self.size());
                h.write_usize(self.rows());
                h.write_usize(self.cols_in());
                h.write_usize(self.cols_out());
                h.finish()
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            /// Returns the whole element view as a scalar slice.
            ///
            /// The prepared matrix is packed in a backend-defined order with no flat
            /// `(col, limb)` indexing, so it exposes the buffer rather than
            /// implementing [`ZnxView`](crate::layouts::ZnxView).
            pub fn raw(&self) -> &[W] {
                let span: usize = crate::layouts::element_view_span(self);
                crate::layouts::raw_scalars(self.data.as_ref(), span)
            }
        }

        impl<D: HostDataMut, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            /// Mutable counterpart of [`Self::raw`].
            pub fn raw_mut(&mut self) -> &mut [W] {
                let span: usize = crate::layouts::element_view_span(self);
                crate::layouts::raw_scalars_mut(self.data.as_mut(), span)
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for $Mat<D, W, B> {
            fn n(&self) -> usize {
                self.shape.n()
            }

            fn size(&self) -> usize {
                self.shape.size()
            }

            fn poly_count(&self) -> usize {
                crate::layouts::checked_product(
                    &[self.rows(), self.cols_in(), self.size(), self.cols_out()],
                    concat!(stringify!($Mat), " polynomial count"),
                )
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> MatZnxInfos for $Mat<D, W, B> {
            fn rows(&self) -> usize {
                self.shape.rows()
            }

            fn cols_in(&self) -> usize {
                self.shape.cols_in()
            }

            fn cols_out(&self) -> usize {
                self.shape.cols_out()
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for $Mat<D, W, B> {
            type D = D;
            fn data(&self) -> &Self::D {
                &self.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for $Mat<D, W, B> {
            fn data_mut(&mut self) -> &mut Self::D {
                &mut self.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            pub fn shape(&self) -> [<$Mat Shape>] {
                self.shape
            }

            pub fn n(&self) -> usize {
                self.shape.n()
            }

            pub fn rows(&self) -> usize {
                self.shape.rows()
            }

            pub fn size(&self) -> usize {
                self.shape.size()
            }

            /// Returns the number of input columns.
            pub fn cols_in(&self) -> usize {
                self.shape.cols_in()
            }

            /// Returns the number of output columns.
            pub fn cols_out(&self) -> usize {
                self.shape.cols_out()
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            #[doc = concat!("Allocates a zero-initialized backend-owned `", stringify!($Mat), "`.")]
            pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> [<$Mat Owned>]<B>
            where
                B: Backend<OwnedBuf = D>,
            {
                let data: <B as Backend>::OwnedBuf =
                    B::alloc_zeroed_bytes(B::[<bytes_of_ $stem>](n, rows, cols_in, cols_out, size));
                $Mat {
                    data,
                    shape: [<$Mat Shape>]::new(n, rows, cols_in, cols_out, size),
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Owned `", stringify!($Mat), "` backed by a backend-owned buffer.")]
        pub type [<$Mat Owned>]<B> = $Mat<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
        #[doc = concat!("Immutably borrowed `", stringify!($Mat), "`.")]
        pub type [<$Mat Ref>]<'a, B> = $Mat<&'a [u8], <B as Backend>::DftWord, B>;
        #[doc = concat!("Shared backend-native borrow of a `", stringify!($Mat), "`.")]
        pub type [<$Mat BackendRef>]<'a, B> = $Mat<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
        #[doc = concat!("Mutable backend-native borrow of a `", stringify!($Mat), "`.")]
        pub type [<$Mat BackendMut>]<'a, B> = $Mat<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

        #[doc = concat!(
            "Reborrow an immutable backend-native `", stringify!($Mat),
            "` view as a shared backend-native view."
        )]
        pub fn [<$stem _backend_ref_from_ref>]<'a, 'b, B: Backend + 'b>(
            mat: &'a $Mat<B::BufRef<'b>, B::DftWord, B>,
        ) -> [<$Mat BackendRef>]<'a, B> {
            $Mat {
                data: B::view_ref(&mat.data),
                shape: mat.shape,
                _phantom: PhantomData,
            }
        }

        #[doc = concat!(
            "Reborrow a mutable backend-native `", stringify!($Mat),
            "` view as a shared backend-native view."
        )]
        pub fn [<$stem _backend_ref_from_mut>]<'a, B: Backend>(
            mat: &'a [<$Mat BackendMut>]<'a, B>,
        ) -> [<$Mat BackendRef>]<'a, B> {
            $Mat {
                data: B::view_ref_mut(&mat.data),
                shape: mat.shape,
                _phantom: PhantomData,
            }
        }

        pub fn [<$stem _backend_mut_from_mut>]<'a, 'b, B: Backend + 'b>(
            mat: &'a mut [<$Mat BackendMut>]<'b, B>,
        ) -> [<$Mat BackendMut>]<'a, B> {
            $Mat {
                data: B::view_mut_ref(&mut mat.data),
                shape: mat.shape,
                _phantom: PhantomData,
            }
        }

        #[doc = concat!("Borrow a backend-owned `", stringify!($Mat), "` using the backend's native view type.")]
        pub trait [<$Mat ToBackendRef>]<B: Backend> {
            fn to_backend_ref(&self) -> [<$Mat BackendRef>]<'_, B>;
        }

        impl<B: Backend> [<$Mat ToBackendRef>]<B> for $Mat<B::OwnedBuf, B::DftWord, B> {
            fn to_backend_ref(&self) -> [<$Mat BackendRef>]<'_, B> {
                $Mat {
                    data: B::view(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        impl<'b, B: Backend + 'b> [<$Mat ToBackendRef>]<B> for &$Mat<B::BufRef<'b>, B::DftWord, B> {
            fn to_backend_ref(&self) -> [<$Mat BackendRef>]<'_, B> {
                $Mat {
                    data: B::view_ref(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!($Mat), "` as a shared backend-native view.")]
        pub trait [<$Mat ReborrowBackendRef>]<B: Backend> {
            fn reborrow_backend_ref(&self) -> [<$Mat BackendRef>]<'_, B>;
        }

        impl<'b, B: Backend + 'b> [<$Mat ReborrowBackendRef>]<B> for $Mat<B::BufMut<'b>, B::DftWord, B> {
            fn reborrow_backend_ref(&self) -> [<$Mat BackendRef>]<'_, B> {
                $Mat {
                    data: B::view_ref_mut(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Mutably borrow a backend-owned `", stringify!($Mat), "` using the backend's native view type.")]
        pub trait [<$Mat ToBackendMut>]<B: Backend> {
            fn to_backend_mut(&mut self) -> [<$Mat BackendMut>]<'_, B>;
        }

        impl<B: Backend> [<$Mat ToBackendMut>]<B> for $Mat<B::OwnedBuf, B::DftWord, B> {
            fn to_backend_mut(&mut self) -> [<$Mat BackendMut>]<'_, B> {
                $Mat {
                    data: B::view_mut(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        impl<'b, B: Backend + 'b> [<$Mat ToBackendMut>]<B> for &mut $Mat<B::BufMut<'b>, B::DftWord, B> {
            fn to_backend_mut(&mut self) -> [<$Mat BackendMut>]<'_, B> {
                [<$stem _backend_mut_from_mut>]::<B>(self)
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!($Mat), "` as a mutable backend-native view.")]
        pub trait [<$Mat ReborrowBackendMut>]<B: Backend> {
            fn reborrow_backend_mut(&mut self) -> [<$Mat BackendMut>]<'_, B>;
        }

        impl<'b, B: Backend + 'b> [<$Mat ReborrowBackendMut>]<B> for $Mat<B::BufMut<'b>, B::DftWord, B> {
            fn reborrow_backend_mut(&mut self) -> [<$Mat BackendMut>]<'_, B> {
                [<$stem _backend_mut_from_mut>]::<B>(self)
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
                Self {
                    data,
                    shape: [<$Mat Shape>]::new(n, rows, cols_in, cols_out, size),
                    _phantom: PhantomData,
                }
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Mat<D, W, B> {
            #[doc = concat!("Zero-copy re-tag of this container to a layout-compatible backend `B2`.")]
            ///
            /// The buffer moves as-is; only the type tag changes. Requires the
            /// layout-compat marker declared by the backend pair. `D` is kept, so
            /// for further backend-native use `B2`'s buffer types must match `D`
            /// (true for all current CPU backends).
            pub fn into_backend<B2>(self) -> $Mat<D, W, B2>
            where
                B2: Backend<DftWord = W>,
                B: crate::layouts::[<$Mat LayoutCompatible>]<B2>,
            {
                let shape = self.shape;
                assert_eq!(
                    B::[<bytes_of_ $stem>](shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
                    B2::[<bytes_of_ $stem>](shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
                    "into_backend: byte sizes diverge despite declared layout compatibility"
                );
                $Mat {
                    data: self.data,
                    shape,
                    _phantom: PhantomData,
                }
            }
        }

        }
    };
}
