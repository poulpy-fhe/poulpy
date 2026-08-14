//! Container family shared by the two SVP scalar prep tiers.
//!
//! [`SvpPPol`](crate::layouts::SvpPPol) and [`SvpTPol`](crate::layouts::SvpTPol)
//! are distinct nominal types with independent `bytes_of` and layout-compat
//! markers, but their container plumbing does not vary by tier.
//! [`svp_pol_family!`] emits it so a tier cannot silently drift from the other.

/// Emits one complete SVP scalar-polynomial container family.
///
/// `$Pol` is the struct name and `$stem` its snake-case counterpart, from which
/// the view aliases, borrow helper, `bytes_of_*` binding and layout-compat
/// marker names are derived.
macro_rules! svp_pol_family {
    (
        $(#[$meta:meta])*
        $Pol:ident, $stem:ident
    ) => {
        paste::paste! {

        $(#[$meta])*
        #[repr(C)]
        pub struct $Pol<D: Data, W: DftWord, B: Backend<DftWord = W>> {
            pub data: D,
            shape: ScalarZnxShape,
            pub _phantom: PhantomData<(W, B)>,
        }

        // Equality (and hashing, where provided) is defined directly on the
        // representation: same shape, same buffer bytes. No `W`/`B` value is ever
        // compared, so no bound on them is needed — in particular `Eq` holds even
        // for non-`Eq` words like `f64` (byte equality is a total equivalence).
        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> PartialEq for $Pol<D, W, B> {
            fn eq(&self, other: &Self) -> bool {
                self.shape == other.shape && self.data == other.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for $Pol<D, W, B> {}

        impl<D: Data + std::hash::Hash, W: DftWord, B: Backend<DftWord = W>> std::hash::Hash for $Pol<D, W, B> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.shape.hash(state);
                self.data.hash(state);
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for $Pol<D, W, B> {
            fn digest_u64(&self) -> u64 {
                let mut h: DefaultHasher = DefaultHasher::new();
                h.write(self.data.as_ref());
                h.write_usize(self.n());
                h.write_usize(self.cols());
                h.finish()
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for $Pol<D, W, B> {
            type Scalar = W;
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for $Pol<D, W, B> {
            fn n(&self) -> usize {
                self.shape.n()
            }

            fn size(&self) -> usize {
                1
            }

            fn poly_count(&self) -> usize {
                crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxInfos for $Pol<D, W, B> {
            fn cols(&self) -> usize {
                self.shape.cols()
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Pol<D, W, B> {
            pub fn n(&self) -> usize {
                self.shape.n()
            }

            pub fn cols(&self) -> usize {
                self.shape.cols()
            }

            pub fn shape(&self) -> ScalarZnxShape {
                self.shape
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for $Pol<D, W, B> {
            type D = D;
            fn data(&self) -> &Self::D {
                &self.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for $Pol<D, W, B> {
            fn data_mut(&mut self) -> &mut Self::D {
                &mut self.data
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Pol<D, W, B> {
            #[doc = concat!("Allocates a zero-initialized backend-owned `", stringify!($Pol), "`.")]
            pub fn alloc(n: usize, cols: usize) -> [<$Pol Owned>]<B>
            where
                B: Backend<OwnedBuf = D>,
            {
                let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::[<bytes_of_ $stem>](n, cols));
                $Pol {
                    data,
                    shape: ScalarZnxShape::new(n, cols),
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Owned `", stringify!($Pol), "` backed by a backend-owned buffer.")]
        pub type [<$Pol Owned>]<B> = $Pol<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
        #[doc = concat!("Shared backend-native borrow of an `", stringify!($Pol), "`.")]
        pub type [<$Pol BackendRef>]<'a, B> = $Pol<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
        #[doc = concat!("Mutable backend-native borrow of an `", stringify!($Pol), "`.")]
        pub type [<$Pol BackendMut>]<'a, B> = $Pol<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

        #[doc = concat!(
            "Reborrow a mutable backend-native `", stringify!($Pol),
            "` view as a shared backend-native view."
        )]
        pub fn [<$stem _backend_ref_from_mut>]<'a, 'b, B: Backend>(
            pol: &'a [<$Pol BackendMut>]<'b, B>,
        ) -> [<$Pol BackendRef>]<'a, B> {
            $Pol {
                data: B::view_ref_mut(&pol.data),
                shape: pol.shape,
                _phantom: PhantomData,
            }
        }

        #[doc = concat!("Borrow a backend-owned `", stringify!($Pol), "` using the backend's native view type.")]
        pub trait [<$Pol ToBackendRef>]<B: Backend> {
            fn to_backend_ref(&self) -> [<$Pol BackendRef>]<'_, B>;
        }

        impl<B: Backend> [<$Pol ToBackendRef>]<B> for $Pol<B::OwnedBuf, B::DftWord, B> {
            fn to_backend_ref(&self) -> [<$Pol BackendRef>]<'_, B> {
                $Pol {
                    data: B::view(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        impl<'b, B: Backend + 'b> [<$Pol ToBackendRef>]<B> for &$Pol<B::BufRef<'b>, B::DftWord, B> {
            fn to_backend_ref(&self) -> [<$Pol BackendRef>]<'_, B> {
                $Pol {
                    data: B::view_ref(&self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!($Pol), "` as a shared backend-native view.")]
        pub trait [<$Pol ReborrowBackendRef>]<B: Backend> {
            fn reborrow_backend_ref(&self) -> [<$Pol BackendRef>]<'_, B>;
        }

        impl<'b, B: Backend + 'b> [<$Pol ReborrowBackendRef>]<B> for $Pol<B::BufMut<'b>, B::DftWord, B> {
            fn reborrow_backend_ref(&self) -> [<$Pol BackendRef>]<'_, B> {
                [<$stem _backend_ref_from_mut>]::<B>(self)
            }
        }

        #[doc = concat!("Mutably borrow a backend-owned `", stringify!($Pol), "` using the backend's native view type.")]
        pub trait [<$Pol ToBackendMut>]<B: Backend> {
            fn to_backend_mut(&mut self) -> [<$Pol BackendMut>]<'_, B>;
        }

        impl<B: Backend> [<$Pol ToBackendMut>]<B> for $Pol<B::OwnedBuf, B::DftWord, B> {
            fn to_backend_mut(&mut self) -> [<$Pol BackendMut>]<'_, B> {
                $Pol {
                    data: B::view_mut(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        impl<'b, B: Backend + 'b> [<$Pol ToBackendMut>]<B> for &mut $Pol<B::BufMut<'b>, B::DftWord, B> {
            fn to_backend_mut(&mut self) -> [<$Pol BackendMut>]<'_, B> {
                $Pol {
                    data: B::view_mut_ref(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        #[doc = concat!("Reborrow an already backend-borrowed `", stringify!($Pol), "` as a mutable backend-native view.")]
        pub trait [<$Pol ReborrowBackendMut>]<B: Backend> {
            fn reborrow_backend_mut(&mut self) -> [<$Pol BackendMut>]<'_, B>;
        }

        impl<'b, B: Backend + 'b> [<$Pol ReborrowBackendMut>]<B> for $Pol<B::BufMut<'b>, B::DftWord, B> {
            fn reborrow_backend_mut(&mut self) -> [<$Pol BackendMut>]<'_, B> {
                $Pol {
                    data: B::view_mut_ref(&mut self.data),
                    shape: self.shape,
                    _phantom: PhantomData,
                }
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Pol<D, W, B> {
            pub fn from_data(data: D, n: usize, cols: usize) -> Self {
                Self {
                    data,
                    shape: ScalarZnxShape::new(n, cols),
                    _phantom: PhantomData,
                }
            }
        }

        impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> fmt::Display for $Pol<D, W, B> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "{}(n={}, cols={})", stringify!($Pol), self.n(), self.cols())?;

                for col in 0..self.cols() {
                    writeln!(f, "Column {col}:")?;
                    let coeffs = self.at(col, 0);
                    write!(f, "[")?;

                    let max_show = 100;
                    let show_count = coeffs.len().min(max_show);

                    for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{coeff}")?;
                    }

                    if coeffs.len() > max_show {
                        write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
                    }

                    writeln!(f, "]")?;
                }
                Ok(())
            }
        }

        impl<D: Data, W: DftWord, B: Backend<DftWord = W>> $Pol<D, W, B> {
            /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
            ///
            /// The buffer moves as-is; only the type tag changes. Requires the
            /// layout-compat marker declared by the backend pair. `D` is kept, so
            /// for further backend-native use `B2`'s buffer types must match `D`
            /// (true for all current CPU backends).
            pub fn into_backend<B2>(self) -> $Pol<D, W, B2>
            where
                B2: Backend<DftWord = W>,
                B: crate::layouts::[<$Pol LayoutCompatible>]<B2>,
            {
                let shape = self.shape;
                assert_eq!(
                    B::[<bytes_of_ $stem>](shape.n(), shape.cols()),
                    B2::[<bytes_of_ $stem>](shape.n(), shape.cols()),
                    "into_backend: byte sizes diverge despite declared layout compatibility"
                );
                $Pol {
                    data: self.data,
                    shape,
                    _phantom: PhantomData,
                }
            }
        }

        }
    };
}
