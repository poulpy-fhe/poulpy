//! Protocol shares that are one unseeded GLWE transcript, a core `GLWE`.

/// Defines a protocol share wrapping one core `GLWE`, with its infos,
/// backend borrows and serialization forwarded to it. The caller imports the
/// names it uses.
macro_rules! glwe_share {
    ($(#[$doc:meta])* $name:ident, $owned:ident) => {
        pub type $owned<BE> = $name<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

        $(#[$doc])*
        ///
        /// Serializes as its core `GLWE`.
        #[derive(Clone)]
        pub struct $name<D: Data, W: ZnxWord> {
            pub(crate) inner: GLWE<D, W>,
        }

        impl<D: Data, W: ZnxWord> PartialEq for $name<D, W>
        where
            GLWE<D, W>: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.inner == other.inner
            }
        }

        impl<D: Data, W: ZnxWord> Eq for $name<D, W> where
            GLWE<D, W>: Eq
        {
        }

        impl<D: Data, W: ZnxWord> LWEInfos for $name<D, W> {
            fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
                self.inner.noise()
            }

            fn n(&self) -> Degree {
                self.inner.n()
            }

            fn k(&self) -> TorusPrecision {
                self.inner.k()
            }

            fn base2k(&self) -> Base2K {
                self.inner.base2k()
            }

            fn max_size(&self) -> usize {
                self.inner.max_size()
            }
        }

        impl<D: Data, W: ZnxWord> GLWEInfos for $name<D, W> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }

        impl<BE: Backend, D: Data> GLWEToBackendRef<BE>
            for $name<D, BE::ZnxWord>
        where
            GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
        {
            fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
                self.inner.to_backend_ref()
            }
        }

        impl<BE: Backend, D: Data> GLWEToBackendMut<BE>
            for $name<D, BE::ZnxWord>
        where
            GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        {
            fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
                GLWEToBackendMut::<BE>::set_noise(&mut self.inner, metadata);
            }

            fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
                self.inner.to_backend_mut()
            }

            fn set_canonical(&mut self, canonical: bool) {
                GLWEToBackendMut::<BE>::set_canonical(&mut self.inner, canonical)
            }
        }

        impl<D: HostDataRef, W: ZnxWord> fmt::Debug for $name<D, W> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}: {:?}", stringify!($name), self.inner)
            }
        }

        impl<D: HostDataMut, W: ZnxWord> ReaderFrom for $name<D, W> {
            fn read_from<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
                self.inner.read_from(reader)
            }
        }

        impl<D: HostDataRef, W: ZnxWord> WriterTo for $name<D, W> {
            fn write_to<Wr: Write>(&self, writer: &mut Wr) -> io::Result<()> {
                self.inner.write_to(writer)
            }
        }
    };
}

pub(crate) use glwe_share;
