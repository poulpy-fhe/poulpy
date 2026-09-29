//! Protocol shares that are one unseeded GLWE transcript, a core `GLWE`.

/// Defines a protocol share wrapping one core `GLWE`, with its infos,
/// backend borrows and serialization forwarded to it.
macro_rules! glwe_share {
    ($(#[$doc:meta])* $name:ident, $owned:ident) => {
        pub type $owned<BE> = $name<<BE as poulpy_hal::layouts::Backend>::OwnedBuf, <BE as poulpy_hal::layouts::Backend>::ZnxWord>;

        $(#[$doc])*
        ///
        /// Serializes as its core `GLWE`.
        #[derive(Clone)]
        pub struct $name<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> {
            pub(crate) inner: poulpy_core::layouts::GLWE<D, W>,
        }

        impl<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> PartialEq for $name<D, W>
        where
            poulpy_core::layouts::GLWE<D, W>: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.inner == other.inner
            }
        }

        impl<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> Eq for $name<D, W> where
            poulpy_core::layouts::GLWE<D, W>: Eq
        {
        }

        impl<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> poulpy_core::layouts::LWEInfos for $name<D, W> {
            fn n(&self) -> poulpy_core::layouts::Degree {
                self.inner.n()
            }

            fn k(&self) -> poulpy_core::layouts::TorusPrecision {
                self.inner.k()
            }

            fn base2k(&self) -> poulpy_core::layouts::Base2K {
                self.inner.base2k()
            }

            fn max_size(&self) -> usize {
                self.inner.max_size()
            }
        }

        impl<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> poulpy_core::layouts::GLWEInfos for $name<D, W> {
            fn rank(&self) -> poulpy_core::layouts::Rank {
                self.inner.rank()
            }
        }

        impl<BE: poulpy_hal::layouts::Backend, D: poulpy_hal::layouts::Data> poulpy_core::layouts::GLWEToBackendRef<BE>
            for $name<D, BE::ZnxWord>
        where
            poulpy_core::layouts::GLWE<D, BE::ZnxWord>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        {
            fn to_backend_ref(&self) -> poulpy_core::layouts::GLWEBackendRef<'_, BE> {
                self.inner.to_backend_ref()
            }
        }

        impl<BE: poulpy_hal::layouts::Backend, D: poulpy_hal::layouts::Data> poulpy_core::layouts::GLWEToBackendMut<BE>
            for $name<D, BE::ZnxWord>
        where
            poulpy_core::layouts::GLWE<D, BE::ZnxWord>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        {
            fn to_backend_mut(&mut self) -> poulpy_core::layouts::GLWEBackendMut<'_, BE> {
                self.inner.to_backend_mut()
            }

            fn set_canonical(&mut self, canonical: bool) {
                poulpy_core::layouts::GLWEToBackendMut::<BE>::set_canonical(&mut self.inner, canonical)
            }
        }

        impl<D: poulpy_hal::layouts::HostDataRef, W: poulpy_hal::layouts::ZnxWord> std::fmt::Debug for $name<D, W> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{}: {:?}", stringify!($name), self.inner)
            }
        }

        impl<D: poulpy_hal::layouts::HostDataMut, W: poulpy_hal::layouts::ZnxWord> poulpy_hal::layouts::ReaderFrom for $name<D, W> {
            fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
                self.inner.read_from(reader)
            }
        }

        impl<D: poulpy_hal::layouts::HostDataRef, W: poulpy_hal::layouts::ZnxWord> poulpy_hal::layouts::WriterTo for $name<D, W> {
            fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
                self.inner.write_to(writer)
            }
        }
    };
}

pub(crate) use glwe_share;
