use std::io::{Read, Result, Write};

/// Serialize a layout type to a byte stream.
///
/// The wire format is type-specific and documented on each implementor.
/// All multi-byte integers are written in little-endian order.
///
/// In practice this is implemented only for host-readable layout storages.
pub trait WriterTo {
    /// Writes the complete serialized representation to `writer`.
    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()>;
}

/// Deserialize a layout type from a byte stream.
///
/// The receiver must be pre-allocated with enough capacity to hold the
/// incoming data. Metadata fields (dimensions, sizes) are updated
/// atomically after a successful read to avoid leaving the object in
/// an inconsistent state on I/O errors.
///
/// In practice this is implemented only for host-mutable layout storages.
pub trait ReaderFrom {
    /// Reads and overwrites `self` from `reader`.
    fn read_from<R: Read>(&mut self, reader: &mut R) -> Result<()>;
}
