use std::io::{Error, ErrorKind, Read, Result, Write};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::layouts::Ring;

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

/// Writes [`Ring::WIRE_TAG`] of `Rg`, the first byte of a coefficient-layout header.
pub(crate) fn write_ring_tag<Rg: Ring, Wr: Write>(writer: &mut Wr) -> Result<()> {
    writer.write_all(&[Rg::WIRE_TAG])
}

/// Reads the ring tag of a coefficient-layout header, refusing another ring's than `Rg`'s.
pub(crate) fn read_ring_tag<Rg: Ring, R: Read>(reader: &mut R) -> Result<()> {
    if reader.read_u8()? != Rg::WIRE_TAG {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "the serialized ring differs from the receiver's",
        ));
    }
    Ok(())
}

/// Reads `N` little-endian `u64` header fields, refusing one that exceeds `usize`.
pub(crate) fn read_usize_fields<const N: usize, R: Read>(reader: &mut R) -> Result<[usize; N]> {
    let mut fields = [0usize; N];
    for field in &mut fields {
        *field = usize::try_from(reader.read_u64::<LittleEndian>()?)
            .map_err(|_| Error::new(ErrorKind::InvalidData, "a serialized dimension exceeds usize"))?;
    }
    Ok(fields)
}

/// Bytes of the `W` words a product of `dimensions` holds, refusing a product that overflows `usize`.
pub(crate) fn serialized_byte_len<W>(dimensions: &[usize]) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(size_of::<W>(), |len, &dimension| len.checked_mul(dimension))
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "the serialized byte size overflows usize"))
}
