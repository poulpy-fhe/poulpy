//! What a GLWE's plaintext is, stored as `Option<GLWEPlaintextMeta>`: schemes set
//! and read it, `None` claims nothing. Core operations never read or write it,
//! and layout conversions (views, clones, transfers, serialization,
//! decompression) carry it.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Scale of the encoded message; the default is unscaled, `Log(0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scale {
    /// Scaled by `2^x`, as in CKKS.
    Log(usize),
}

impl Default for Scale {
    fn default() -> Self {
        Self::Log(0)
    }
}

/// Subring the slots are known to live in. The variants are ordered claims,
/// `Integer ⊂ Real ⊂ Complex`: `Complex` is always sound and is the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SlotsKind {
    /// Every slot is an integer.
    Integer,
    /// Every slot has a zero imaginary part.
    Real,
    /// Slots may carry a nonzero imaginary part.
    #[default]
    Complex,
}

impl SlotsKind {
    /// Kind of a value built from two operands: the coarser one.
    pub fn join(self, other: Self) -> Self {
        self.max(other)
    }

    /// Kind of a value that lies in both: the finer one.
    pub fn meet(self, other: Self) -> Self {
        self.min(other)
    }

    /// Whether the slots are known to be real.
    pub fn is_real(self) -> bool {
        self != Self::Complex
    }
}

/// Metadata of the plaintext a GLWE encrypts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GLWEPlaintextMeta {
    pub scale: Scale,
    pub slots: SlotsKind,
    /// The plaintext lies in `Z[X^(2^log_sparsity)]`.
    pub log_sparsity: usize,
}

impl GLWEPlaintextMeta {
    pub(crate) fn write_to<W: std::io::Write>(meta: &Option<Self>, writer: &mut W) -> std::io::Result<()> {
        let Some(meta) = meta else {
            return writer.write_u8(0);
        };
        let Scale::Log(x) = meta.scale;
        writer.write_u8(1)?;
        writer.write_u64::<LittleEndian>(x as u64)?;
        writer.write_u8(match meta.slots {
            SlotsKind::Integer => 0,
            SlotsKind::Real => 1,
            SlotsKind::Complex => 2,
        })?;
        writer.write_u32::<LittleEndian>(meta.log_sparsity as u32)
    }

    /// Reads metadata; validate it with [`Self::validate_degree`] once the
    /// serialized plaintext degree is known.
    pub(crate) fn read_from<R: std::io::Read>(reader: &mut R) -> std::io::Result<Option<Self>> {
        let invalid = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid plaintext {what}"));
        let scale = match reader.read_u8()? {
            0 => return Ok(None),
            1 => Scale::Log(reader.read_u64::<LittleEndian>()? as usize),
            _ => return Err(invalid("scale")),
        };
        let slots = match reader.read_u8()? {
            0 => SlotsKind::Integer,
            1 => SlotsKind::Real,
            2 => SlotsKind::Complex,
            _ => return Err(invalid("slots")),
        };
        let log_sparsity = reader.read_u32::<LittleEndian>()? as usize;
        Ok(Some(Self {
            scale,
            slots,
            log_sparsity,
        }))
    }

    pub(crate) fn validate_degree(&self, n: usize) -> std::io::Result<()> {
        // `Z[X^(2^s)]` needs `2^s <= n`.
        if n == 0 || self.log_sparsity > n.ilog2() as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid plaintext sparsity",
            ));
        }
        Ok(())
    }
}

/// Read access to the [`GLWEPlaintextMeta`] a scheme set, if any.
pub trait GLWEPlaintextInfos {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta>;

    /// The scale; unset reads as unscaled, `Log(0)`.
    fn scale(&self) -> Scale {
        self.plaintext_meta().unwrap_or_default().scale
    }

    /// The slot kind; unset reads as `Complex`, which always holds.
    fn slots(&self) -> SlotsKind {
        self.plaintext_meta().unwrap_or_default().slots
    }

    /// The sparsity; unset reads as `0`, which always holds.
    fn log_sparsity(&self) -> usize {
        self.plaintext_meta().unwrap_or_default().log_sparsity
    }
}

/// Write access to [`GLWEPlaintextMeta`]. A field setter on unset metadata
/// starts from [`GLWEPlaintextMeta::default`].
pub trait SetGLWEPlaintextInfos: GLWEPlaintextInfos {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>);

    fn set_scale(&mut self, scale: Scale) {
        let meta = self.plaintext_meta().unwrap_or_default();
        self.set_plaintext_meta(Some(GLWEPlaintextMeta { scale, ..meta }));
    }

    fn set_slots(&mut self, slots: SlotsKind) {
        let meta = self.plaintext_meta().unwrap_or_default();
        self.set_plaintext_meta(Some(GLWEPlaintextMeta { slots, ..meta }));
    }

    fn set_log_sparsity(&mut self, log_sparsity: usize) {
        let meta = self.plaintext_meta().unwrap_or_default();
        self.set_plaintext_meta(Some(GLWEPlaintextMeta { log_sparsity, ..meta }));
    }
}

impl<T: GLWEPlaintextInfos + ?Sized> GLWEPlaintextInfos for &T {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        (**self).plaintext_meta()
    }
}

impl<T: GLWEPlaintextInfos + ?Sized> GLWEPlaintextInfos for &mut T {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        (**self).plaintext_meta()
    }
}

impl<T: SetGLWEPlaintextInfos + ?Sized> SetGLWEPlaintextInfos for &mut T {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        (**self).set_plaintext_meta(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_kind_is_ordered() {
        use SlotsKind::{Complex, Integer, Real};
        assert_eq!(Integer.join(Real), Real);
        assert_eq!(Real.join(Complex), Complex);
        assert_eq!(Integer.meet(Complex), Integer);
        assert!(Integer.is_real() && Real.is_real() && !Complex.is_real());
    }

    #[test]
    fn serialization_round_trips_and_rejects_impossible_sparsity() {
        let sparse = GLWEPlaintextMeta {
            scale: Scale::Log(40),
            slots: SlotsKind::Real,
            log_sparsity: 3,
        };
        let integer = GLWEPlaintextMeta {
            slots: SlotsKind::Integer,
            ..Default::default()
        };
        for meta in [None, Some(sparse), Some(integer)] {
            let mut bytes = Vec::new();
            GLWEPlaintextMeta::write_to(&meta, &mut bytes).unwrap();
            let decoded = GLWEPlaintextMeta::read_from(&mut bytes.as_slice()).unwrap();
            assert_eq!(decoded, meta);
            if let Some(decoded) = decoded {
                decoded.validate_degree(8).unwrap();
            }
        }
        let mut bytes = Vec::new();
        GLWEPlaintextMeta::write_to(
            &Some(GLWEPlaintextMeta {
                log_sparsity: 4,
                ..sparse
            }),
            &mut bytes,
        )
        .unwrap();
        assert_eq!(
            GLWEPlaintextMeta::read_from(&mut bytes.as_slice())
                .unwrap()
                .unwrap()
                .validate_degree(8)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}
