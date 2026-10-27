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

/// Which slots carry data and the shape of the error on them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SlotLayout {
    /// Every slot carries data and errors are unshaped.
    #[default]
    Full,
    /// Data on one slot class out of `R`, errors multiples of a shaping polynomial.
    Replicated(ReplicatedLayout),
}

/// Replicated layout: with `R = 2^log_r`, slot `t` carries data iff `R | t`,
/// holding value `(t/R) mod 2^log_k`, so the batch of `2^log_k` values repeats
/// every `2^log_k * R` slots and the other slots are guards. Errors are
/// multiples of `g = U(X^(N/(2R)))`, where `U ∈ Z[Y]/(Y^(2R)+1)` is invertible
/// modulo every power of two, and shaped rounding uses `w` extra working bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplicatedLayout {
    log_k: u8,
    log_r: u8,
    w: u8,
    shaping: [i16; 2 << ReplicatedLayout::MAX_LOG_R],
}

impl ReplicatedLayout {
    /// Largest supported `log_r`.
    pub const MAX_LOG_R: usize = 5;

    /// Checks `1 <= log_r <= MAX_LOG_R`, that `shaping` holds the `2R`
    /// coefficients of `U`, and that `U(1)` is odd, which makes `U` invertible
    /// modulo every power of two.
    pub fn new(log_k: u8, log_r: u8, w: u8, shaping: &[i16]) -> crate::Result<Self> {
        let invalid = |detail| Err(crate::CoreError::InvalidReplicatedLayout { detail });
        if log_r == 0 || log_r as usize > Self::MAX_LOG_R {
            return invalid("log_r must lie in 1..=MAX_LOG_R");
        }
        if log_k as u32 + log_r as u32 >= usize::BITS - 1 {
            return invalid("2kR must fit in a usize");
        }
        if shaping.len() != 2 << log_r {
            return invalid("the shaping polynomial must have 2R coefficients");
        }
        if shaping.iter().map(|&c| c as i64).sum::<i64>() & 1 == 0 {
            return invalid("the shaping polynomial must be invertible modulo 2");
        }
        let mut coeffs = [0; 2 << Self::MAX_LOG_R];
        coeffs[..shaping.len()].copy_from_slice(shaping);
        Ok(Self {
            log_k,
            log_r,
            w,
            shaping: coeffs,
        })
    }

    /// Base-2 logarithm of the batch size `k`.
    pub fn log_k(&self) -> usize {
        self.log_k as usize
    }

    /// Base-2 logarithm of the slot period `R`.
    pub fn log_r(&self) -> usize {
        self.log_r as usize
    }

    /// Extra working bits of shaped rounding.
    pub fn w(&self) -> usize {
        self.w as usize
    }

    /// The `2R` coefficients of `U`.
    pub fn shaping(&self) -> &[i16] {
        &self.shaping[..2 << self.log_r]
    }
}

/// Metadata of the plaintext a GLWE encrypts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GLWEPlaintextMeta {
    pub scale: Scale,
    pub slots: SlotsKind,
    /// The plaintext lies in `Z[X^(2^log_sparsity)]`.
    pub log_sparsity: usize,
    pub slot_layout: SlotLayout,
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
        writer.write_u32::<LittleEndian>(meta.log_sparsity as u32)?;
        match meta.slot_layout {
            SlotLayout::Full => writer.write_u8(0),
            SlotLayout::Replicated(layout) => {
                writer.write_u8(1)?;
                writer.write_all(&[layout.log_k, layout.log_r, layout.w])?;
                layout.shaping().iter().try_for_each(|&c| writer.write_i16::<LittleEndian>(c))
            }
        }
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
        let slot_layout = match reader.read_u8()? {
            0 => SlotLayout::Full,
            1 => {
                let mut params = [0u8; 3];
                reader.read_exact(&mut params)?;
                let [log_k, log_r, w] = params;
                if log_r as usize > ReplicatedLayout::MAX_LOG_R {
                    return Err(invalid("slot layout"));
                }
                let mut shaping = [0i16; 2 << ReplicatedLayout::MAX_LOG_R];
                reader.read_i16_into::<LittleEndian>(&mut shaping[..2 << log_r])?;
                let layout =
                    ReplicatedLayout::new(log_k, log_r, w, &shaping[..2 << log_r]).map_err(|_| invalid("slot layout"))?;
                SlotLayout::Replicated(layout)
            }
            _ => return Err(invalid("slot layout")),
        };
        Ok(Some(Self {
            scale,
            slots,
            log_sparsity,
            slot_layout,
        }))
    }

    pub(crate) fn validate_degree(&self, n: usize) -> std::io::Result<()> {
        let invalid = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid plaintext {what}"));
        // `Z[X^(2^s)]` needs `2^s <= n`.
        if n == 0 || self.log_sparsity > n.ilog2() as usize {
            return Err(invalid("sparsity"));
        }
        // A batch of `k` values with period `R` needs `kR <= N/2` slots.
        if let SlotLayout::Replicated(layout) = self.slot_layout
            && layout.log_k() + layout.log_r() >= n.ilog2() as usize
        {
            return Err(invalid("slot layout"));
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

    /// The slot layout; unset reads as `Full`.
    fn slot_layout(&self) -> SlotLayout {
        self.plaintext_meta().unwrap_or_default().slot_layout
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

    fn set_slot_layout(&mut self, slot_layout: SlotLayout) {
        let meta = self.plaintext_meta().unwrap_or_default();
        self.set_plaintext_meta(Some(GLWEPlaintextMeta { slot_layout, ..meta }));
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
            slot_layout: SlotLayout::Full,
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

    #[test]
    fn replicated_layout_checks_its_invariants() {
        let layout = ReplicatedLayout::new(2, 1, 8, &[1, -1, 0, 1]).unwrap();
        assert_eq!((layout.log_k(), layout.log_r(), layout.w()), (2, 1, 8));
        assert_eq!(layout.shaping(), &[1, -1, 0, 1]);
        assert!(ReplicatedLayout::new(2, 0, 8, &[1, 0]).is_err());
        assert!(ReplicatedLayout::new(2, ReplicatedLayout::MAX_LOG_R as u8 + 1, 8, &[1; 128]).is_err());
        assert!(ReplicatedLayout::new(2, 1, 8, &[1, 0, 0]).is_err());
        assert!(ReplicatedLayout::new(2, 1, 8, &[1, 1, 0, 0]).is_err());
        assert!(ReplicatedLayout::new((usize::BITS - 2) as u8, 1, 8, &[1, -1, 0, 1]).is_err());
        assert!(ReplicatedLayout::new((usize::BITS - 3) as u8, 1, 8, &[1, -1, 0, 1]).is_ok());
    }

    #[test]
    fn replicated_layout_round_trips_and_rejects_impossible_degree() {
        let shaping: Vec<i16> = (0..64).map(|i| if i == 0 { 3 } else { (i % 5) as i16 - 2 }).collect();
        for (log_r, shaping) in [(1, &[1, -1, 0, 1][..]), (5, &shaping[..])] {
            let meta = GLWEPlaintextMeta {
                scale: Scale::Log(20),
                slot_layout: SlotLayout::Replicated(ReplicatedLayout::new(3, log_r, 24, shaping).unwrap()),
                ..Default::default()
            };
            let mut bytes = Vec::new();
            GLWEPlaintextMeta::write_to(&Some(meta), &mut bytes).unwrap();
            let decoded = GLWEPlaintextMeta::read_from(&mut bytes.as_slice()).unwrap().unwrap();
            assert_eq!(decoded, meta);
            decoded.validate_degree(2 << (3 + log_r)).unwrap();
            assert_eq!(
                decoded.validate_degree(1 << (3 + log_r)).unwrap_err().kind(),
                std::io::ErrorKind::InvalidData
            );

            // Flipping the low bit of the constant coefficient flips the parity of `U(1)`.
            let constant = bytes.len() - 2 * shaping.len();
            bytes[constant] ^= 1;
            assert!(GLWEPlaintextMeta::read_from(&mut bytes.as_slice()).is_err());
        }
    }
}

#[cfg(test)]
mod reader_tests {
    use std::io::ErrorKind;

    use poulpy_hal::layouts::{HostBytesBackend, Module, ReaderFrom, WriterTo};

    use crate::layouts::{
        Base2K, Degree, GLWELayout, GLWEPlaintextMeta, LWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank,
        ReplicatedLayout, Scale, SetGLWEPlaintextInfos, SetK, SlotLayout, TorusPrecision,
    };

    fn layout(n: u32, k: u32) -> GLWELayout {
        GLWELayout {
            n: Degree(n),
            base2k: Base2K(16),
            k: TorusPrecision(k),
            rank: Rank(1),
        }
    }

    fn replicated_meta() -> GLWEPlaintextMeta {
        GLWEPlaintextMeta {
            scale: Scale::Log(8),
            slot_layout: SlotLayout::Replicated(
                ReplicatedLayout::new(3, 3, 16, &[1, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]).unwrap(),
            ),
            ..Default::default()
        }
    }

    fn assert_accepts_larger_serialized_degree<T>(mut source: T, mut receiver: T, meta: GLWEPlaintextMeta)
    where
        T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
    {
        assert_eq!(source.n(), Degree(128));
        assert_eq!(receiver.n(), Degree(64));
        assert_eq!(receiver.k(), source.k());
        source.set_plaintext_meta(Some(meta));
        let mut bytes = Vec::new();
        source.write_to(&mut bytes).unwrap();

        receiver.read_from(&mut bytes.as_slice()).unwrap();

        assert_eq!(receiver.n(), source.n());
        assert_eq!(receiver.k(), source.k());
        assert_eq!(receiver.plaintext_meta(), Some(meta));
        let mut round_trip = Vec::new();
        receiver.write_to(&mut round_trip).unwrap();
        assert!(round_trip == bytes, "a read plaintext serializes as the original");
    }

    fn assert_rejects_meta_exceeding_serialized_degree<T>(mut source: T, mut receiver: T, meta: GLWEPlaintextMeta, what: &str)
    where
        T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
    {
        assert_eq!(source.n(), Degree(64));
        assert_eq!(receiver.n(), Degree(128));
        assert_eq!(receiver.k(), source.k());
        // The metadata fits the receiver's initial degree, but not the stream's degree.
        source.set_plaintext_meta(Some(meta));
        let mut bytes = Vec::new();
        source.write_to(&mut bytes).unwrap();

        let error = receiver.read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains(what));
        assert!(receiver.plaintext_meta().is_none());
    }

    #[test]
    fn glwe_reader_round_trips_replicated_layout() {
        let module = Module::<HostBytesBackend>::new(128);
        let source = module.glwe_alloc_from_infos(&layout(128, 16));
        let mut receiver = module.glwe_alloc_from_infos(&layout(64, 32));
        receiver.set_k(TorusPrecision(16));
        assert_accepts_larger_serialized_degree(source, receiver, replicated_meta());
    }

    #[test]
    fn glwe_reader_rejects_replicated_layout_exceeding_smaller_serialized_degree() {
        let module = Module::<HostBytesBackend>::new(128);
        let source = module.glwe_alloc_from_infos(&layout(64, 16));
        let receiver = module.glwe_alloc_from_infos(&layout(128, 16));
        assert_rejects_meta_exceeding_serialized_degree(source, receiver, replicated_meta(), "slot layout");
    }

    #[test]
    fn compressed_glwe_reader_round_trips_replicated_layout() {
        let module = Module::<HostBytesBackend>::new(128);
        let source = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
        let mut receiver = module.glwe_compressed_alloc_from_infos(&layout(64, 32));
        receiver.k = TorusPrecision(16);
        assert_accepts_larger_serialized_degree(source, receiver, replicated_meta());
    }
}
