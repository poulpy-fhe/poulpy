use poulpy_core::api::TransferInto;
use poulpy_core::layouts::{Base2K, Degree, GLWE, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision};
use poulpy_hal::layouts::{CopyFromHost, CopyToHost, Data, ZnxWord};

/// Specifies in which direction the LUT is rotated by the LWE constant term
/// during blind rotation.
///
/// By default (`Left`) the rotation is by `X^{-dec(lwe)}`, which is the
/// standard convention for blind rotation: decoding the result at the constant
/// coefficient yields `f(dec(lwe))`.  Setting `Right` reverses the sign of the
/// LWE constant-term contribution, which is useful when the encryption of the
/// exponent is desired instead of the function value (as in circuit
/// bootstrapping's `execute_to_exponent` mode).
#[derive(Debug, Clone, Copy)]
pub enum LookUpTableRotationDirection {
    /// Rotate by `X^{-dec(lwe)}` (standard).
    Left,
    /// Rotate by `X^{+dec(lwe)}` (reversed).
    Right,
}

/// Plain-old-data descriptor used to allocate a [`LookupTable<BE::OwnedBuf, BE::ZnxWord>`].
///
/// All fields are public and must be consistent:
/// - `n` is the GLWE polynomial degree.
/// - `extension_factor` must be a non-zero power of two; a value of 1
///   yields the classical single-polynomial LUT, larger values split the
///   table across `extension_factor` polynomials giving an effective domain
///   size of `n × extension_factor`.
/// - `k` is the torus precision (total number of message bits).
/// - `base2k` is the decomposition base (number of bits per limb).
pub struct LookUpTableLayout {
    pub n: Degree,
    pub extension_factor: usize,
    pub k: TorusPrecision,
    pub base2k: Base2K,
}

/// Accessor trait for the dimensional parameters of a lookup table or its
/// descriptor.
///
/// Implemented by both [`LookUpTableLayout`] and [`LookupTable<BE::OwnedBuf, BE::ZnxWord>`].
pub trait LookupTableInfos {
    /// GLWE polynomial degree `N`.
    fn n(&self) -> Degree;
    /// Number of polynomials the LUT is split across (must be a power of two).
    fn extension_factor(&self) -> usize;
    /// Total torus precision `k` (message bits).
    fn k(&self) -> TorusPrecision;
    /// Decomposition base (bits per limb).
    fn base2k(&self) -> Base2K;
    /// Number of limbs: `ceil(k / base2k)`.
    fn size(&self) -> usize;
}

impl LookupTableInfos for LookUpTableLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn extension_factor(&self) -> usize {
        self.extension_factor
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.k().as_usize().div_ceil(self.base2k().as_usize())
    }

    fn n(&self) -> Degree {
        self.n
    }
}

/// An encoded lookup table ready for use in blind rotation.
///
/// A `LookupTable<BE::OwnedBuf, BE::ZnxWord>` stores a function `f : Z_{domain_size} -> T_q` encoded as
/// one or more `VecZnx` polynomials in a representation compatible with the
/// blind-rotation accumulator update.  When `extension_factor > 1` the domain
/// is split across `extension_factor` polynomials of degree `n`, giving an
/// effective domain of size `n × extension_factor`.
///
/// ## Construction
///
/// Use [`LookupTable::alloc`] to allocate storage, then [`LookupTable::set`]
/// to encode the function values.  The `set` method handles scaling and
/// polynomial-domain encoding internally; callers supply integer-valued
/// function samples `f[i]`.
///
/// ## Rotation Direction
///
/// By default the table is configured for left rotation (standard decoding).
/// Call [`LookupTable::set_rotation_direction`] before evaluation to switch
/// to right rotation when the exponent encoding is needed.
///
/// ## Invariants
///
/// - `data` is non-empty; its length equals `extension_factor`.
/// - All `VecZnx` elements share the same `n`, `base2k`, and `size`.
/// - `drift` records the half-step pre-rotation applied during encoding.
pub struct LookupTable<D: Data, W: ZnxWord> {
    pub(crate) data: Vec<GLWE<D, W>>,
    pub(crate) rot_dir: LookUpTableRotationDirection,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) drift: usize,
}

impl<D: Data, W: ZnxWord> LookupTableInfos for LookupTable<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn extension_factor(&self) -> usize {
        self.data.len()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.data[0].n()
    }

    fn size(&self) -> usize {
        self.data[0].size()
    }
}

pub use crate::api::blind_rotation::LookupTableFactory;

impl<D: Data, W: ZnxWord> LookupTable<D, W> {
    /// Returns `log2(extension_factor)`.
    pub fn log_extension_factor(&self) -> usize {
        (usize::BITS - (self.extension_factor() - 1).leading_zeros()) as _
    }

    /// Returns the number of polynomials in the table (the extension factor).
    pub fn extension_factor(&self) -> usize {
        self.data.len()
    }

    /// Returns the total number of coefficients across all polynomials:
    /// `extension_factor × n`.
    pub fn domain_size(&self) -> usize {
        self.data.len() * self.data[0].n().as_usize()
    }

    /// Returns the currently configured rotation direction.
    pub fn rotation_direction(&self) -> LookUpTableRotationDirection {
        self.rot_dir
    }
}

impl<D: Data, W: ZnxWord> LookupTable<D, W> {
    /// Allocates a zero-initialised `LookupTable<BE::OwnedBuf, BE::ZnxWord>` with dimensions taken from
    /// `infos`.
    ///
    /// # Panics
    ///
    /// Panics if `infos.extension_factor()` is zero or not a power of two.
    pub fn alloc<M, A>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>,
        A: LookupTableInfos,
    {
        assert!(
            infos.extension_factor() > 0 && infos.extension_factor().is_power_of_two(),
            "extension_factor must be a non-zero power of two, got: {}",
            infos.extension_factor()
        );
        LookupTable {
            data: (0..infos.extension_factor())
                .map(|_| module.glwe_alloc(infos.base2k(), infos.k(), Rank(0)))
                .collect(),
            base2k: infos.base2k(),
            k: infos.k(),
            drift: 0,
            rot_dir: LookUpTableRotationDirection::Left,
        }
    }

    /// Overrides the rotation direction used during blind rotation.
    ///
    /// By default (`Left`) the rotation is `X^{-dec(lwe)}`, which decodes the
    /// function value at coefficient 0 of the result.  Set `Right` to compute
    /// `X^{+dec(lwe)}` instead (used in circuit bootstrapping's exponent mode).
    pub fn set_rotation_direction(&mut self, rot_dir: LookUpTableRotationDirection) {
        self.rot_dir = rot_dir
    }

    /// Encodes the function `f` into this lookup table using the given module.
    ///
    /// Delegates to [`LookupTableFactory::lookup_table_set`].
    ///
    /// `k` is the number of message bits to encode (e.g., `1` for a binary
    /// outcome, `res.base2k * res.size()` for the full precision).
    pub fn set<M>(&mut self, module: &M, f: &[i64], k: usize)
    where
        M: LookupTableFactory<D, W>,
    {
        module.lookup_table_set(self, f, k);
    }

    pub(crate) fn rotate<M>(&mut self, module: &M, k: i64)
    where
        M: LookupTableFactory<D, W>,
    {
        module.lookup_table_rotate(k, self);
    }
}

/// Uploads a host-built table into an already-allocated destination.
///
/// The scalars travel with the limbs: `drift` and `rot_dir` are set by
/// [`LookupTable::set`], and a destination left at its `alloc` defaults would
/// blind-rotate by the wrong offset.
impl<D1, D2, W> TransferInto<LookupTable<D2, W>> for LookupTable<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut LookupTable<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: LookupTable base2k");
        assert_eq!(self.k, dst.k, "transfer_into: LookupTable k");
        assert_eq!(self.data.len(), dst.data.len(), "transfer_into: LookupTable extension_factor");
        for (src, dst) in self.data.iter().zip(dst.data.iter_mut()) {
            src.transfer_into(dst);
        }
        dst.rot_dir = self.rot_dir;
        dst.drift = self.drift;
    }
}

pub(crate) trait DivRound {
    fn div_round(self, rhs: Self) -> Self;
}

impl DivRound for usize {
    #[inline]
    fn div_round(self, rhs: Self) -> Self {
        (self + rhs / 2) / rhs
    }
}

impl<D: Data, W: ZnxWord> LookupTable<D, W> {
    /// Coefficient-domain polynomials available to backend execution overrides.
    pub fn polynomials(&self) -> &[GLWE<D, W>] {
        &self.data
    }

    /// Mutable coefficient-domain polynomials for backend execution overrides.
    pub fn polynomials_mut(&mut self) -> &mut [GLWE<D, W>] {
        &mut self.data
    }

    /// Half-step pre-rotation applied while encoding the function samples.
    pub fn drift(&self) -> usize {
        self.drift
    }

    /// Records the half-step pre-rotation applied by a backend encoding override.
    pub fn set_drift(&mut self, drift: usize) {
        self.drift = drift;
    }
}
