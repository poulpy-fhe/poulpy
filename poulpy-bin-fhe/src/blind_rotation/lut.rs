use poulpy_core::layouts::{Base2K, Degree, GLWE, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision};
use poulpy_cpu_ref::reference::znx::{ZnxCopy, ZnxRef, ZnxRotate, ZnxSwitchRing};
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VecZnxRotateAssignBackend,
        VecZnxRotateAssignTmpBytes,
    },
    layouts::{
        Backend, Data, HostDataRef, Module, ScratchOwned, TransferFrom, VecZnx, VecZnxToBackendMut, ZnxViewMut,
        vec_znx_host_backend_mut,
    },
};

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

/// Plain-old-data descriptor used to allocate a [`LookupTable`].
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
/// Implemented by both [`LookUpTableLayout`] and [`LookupTable`].
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
/// A `LookupTable` stores a function `f : Z_{domain_size} -> T_q` encoded as
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
pub struct LookupTable<D: Data = Vec<u8>> {
    pub(crate) data: Vec<GLWE<D>>,
    pub(crate) rot_dir: LookUpTableRotationDirection,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) drift: usize,
}

impl<D: Data> LookupTableInfos for LookupTable<D> {
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

/// Backend-level operations for filling and rotating a [`LookupTable`].
///
/// This trait is implemented for `Module<BE>` when the backend supports the
/// required polynomial operations.  Callers interact with the higher-level
/// [`LookupTable::set`] and `rotate` methods rather than calling these
/// directly.
pub trait LookupTableFactory {
    /// Encode the function `f` into `res`, scaling by the appropriate power of
    /// the decomposition base so that the most significant limb carries the
    /// message.
    ///
    /// `k` is the message-bit count (e.g., 1 for a binary-valued LUT).
    /// `f` must have length at most `res.domain_size()`.
    fn lookup_table_set(&self, res: &mut LookupTable<Vec<u8>>, f: &[i64], k: usize);

    /// Rotate the lookup table in-place by `k` positions in the ring
    /// `Z[X] / (X^{domain_size} + 1)`.
    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable<Vec<u8>>);
}

impl LookupTable<Vec<u8>> {
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

impl LookupTable<Vec<u8>> {
    /// Allocates a zero-initialised `LookupTable` with dimensions taken from
    /// `infos`.
    ///
    /// # Panics
    ///
    /// Panics if `infos.extension_factor()` is zero or not a power of two.
    pub fn alloc<M, A>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
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
        M: LookupTableFactory,
    {
        module.lookup_table_set(self, f, k);
    }

    pub(crate) fn rotate<M>(&mut self, module: &M, k: i64)
    where
        M: LookupTableFactory,
    {
        module.lookup_table_rotate(k, self);
    }
}

impl<D: HostDataRef> LookupTable<D> {
    pub fn to_backend<From, To>(&self, dst: &Module<To>) -> LookupTable<To::OwnedBuf>
    where
        From: Backend<OwnedBuf = D>,
        To: Backend + TransferFrom<From>,
    {
        LookupTable {
            data: self.data.iter().map(|glwe| glwe.to_backend::<From, To>(dst)).collect(),
            rot_dir: self.rot_dir,
            base2k: self.base2k,
            k: self.k,
            drift: self.drift,
        }
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

#[allow(dead_code)]
fn max_bit_size(vec: &[i64]) -> u32 {
    vec.iter()
        .map(|&v| if v == 0 { 0 } else { v.unsigned_abs().ilog2() + 1 })
        .max()
        .unwrap_or(0)
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> LookupTableFactory for Module<BE>
where
    Self: VecZnxRotateAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxRotateAssignTmpBytes
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn lookup_table_set(&self, res: &mut LookupTable<Vec<u8>>, f: &[i64], k: usize) {
        // TODO: add a direct backend-native LUT construction path. This builder
        // still materializes host-owned `Vec<u8>` polynomials before any upload.
        assert!(f.len() <= self.n());

        let base2k: usize = res.base2k.into();

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.vec_znx_normalize_tmp_bytes().max(res.domain_size() << 3));

        // Get the number minimum limb to store the message modulus
        let limbs: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            assert!(
                (max_bit_size(f) + (k % base2k) as u32) < i64::BITS,
                "overflow: max(|f|) << (k%base2k) > i64::BITS"
            );
            assert!(limbs <= res.data[0].size());
        }

        // Scaling factor
        let mut scale = 1;
        if !k.is_multiple_of(base2k) {
            scale <<= base2k - (k % base2k);
        }

        // #elements in lookup table
        let f_len: usize = f.len();

        // If LUT size > TakeScalarZnx
        let domain_size: usize = res.domain_size();

        let size: usize = res.k.as_usize().div_ceil(base2k);

        // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
        // but staged purely in host vectors so we do not need to allocate a
        // second module with ring degree `domain_size`.
        let mut lut_full_limbs: Vec<Vec<i64>> = (0..size).map(|_| vec![0i64; domain_size]).collect();

        let lut_at: &mut [i64] = &mut lut_full_limbs[limbs - 1];

        let step: usize = domain_size.div_round(f_len);

        for (i, fi) in f.iter().enumerate() {
            let start: usize = i * step;
            let end: usize = start + step;
            lut_at[start..end].fill(fi * scale);
        }

        let drift: usize = step >> 1;

        // Rotates half the step to the left
        if res.extension_factor() > 1 {
            let mut tmp: Vec<i64> = vec![0i64; domain_size];

            for i in 0..res.extension_factor() {
                let mut res_at = vec_znx_host_backend_mut(res.data[i].data_mut());
                for (limb, limb_data) in lut_full_limbs.iter().enumerate().take(res_at.size()) {
                    ZnxRef::znx_switch_ring(res_at.at_mut(0, limb), limb_data);
                }
                if i + 1 < res.extension_factor() {
                    for limb_data in &mut lut_full_limbs {
                        ZnxRef::znx_rotate(-1, &mut tmp, limb_data);
                        ZnxRef::znx_copy(limb_data, &tmp);
                    }
                }
            }
        } else {
            let mut res_at = vec_znx_host_backend_mut(res.data[0].data_mut());
            for (limb, limb_data) in lut_full_limbs.iter().enumerate().take(res_at.size()) {
                res_at.at_mut(0, limb).copy_from_slice(limb_data);
            }
        }

        for a in res.data.iter_mut() {
            let mut a_data = <VecZnx<Vec<u8>> as VecZnxToBackendMut<BE>>::to_backend_mut(a.data_mut());
            self.vec_znx_normalize_assign_backend(res.base2k.into(), &mut a_data, 0, &mut scratch.borrow());
        }

        res.rotate(self, -(drift as i64));

        res.drift = drift
    }

    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable<Vec<u8>>) {
        // TODO: make LUT rotation backend-native once lookup tables can be
        // constructed and stored directly in backend-owned buffers.
        let extension_factor: usize = res.extension_factor();
        let two_n: usize = 2 * res.data[0].n().as_usize();
        let two_n_ext: usize = two_n * extension_factor;

        let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(self.vec_znx_rotate_assign_tmp_bytes());

        let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

        let k_hi: usize = k_pos / extension_factor;
        let k_lo: usize = k_pos % extension_factor;

        (0..extension_factor - k_lo).for_each(|i| {
            let mut data: poulpy_hal::layouts::VecZnxBackendMut<'_, BE> =
                <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(res.data[i].data_mut());
            self.vec_znx_rotate_assign_backend(k_hi as i64, &mut data, 0, &mut scratch.borrow());
        });

        (extension_factor - k_lo..extension_factor).for_each(|i| {
            let mut data: poulpy_hal::layouts::VecZnxBackendMut<'_, BE> =
                <poulpy_hal::layouts::VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(res.data[i].data_mut());
            self.vec_znx_rotate_assign_backend(k_hi as i64 + 1, &mut data, 0, &mut scratch.borrow());
        });

        res.data.rotate_right(k_lo);
    }
}
