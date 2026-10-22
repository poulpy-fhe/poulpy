use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
    MatZnxToBackendRef, ReaderFrom, WriterTo,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::{
    error::{CoreError, Result},
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWAtViewRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

pub trait GGLWEInfos
where
    Self: GLWEInfos,
{
    /// Auxiliary guard precision (in bits) stored below the gadget region and
    /// used for noise management during key operations. Free value, need not be
    /// a multiple of `base2k`. In practice `k_digit + logN`.
    fn k_aux(&self) -> TorusPrecision;
    /// Number of gadget-decomposition rows.
    fn dnum(&self) -> Dnum;
    /// Width in bits of one gadget digit: row `q` carries `2^-((q + 1) * k_digit)`.
    /// Free value, need not be a multiple of `base2k`.
    fn k_digit(&self) -> TorusPrecision;
    /// Limbs one digit spans at the key radix, `ceil(k_digit / base2k)`: the
    /// sub-limb count the interleaved-digit product runs with.
    fn dsize(&self) -> Dsize {
        Dsize(self.k_digit().0.div_ceil(self.base2k().0))
    }
    fn rank_in(&self) -> Rank;
    fn rank_out(&self) -> Rank;
    /// Row offset in bits: row `q` carries `2^-(k_offset + (q + 1) * k_digit)`,
    /// so a key switch lowers its input by `k_offset` bits. 0 for every key
    /// that is not built with one.
    fn k_offset(&self) -> TorusPrecision;
    /// Row stride into the stored matrix: digit `i` is stored row
    /// `(i + 1) * stride - 1`. A key used as stored has stride 1; a
    /// [`crate::layouts::GGLWEBackendRef`] coarsening it to `k_digit` reports
    /// `k_digit / stored k_digit`.
    fn stride(&self) -> usize {
        1
    }
    /// Whether the digits are whole limbs (see [`crate::layouts::digits_aligned`]).
    ///
    /// Decided on the width in use, `k_digit`, which for a view is `stride`
    /// stored digits read as one: that width sets the operand of a gadget
    /// product, the input limbs or their layout on the digit grid.
    fn digits_aligned(&self) -> bool {
        crate::layouts::digits_aligned(self.base2k(), self.k_digit())
    }
    fn gglwe_layout(&self) -> GGLWELayout {
        let stride = self.stride();
        GGLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            dnum: self.dnum(),
            k_aux: self.k_aux(),
            rank_in: self.rank_in(),
            rank_out: self.rank_out(),
            k_digit: self.k_digit(),
            k_offset: self.k_offset(),
            stride,
        }
    }
}

/// Every digit width this key can be read at, with the digits each reaches.
///
/// Ascending, always starting with the stored `(k_digit(), dnum())`.
pub(crate) fn valid_k_digits<A: GGLWEInfos + ?Sized>(infos: &A) -> Vec<(TorusPrecision, Dnum)> {
    let layout = infos.gglwe_layout();
    let stored: usize = layout.k_digit.as_usize();
    (1..=infos.dnum().as_usize())
        .filter_map(|stride| {
            let k_digit = TorusPrecision((stride * stored) as u32);
            layout.at_k_digit(k_digit).ok().map(|l| (k_digit, l.dnum))
        })
        .collect()
}

impl<T: GGLWEInfos + ?Sized> GGLWEInfos for &T {
    fn k_aux(&self) -> TorusPrecision {
        (**self).k_aux()
    }

    fn dnum(&self) -> Dnum {
        (**self).dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        (**self).k_digit()
    }

    fn rank_in(&self) -> Rank {
        (**self).rank_in()
    }

    fn rank_out(&self) -> Rank {
        (**self).rank_out()
    }

    fn k_offset(&self) -> TorusPrecision {
        (**self).k_offset()
    }

    fn stride(&self) -> usize {
        (**self).stride()
    }
}

impl<T: GGLWEInfos + ?Sized> GGLWEInfos for &mut T {
    fn k_aux(&self) -> TorusPrecision {
        (**self).k_aux()
    }

    fn dnum(&self) -> Dnum {
        (**self).dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        (**self).k_digit()
    }

    fn rank_in(&self) -> Rank {
        (**self).rank_in()
    }

    fn rank_out(&self) -> Rank {
        (**self).rank_out()
    }

    fn k_offset(&self) -> TorusPrecision {
        (**self).k_offset()
    }

    fn stride(&self) -> usize {
        (**self).stride()
    }
}

pub trait SetGGLWEInfos {
    fn set_k_digit(&mut self, k_digit: TorusPrecision);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWELayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub k_digit: TorusPrecision,
    /// Row offset in bits (see [`GGLWEInfos::k_offset`]).
    pub k_offset: TorusPrecision,
    /// Row stride for a coarsened view of a stored key. `1` means the key is
    /// used as stored.
    pub stride: usize,
}

impl GGLWELayout {
    /// This layout read at a coarser digit width, where digit `i` is stored row
    /// `(i + 1) * k_digit / self.k_digit - 1`.
    ///
    /// The last coarse digit still needs a full digit of padding behind it, so
    /// the digits the stride allows are capped by what the precision can guard
    /// and the rest stays as that guard: `k()` and `max_size()` do not move.
    ///
    /// Fails unless `k_digit` is a multiple of the stored one and the key is
    /// wide enough for a digit of that width plus its guard.
    pub fn at_k_digit(&self, k_digit: TorusPrecision) -> Result<Self> {
        let stored: usize = self.k_digit.as_usize();
        let fail = |detail: String| CoreError::GGLWEKeyUse {
            op: "GGLWELayout::at_k_digit",
            detail,
        };

        if stored == 0 || !k_digit.as_usize().is_multiple_of(stored) {
            return Err(fail(format!(
                "k_digit={k_digit} is not a multiple of the key's k_digit={stored}"
            )));
        }
        let stride: usize = k_digit.as_usize() / stored;
        if stride == 1 {
            return Ok(*self);
        }

        let (total, digit) = (self.k().as_usize() - self.k_offset.as_usize(), k_digit.as_usize());
        if total <= digit {
            return Err(fail(format!(
                "key holds {total} bits, too few for a digit of {digit} plus its guard"
            )));
        }
        let dnum: usize = (self.dnum.as_usize() / stride).min(total / digit - 1);
        if dnum == 0 {
            return Err(fail(format!("key has no digit reachable at k_digit={k_digit}")));
        }

        Ok(Self {
            dnum: Dnum(dnum as u32),
            k_digit,
            k_aux: TorusPrecision((total - dnum * digit) as u32),
            stride: self.stride * stride,
            ..*self
        })
    }
}

impl GGLWELayout {
    /// Number of gadget digits a key needs to cover an input ciphertext of
    /// precision `input_k`: `ceil(input_k / k_digit)`.
    ///
    /// This is the inverse of [`Self::gadget_k`] and the rounding rule used by
    /// [`key_work_size`](crate::layouts::key_work_size): the gadget is consumed
    /// in whole digits, so a partial digit counts as a full one. The auxiliary
    /// guard `k_aux` is chosen by the caller (typically `k_digit + log2(n)`)
    /// and is not part of this derivation. The digit width must be nonzero.
    pub fn dnum_for_input(input_k: TorusPrecision, k_digit: TorusPrecision) -> Dnum {
        debug_assert!(k_digit.0 > 0, "dnum_for_input requires a nonzero gadget digit");
        Dnum(input_k.0.div_ceil(k_digit.0))
    }

    /// Gadget precision of the key: `dnum * k_digit`, i.e. the width covered by
    /// its `dnum` digits, excluding the auxiliary guard `k_aux`. An input
    /// ciphertext of at most this many bits is decomposed entirely by the key.
    /// Any [`GGLWEInfos`] reaches it through [`GGLWEInfos::gglwe_layout`].
    pub fn gadget_k(&self) -> TorusPrecision {
        TorusPrecision(self.dnum.0 * self.k_digit().0)
    }

    /// Widest input the key switches whole: what its digits cover, and no more
    /// than its precision less its offset and one digit (see [`key_k`](crate::layouts::key_k)),
    /// the point where the key noise reaches `sigma * sqrt(rank_in * n * dnum / 12)`
    /// units of the input's last bit. A guard of a full digit and `log2(n)` bits
    /// keeps it below one unit over all its digits.
    pub fn max_input_k(&self) -> TorusPrecision {
        let below_offset = self.k().0.saturating_sub(self.k_offset.0);
        TorusPrecision(self.gadget_k().0.min(below_offset.saturating_sub(self.k_digit().0)))
    }
}

impl LWEInfos for GGLWELayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        crate::layouts::offset_key_size(self.base2k, self.k_offset, self.dnum, self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::offset_key_k(self.k_offset, self.dnum, self.k_digit, self.k_aux)
    }
}

impl GLWEInfos for GGLWELayout {
    fn rank(&self) -> Rank {
        self.rank_out
    }
}

impl GGLWEInfos for GGLWELayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn k_offset(&self) -> TorusPrecision {
        self.k_offset
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWE<D: Data, W: ZnxWord> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: MatZnx<D, W>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) k_digit: TorusPrecision,
    pub(crate) k_offset: TorusPrecision,
}

pub struct GGLWEBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEBackendRef<'a, BE> {
    type Target = GGLWE<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_mut::<BE>(&self.inner, row, col))
    }

    /// Clears the owner's component noise metadata.
    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(gglwe_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEBackendMut<'a, BE> {
    type Target = GGLWE<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> DerefMut for GGLWEBackendMut<'_, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWEBackendRef<'a, BE>, ['a, BE: Backend + 'a]; inner);
impl_gglwe_infos_for_inner!(GGLWEBackendMut<'a, BE>, ['a, BE: Backend + 'a]; inner);

impl<BE: Backend> GGLWEToBackendRef<BE> for GGLWEBackendRef<'_, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGLWEBackendRef<'_, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGLWEBackendRef::at_view(self, row, col)
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for GGLWEBackendMut<'_, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGLWEToBackendMut<BE> for GGLWEBackendMut<'_, BE> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.inner.noise = None;
        GGLWEBackendMut::from_inner(GGLWE {
            noise: None,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            k_digit: self.inner.k_digit,
            k_offset: self.inner.k_offset,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GGLWE<D, W> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::offset_key_k(self.k_offset, self.dnum(), self.k_digit, self.k_aux)
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GGLWE<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GGLWE<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }

    fn k_offset(&self) -> TorusPrecision {
        self.k_offset
    }
}

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Returns a shared reference to the underlying [`MatZnx`].
    pub fn data(&self) -> &MatZnx<D, W> {
        &self.data
    }
}

/// Backend-native shared view of one GLWE row.
pub trait GGLWEAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGLWEAtBackendRef<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k: self.k(),
            canonical: true,
            plaintext_meta: None,
            data,
        }
    }
}

pub(crate) fn gglwe_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufRef<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&gglwe.data, row, col);
    GLWE {
        noise: crate::layouts::LWEInfos::noise(&gglwe),
        base2k: gglwe.base2k,
        k: gglwe.k(),
        canonical: true,
        plaintext_meta: None,
        data,
    }
}

pub trait GGLWEAtViewRef<BE: Backend> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewRef<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(<GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendRef<BE>>::at_backend(
            self, row, col,
        ))
    }
}

pub(crate) fn gglwe_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&gglwe.data, row, col);
    GLWE {
        noise: crate::layouts::LWEInfos::noise(&gglwe),
        base2k: gglwe.base2k,
        k: gglwe.k(),
        canonical: true,
        plaintext_meta: None,
        data,
    }
}

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Returns a mutable reference to the underlying [`MatZnx`].
    pub fn data_mut(&mut self) -> &mut MatZnx<D, W> {
        self.noise = None;
        &mut self.data
    }
}

/// Backend-native mutable view of one GLWE row.
pub trait GGLWEAtBackendMut<BE: Backend> {
    /// Clears the owner's component noise metadata.
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGLWEAtBackendMut<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    /// Clears the owner's component noise metadata.
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        let base2k = self.base2k;
        let k = self.k();
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        self.noise = None;
        GLWE {
            noise: None,
            base2k,
            k,
            canonical: true,
            plaintext_meta: None,
            data,
        }
    }
}

pub(crate) fn gglwe_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a mut GGLWE<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>, BE::ZnxWord> {
    let base2k = gglwe.base2k;
    let k = gglwe.k();
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut gglwe.data, row, col);
    gglwe.noise = None;
    GLWE {
        noise: None,
        base2k,
        k,
        canonical: true,
        plaintext_meta: None,
        data,
    }
}

pub trait GGLWEAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewMut<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(<GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendMut<BE>>::at_backend_mut(
            self, row, col,
        ))
    }
}

impl<BE: Backend> GGLWEAtViewMut<BE> for &mut GGLWE<BE::BufMut<'_>, BE::ZnxWord> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(gglwe_at_backend_mut_from_mut::<BE>(*self, row, col))
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWE: k={} base2k={} k_digit={} k_offset={}) {}",
            self.k().0,
            self.base2k().0,
            self.k_digit().0,
            self.k_offset.0,
            self.data
        )
    }
}

impl<D: HostDataRef, W: ZnxWord> GGLWE<D, W> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8], W> {
        let data = self.data.at(row, col);
        GLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k: self.k(),
            canonical: true,
            plaintext_meta: None,
            data,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> GGLWE<D, W> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8], W> {
        let base2k = self.base2k;
        let k = self.k();
        let data = self.data.at_mut(row, col);
        self.noise = None;
        GLWE {
            noise: None,
            base2k,
            k,
            canonical: true,
            plaintext_meta: None,
            data,
        }
    }
}

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGLWE<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let (n, rows, cols_in, cols_out, size) = (
            self.data.n(),
            self.data.rows(),
            self.data.cols_in(),
            self.data.cols_out(),
            self.data.size(),
        );
        GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            base2k: self.base2k,
            k_digit: self.k_digit,
            k_offset: self.k_offset,
            k_aux: self.k_aux,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GGLWE<AlignedBuf, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc_with_offset(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_offset(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        Self::alloc_with_offset(n, base2k, dnum, k_digit, TorusPrecision(0), k_aux, rank_in, rank_out)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the GGLWE shape is its eight dimensions, as in `alloc`"
    )]
    pub(crate) fn alloc_with_offset(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_offset: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        let size: usize = crate::layouts::offset_key_size(base2k, k_offset, dnum, k_digit, k_aux);

        GGLWE {
            noise: None,
            data: MatZnx::from_data(
                alloc_aligned::<u8>(MatZnx::<AlignedBuf, W>::bytes_of(
                    n.into(),
                    dnum.into(),
                    rank_in.into(),
                    (rank_out + 1).into(),
                    size,
                )),
                n.into(),
                dnum.into(),
                rank_in.into(),
                (rank_out + 1).into(),
                size,
            ),
            base2k,
            k_digit,
            k_offset,
            k_aux,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let size: usize =
            crate::layouts::offset_key_size(infos.base2k(), infos.k_offset(), infos.dnum(), infos.k_digit(), infos.k_aux());
        MatZnx::<AlignedBuf, W>::bytes_of(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            size,
        )
    }

    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, k_digit, k_aux);

        MatZnx::<AlignedBuf, W>::bytes_of(n.into(), dnum.into(), rank_in.into(), (rank_out + 1).into(), size)
    }
}

pub trait GGLWEToBackendMut<BE: Backend>: GGLWEToBackendRef<BE> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for GGLWE<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.noise = None;
        GGLWEBackendMut::from_inner(GGLWE {
            noise: None,
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            k_offset: self.k_offset(),
            k_aux: self.k_aux(),
            data: self.data.to_backend_mut(),
        })
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for &mut GGLWE<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            k_offset: self.k_offset(),
            k_aux: self.k_aux(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<BE: Backend> GGLWEToBackendMut<BE> for &mut GGLWE<BE::BufMut<'_>, BE::ZnxWord> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.noise = None;
        GGLWEBackendMut::from_inner(GGLWE {
            noise: None,
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            k_offset: self.k_offset(),
            k_aux: self.k_aux(),
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.data),
        })
    }
}

pub trait GGLWEToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for GGLWE<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            k_offset: self.k_offset(),
            k_aux: self.k_aux(),
            data: self.data.to_backend_ref(),
        })
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGLWE<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.noise = None;
        let components = self.data.cols_out();
        let noise = crate::ComponentNoise::read_optional(reader, components)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k_digit = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let k_offset = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        crate::layouts::read_mat_znx_with_shape(&mut self.data, reader, None, None, components)?;
        crate::layouts::validate_noise_components(noise.as_ref(), self.data.cols_out())?;
        self.base2k = base2k;
        self.k_digit = k_digit;
        self.k_offset = k_offset;
        self.k_aux = k_aux;
        self.noise = noise;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGLWE<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.data.cols_out())?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k_digit.0)?;
        writer.write_u32::<LittleEndian>(self.k_offset.0)?;
        writer.write_u32::<LittleEndian>(self.k_aux.0)?;
        self.data.write_to(writer)
    }
}
