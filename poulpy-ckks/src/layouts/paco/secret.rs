//! The PaCo structured secret key (paper Algorithm 1, skGen), the per-block
//! blockwise coefficient packing, and the packed bootstrapping-key vectors σ_t
//! (Algorithm 2, bskGen, pre-encryption).
//!
//! The secret is `s = Σ_{v<4h} d_v · X^{u_v·4h + v}` with shift indices
//! `u_v ∈ [0, B)` (`u_0 = 0`) and binary selectors `d_v` such that for every
//! `v ∈ [0, h)` exactly one of `d_v, d_{h+v}, d_{2h+v}, d_{3h+v}` is set
//! (`d_0 = 1`). Its Hamming weight is exactly `h`.
//!
//! For each `t ∈ [0, 4)`, [`PaCoSecretSpec::sigma_slots_reim`] builds the slot
//! vector σ_t that the homomorphic pipeline encrypts as the bootstrapping key
//! `bsk_t`: per chunk `r ∈ [0, k)`, block `v` of width `2C` carries
//! `d_{th+v} · Z^{(u_{th+v}+r)/k}` whenever `u_{th+v} + r ≡ 0 (mod k)`
//! (coefficient packing of the blind-rotation monomial in the ring
//! `ℂ[Z]/(Z^{2C} − i)`), followed by the per-block coefficient→evaluation
//! packing — the backend's coefficient-to-slot FFT — identical to the
//! treatment of the public β_t vectors in [`coeff_enc`](super::coeff_enc).

use std::fmt::Debug;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{GLWEInfos, GLWESecret, LWEInfos};
use poulpy_hal::{
    layouts::{Backend, HostDataMut, Module, ScratchArena},
    source::Source,
};

use super::{context::PaCoContext, plan::PaCoPlan};
use crate::default::dft::DftScalar;
use crate::encoding::paco::cpx::Cpx;
use crate::{
    api::{CKKSEncodingOps, PaCoScalar},
    layouts::{CKKSEncodingBuffer, ScratchArenaTakeCKKS, copy_encoding_buffer_into_reim_host, copy_host_into_encoding_buffer},
};

/// Packs one raw chunk (length `n = 2hC`, `h` blocks of `2C` complex
/// coefficients) into its slot values, per block: block coefficients `c_j`
/// (natural order) become evaluations `slot k = Σ_j c_j·x_k^j`,
/// `x_k = ζ^{5^{bitrev(k)}}`, at the `2C` roots of `Z^{2C} = i`.
///
/// The block's complex coefficients are laid out as the ring element
/// `[Re(c_j) | Im(c_j)]` (`c_j = coeffs[j] + i·coeffs[j+2C]`), run through
/// the backend's coefficient→slot FFT, and gathered through the bit-reversal.
/// The bit-reversed point *order* is irrelevant to the
/// pipeline, which only ever combines equal slot positions until the
/// homomorphic [`paco_c2s_factors`](super::lt::paco_c2s_factors) chain — its
/// exact inverse, pinned per backend by the `paco_packing` suite gate —
/// restores natural-order coefficients.
///
/// `transform` must implement the backend's `2C`-slot coefficient→slot map.
#[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
pub(crate) fn pack_chunk<F, T>(p: &PaCoPlan, transform: &mut T, chunk: &[Cpx<F>]) -> Result<Vec<Cpx<F>>>
where
    F: DftScalar + Debug,
    T: FnMut(&[F], &mut [F], &mut [F]) -> Result<()>,
{
    p.check().context("invalid PaCo plan for chunk packing")?;
    ensure!(chunk.len() == p.slots(), "PaCo chunk has the wrong slot count");
    let mut workspace = PaCoPackingWorkspace::new(p)?;
    let mut out = vec![Cpx::zero(); p.slots()];
    pack_chunk_into_with(p, transform, chunk, &mut out, &mut workspace)?;
    Ok(out)
}

/// Reusable real/imaginary buffers for per-block coefficient packing.
pub(crate) struct PaCoPackingWorkspace<F> {
    coeffs: Vec<F>,
    re: Vec<F>,
    im: Vec<F>,
}

impl<F: DftScalar> PaCoPackingWorkspace<F> {
    pub(crate) fn new(p: &PaCoPlan) -> Result<Self> {
        let two_c = 2usize.checked_mul(p.c()).context("2*C overflows usize")?;
        let coeff_count = 2usize.checked_mul(two_c).context("4*C overflows usize")?;
        Ok(Self {
            coeffs: vec![F::zero(); coeff_count],
            re: vec![F::zero(); two_c],
            im: vec![F::zero(); two_c],
        })
    }
}

/// Transform-parameterized packing core: `transform` is the `2C`-slot
/// coefficient → slot transform (a reference FFT in tests or a backend's
/// [`ckks_coeffs_to_slots_assign`](crate::oep::CKKSEncodingImpl::ckks_coeffs_to_slots_assign)
/// on the operation path). Both paths realize the identical packing map.
pub(crate) fn pack_chunk_into_with<F, T>(
    p: &PaCoPlan,
    transform: &mut T,
    chunk: &[Cpx<F>],
    out: &mut [Cpx<F>],
    workspace: &mut PaCoPackingWorkspace<F>,
) -> Result<()>
where
    F: DftScalar + Debug,
    T: FnMut(&[F], &mut [F], &mut [F]) -> Result<()>,
{
    ensure!(chunk.len() == p.slots(), "PaCo chunk has the wrong slot count");
    ensure!(out.len() == p.slots(), "PaCo packed output has the wrong slot count");
    let two_c = 2usize.checked_mul(p.c()).context("2*C overflows usize")?;
    let coeff_count = 2usize.checked_mul(two_c).context("4*C overflows usize")?;
    ensure!(
        workspace.coeffs.len() == coeff_count && workspace.re.len() == two_c && workspace.im.len() == two_c,
        "PaCo packing workspace does not match C={}",
        p.c(),
    );
    let log_two_c = two_c.trailing_zeros();
    // The `br` readout cancels the input permutation of the bit-reversed
    // Encode factorization used by the partial C2S. Under `BitRevLow` the
    // gather additionally composes the tiled low-bit reversal `P` — the
    // telescoped chain's input-side permutation, absorbed here at the
    // cleartext boundary for free (see `paco_c2s_factors`).
    let br = |x: usize| x.reverse_bits() >> (usize::BITS - log_two_c);
    let gather: Box<dyn Fn(usize) -> usize> = match p.slot_order() {
        crate::layouts::PaCoSlotOrder::Natural => Box::new(br),
        crate::layouts::PaCoSlotOrder::BitRevLow => {
            let log_p = p.log_c() - 1;
            Box::new(move |k: usize| br(crate::default::paco::ops::ext_bitrev_low(k, log_p)))
        }
    };

    for v in 0..p.h() {
        let start = v.checked_mul(two_c).context("PaCo block offset overflows usize")?;
        let end = start.checked_add(two_c).context("PaCo block end overflows usize")?;
        let block = &chunk[start..end];
        for (j, x) in block.iter().enumerate() {
            workspace.coeffs[j] = x.re;
            workspace.coeffs[j + two_c] = x.im;
        }
        transform(&workspace.coeffs, &mut workspace.re, &mut workspace.im)
            .with_context(|| format!("cannot pack PaCo block {v}"))?;
        for k in 0..two_c {
            let g = gather(k);
            out[start + k] = Cpx::new(workspace.re[g], workspace.im[g]);
        }
    }
    Ok(())
}

/// The combinatorial description of a PaCo structured secret key.
///
/// Reshaping the secret's coefficient vector as an `(N/h) × h` matrix
/// `M[j][v] = s[j·h + v]` (the columns are the `h` interleaved residue
/// classes mod `h`), every column holds exactly one nonzero coefficient
/// (a `1`), and its row index factors as `j = t + 4u` with `t ∈ [0, 4)` and
/// `u ∈ [0, B)`, where `B = N/(4h)` is the shift range [`PaCoPlan::b`]:
///
/// ```text
///                  column v:    0    1    2    …   h−1
///    row 0     (t=0, u=0)    [  1    .    .    …    .  ]  ← class 0 pinned: d_0 = 1, u_0 = 0
///    row 1     (t=1, u=0)    [  .    .    .    …    .  ]
///    row 2     (t=2, u=0)    [  .    .    .    …    .  ]
///    row 3     (t=3, u=0)    [  .    .    .    …    .  ]
///    row 4     (t=0, u=1)    [  .    .    1    …    .  ]  ← class 2: group t = 0, shift u = 1
///    row 5     (t=1, u=1)    [  .    .    .    …    .  ]
///    row 6     (t=2, u=1)    [  .    1    .    …    .  ]  ← class 1: group t = 2, shift u = 1
///    ⋮              ⋮                               ⋱
///    row N/h−2 (t=2, u=B−1)  [  .    .    .    …    1  ]  ← class h−1: group t = 2, shift u = B−1
///    row N/h−1 (t=3, u=B−1)  [  .    .    .    …    .  ]
/// ```
///
/// `t = j mod 4` selects which of the four bootstrapping keys `bsk_t` serves
/// the class, and `u = ⌊j/4⌋` is its blind-rotation shift, so each `bsk_t`
/// addresses the `B` interleaved rows `j ≡ t (mod 4)`. Since
/// `(t, u) ↦ t + 4u` is a bijection onto `[0, N/h)`, the nonzero may sit
/// anywhere in its column — the structure costs no key-space beyond
/// one-nonzero-per-class.
///
/// The fields are the two coordinates of that factorization. The selectors
/// [`d`](Self::d), read as a `4 × h` matrix `D[t][v] = d[t·h + v]` (row `t`
/// is the contiguous slice `d[t·h..(t+1)·h]`), are the matrix above summed
/// over every fourth row — `D[t][v] = Σ_{u<B} M[t + 4u][v]` — same example
/// key:
///
/// ```text
///                 class v:    0    1    2    …   h−1
///    group t = 0  d[0..h)  [  1    .    1    …    .  ]  ← d[0] = 1 pinned
///    group t = 1  d[h..2h) [  .    .    .    …    .  ]
///    group t = 2  d[2h..3h)[  .    1    .    …    1  ]
///    group t = 3  d[3h..4h)[  .    .    .    …    .  ]
/// ```
///
/// Each column still carries exactly one `1` (the paper's "exactly one of
/// `d_v, d_{h+v}, d_{2h+v}, d_{3h+v}`"), now answering just "which group `t`
/// — hence which `bsk_t` — serves class `v`"; rows are unconstrained, so a
/// group may serve several classes or none. The shift [`u`](Self::u)`[λ]`
/// (`λ = t·h + v`) is the coordinate the sum forgot: it puts the `1` back at
/// row `t + 4·u[λ]`, i.e. secret coefficient `s[λ + 4h·u[λ]] = 1`
/// (the representation produced internally by `sk_coeffs`). Entries of `u` at
/// unselected indices (`d[λ] == false`) are ignored.
#[derive(Clone, PartialEq, Eq)]
pub struct PaCoSecretSpec {
    /// Shift indices `u_v ∈ [0, B)` for `v ∈ [0, 4h)`; `u[0] == 0`.
    u: Vec<usize>,
    /// Selectors `d_v`; exactly one set per residue class `v mod h`; `d[0] == true`.
    d: Vec<bool>,
}

/// Redacting [`Debug`]: `u` and `d` completely determine the structured
/// bootstrapping secret, so only the vector lengths are printed. Never derive
/// `Debug` on this type. Note the derived [`PartialEq`] is **not**
/// constant-time; do not use it to compare secrets across a trust boundary.
impl core::fmt::Debug for PaCoSecretSpec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PaCoSecretSpec")
            .field("u", &format_args!("<redacted; len {}>", self.u.len()))
            .field("d", &format_args!("<redacted; len {}>", self.d.len()))
            .finish()
    }
}

impl PaCoSecretSpec {
    /// Builds and validates a structured secret specification.
    pub fn new(p: &PaCoPlan, u: Vec<usize>, d: Vec<bool>) -> Result<Self> {
        let spec = Self { u, d };
        spec.check(p)?;
        Ok(spec)
    }

    /// Shift indices `u_v`, in the canonical `4h`-entry layout.
    pub fn u(&self) -> &[usize] {
        &self.u
    }

    /// Binary selectors `d_v`, in the canonical `4h`-entry layout.
    pub fn d(&self) -> &[bool] {
        &self.d
    }

    fn check(&self, p: &PaCoPlan) -> Result<()> {
        p.check().context("invalid PaCo plan for structured secret")?;
        let h4 = 4usize.checked_mul(p.h()).context("4*h overflows usize")?;
        ensure!(
            self.u.len() == h4,
            "PaCo secret shift vector has length {}, expected 4*h = {h4}",
            self.u.len(),
        );
        ensure!(
            self.d.len() == h4,
            "PaCo secret selector vector has length {}, expected 4*h = {h4}",
            self.d.len(),
        );
        ensure!(self.u[0] == 0, "PaCo secret requires u[0] = 0, got {}", self.u[0]);
        ensure!(self.d[0], "PaCo secret requires d[0] = true");

        let b = p.b();
        for (lambda, &shift) in self.u.iter().enumerate() {
            ensure!(shift < b, "PaCo secret shift u[{lambda}] = {shift} is outside [0, B = {b})",);
            let position = shift
                .checked_mul(h4)
                .and_then(|offset| offset.checked_add(lambda))
                .with_context(|| format!("PaCo secret coefficient position for index {lambda} overflows usize"))?;
            ensure!(
                position < p.n(),
                "PaCo secret coefficient position {position} for index {lambda} exceeds degree N = {}",
                p.n(),
            );
        }
        for v in 0..p.h() {
            let mut picks = 0usize;
            for t in 0..4usize {
                let lambda = t
                    .checked_mul(p.h())
                    .and_then(|offset| offset.checked_add(v))
                    .context("PaCo selector index overflows usize")?;
                picks = picks
                    .checked_add(usize::from(self.d[lambda]))
                    .context("PaCo selector count overflows usize")?;
            }
            ensure!(
                picks == 1,
                "PaCo secret residue class {v} has {picks} selectors set, expected exactly one"
            );
        }
        Ok(())
    }

    /// Samples a fresh key description (Algorithm 1).
    pub fn sample(p: &PaCoPlan, source: &mut Source) -> Result<Self> {
        p.check().context("invalid PaCo plan for structured-secret sampling")?;
        let h4 = 4usize.checked_mul(p.h()).context("4*h overflows usize")?;
        let b = u64::try_from(p.b()).context("B does not fit u64")?;
        ensure!(b > 0, "PaCo shift range B must be non-zero");
        let mut u = vec![0usize; h4];
        for uv in u.iter_mut().skip(1) {
            *uv = usize::try_from(source.next_u64n(b, b - 1)).context("sampled PaCo shift does not fit usize")?;
        }
        let mut d = vec![false; h4];
        d[0] = true;
        for v in 1..p.h() {
            let t = usize::try_from(source.next_u64n(4, 3)).context("sampled PaCo group does not fit usize")?;
            let lambda = t
                .checked_mul(p.h())
                .and_then(|offset| offset.checked_add(v))
                .context("PaCo selector index overflows usize")?;
            d[lambda] = true;
        }
        Self::new(p, u, d)
    }

    /// The secret polynomial's coefficient vector: `s[v + u_v·4h] = d_v`.
    pub(crate) fn sk_coeffs(&self, p: &PaCoPlan) -> Result<Vec<i64>> {
        self.check(p)?;
        let h4 = 4usize.checked_mul(p.h()).context("4*h overflows usize")?;
        let mut s = vec![0i64; p.n()];
        for lambda in 0..h4 {
            if self.d[lambda] {
                let position = self.u[lambda]
                    .checked_mul(h4)
                    .and_then(|offset| offset.checked_add(lambda))
                    .with_context(|| format!("PaCo secret coefficient position for index {lambda} overflows usize"))?;
                s[position] = 1;
            }
        }
        Ok(s)
    }

    /// One pre-packing chunk of σ_t (Algorithm 2, before [`Self::sigma_slots_reim`]
    /// packs it).
    ///
    /// - **Input**: the group `t ∈ [0, 4)` and the chunk index `r ∈ [0, k)`,
    ///   where `k = B/C` ([`PaCoPlan::k`]) is the number of chunks: a
    ///   `2C`-block encodes only `C` of the `B` shifts a class can have, so a
    ///   shift is split radix-`C` into (chunk, exponent) and `k` chunks cover
    ///   the range. From `self` it reads the coordinates `u[λ]`, `d[λ]` of
    ///   the group's `h` candidates `λ = t·h + v`; from `p` the shape `h`,
    ///   `C`, `k`.
    /// - **Rule**: block `v` gets a single `1` at block-slot `(u[λ] + r)/k`
    ///   if `d[λ]` is set and `u[λ] + r ≡ 0 (mod k)`; otherwise it stays
    ///   zero. Each served class thus appears in exactly one of the `k`
    ///   chunks, `r = −u[λ] mod k`, as the monomial `Z^{⌈u[λ]/k⌉}`.
    /// - **Output**: `n = 2hC` complex slots, read as `h` blocks of `2C`
    ///   (block `v` = slots `[v·2C, (v+1)·2C)`); every entry is `0` or `1`.
    ///
    /// Walking the example key of the [`PaCoSecretSpec`] docs, instantiated
    /// with `h = 4`, `C = 4`, `k = 4` (`N = 256`, `B = 16`). Step 1 — pick
    /// the group: calling with `t = 0` selects row 0 of the selector matrix,
    /// which serves class 0 (`u[0] = 0`) and class 2 (`u[2] = 1`):
    ///
    /// ```text
    ///               class v:    0    1    2    3
    ///    t = 0  →  d[0..4)   [  1    .    1    .  ]   ← this call's row
    ///    t = 1     d[4..8)   [  .    .    .    .  ]
    ///    t = 2     d[8..12)  [  .    1    .    1  ]     (other groups: other
    ///    t = 3     d[12..16) [  .    .    .    .  ]      bsk_t, not this call)
    /// ```
    ///
    /// Step 2 — split each served shift `u` into (chunk, exponent):
    ///
    /// ```text
    ///    (v = 0, u = 0)  →  chunk r = −0 mod 4 = 0,  block 0,  slot (0+0)/4 = 0   (Z^0)
    ///    (v = 2, u = 1)  →  chunk r = −1 mod 4 = 3,  block 2,  slot (1+3)/4 = 1   (Z^1)
    /// ```
    ///
    /// Step 3 — so of group 0's four chunks (one call each), only `r = 0` and
    /// `r = 3` are non-empty:
    ///
    /// ```text
    ///        block v=0    block v=1    block v=2    block v=3      (2C = 8 slots each)
    /// r=0  [ 1·······  |  ········  |  ········  |  ········ ]  ← class 0's Z^0
    /// r=1  [ ········  |  ········  |  ········  |  ········ ]
    /// r=2  [ ········  |  ········  |  ········  |  ········ ]
    /// r=3  [ ········  |  ········  |  ·1······  |  ········ ]  ← class 2's Z^1
    /// ```
    ///
    /// The `2C` block width (monomial degree ≤ C, data degree < C) is
    /// headroom so products never wrap modulo `Z^{2C} − i`.
    #[cfg(test)]
    pub(crate) fn sigma_raw_chunk<F: DftScalar>(&self, p: &PaCoPlan, t: usize, r: usize) -> Result<Vec<Cpx<F>>> {
        self.check(p)?;
        ensure!(t < 4, "PaCo sigma group t must be in [0, 4), got {t}");
        ensure!(r < p.k(), "PaCo sigma chunk r must be in [0, k = {}), got {r}", p.k());
        let mut z = vec![Cpx::zero(); p.slots()];
        self.sigma_raw_chunk_into_validated(p, t, r, &mut z)?;
        Ok(z)
    }

    fn sigma_raw_chunk_into_validated<F: DftScalar>(&self, p: &PaCoPlan, t: usize, r: usize, z: &mut [Cpx<F>]) -> Result<()> {
        let (h, c, k) = (p.h(), p.c(), p.k());
        let two_c = 2usize.checked_mul(c).context("2*C overflows usize")?;
        let group_start = t.checked_mul(h).context("PaCo sigma group offset overflows usize")?;
        z.fill(Cpx::zero());
        for v in 0..h {
            let lambda = group_start
                .checked_add(v)
                .context("PaCo sigma selector index overflows usize")?;
            let i = self.u[lambda]
                .checked_add(r)
                .context("PaCo sigma shift plus chunk overflows usize")?;
            if i.is_multiple_of(k) && self.d[lambda] {
                let position = v
                    .checked_mul(two_c)
                    .and_then(|offset| offset.checked_add(i / k))
                    .context("PaCo sigma slot index overflows usize")?;
                ensure!(
                    position < z.len(),
                    "PaCo sigma slot index {position} exceeds chunk length {}",
                    z.len()
                );
                z[position] = Cpx::one();
            }
        }
        Ok(())
    }

    /// Writes the structured secret into a host-backed rank-1 [`GLWESecret`]
    /// (the coefficients of the internal `sk_coeffs` representation, tagged
    /// `BinaryFixed(h)` — which is literally true: the key is binary of Hamming
    /// weight `h`). The secret can then be uploaded/prepared like any other
    /// GLWE secret.
    pub fn fill_glwe_secret<D: HostDataMut>(&self, p: &PaCoPlan, sk: &mut GLWESecret<D>) -> Result<()> {
        self.check(p)?;
        ensure!(
            sk.n().as_usize() == p.n(),
            "GLWE secret degree is {}, expected PaCo degree {}",
            sk.n().as_usize(),
            p.n(),
        );
        ensure!(
            sk.rank().as_usize() == 1,
            "PaCo expects a rank-1 GLWE secret, got rank {}",
            sk.rank().as_usize(),
        );
        let coeffs = self.sk_coeffs(p)?;
        sk.fill_binary_coeffs(0, &coeffs);
        Ok(())
    }

    /// The packed slot vector σ_t (Algorithm 2, the value encrypted as
    /// `bsk_t`): each chunk is passed through the per-block coefficient→
    /// evaluation packing (every `2C`-block becomes its evaluations at the
    /// roots of `Z^{2C} = i`) and the `k` chunks are concatenated to the full
    /// `N/2` slots.
    ///
    /// ```text
    /// raw chunk z⁽⁰⁾ ──2C-point FFT per block──► packed₀ ┐
    /// raw chunk z⁽¹⁾ ──2C-point FFT per block──► packed₁ ├─ concat ─► σ_t
    ///        ⋮                                           │   (k·n = N/2 slots)
    /// raw chunk z⁽ᵏ⁻¹⁾ ─────────────────────────► ………  ──┘
    /// ```
    ///
    /// One `N/2`-slot vector therefore encodes, per class, any of the
    /// `k·C = B = N/(4h)` shifts reachable by group `t` — which is why four
    /// keys `bsk_0..bsk_3` are needed to cover the full `[0, N/h)` row range
    /// of the secret matrix (see the type-level docs).
    pub(crate) fn sigma_slots_with<F, T>(&self, p: &PaCoPlan, t: usize, transform: &mut T) -> Result<Vec<Cpx<F>>>
    where
        F: DftScalar + Debug,
        T: FnMut(&[F], &mut [F], &mut [F]) -> Result<()>,
    {
        self.check(p)?;
        ensure!(t < 4, "PaCo sigma group t must be in [0, 4), got {t}");
        let mut packed = Vec::with_capacity(p.half_n());
        let mut raw = vec![Cpx::zero(); p.slots()];
        let mut workspace = PaCoPackingWorkspace::new(p)?;
        for r in 0..p.k() {
            self.sigma_raw_chunk_into_validated(p, t, r, &mut raw)?;
            let start = packed.len();
            packed.resize(start + p.slots(), Cpx::zero());
            pack_chunk_into_with(p, transform, &raw, &mut packed[start..], &mut workspace)?;
        }
        ensure!(
            packed.len() == p.half_n(),
            "packed PaCo sigma has length {}, expected {}",
            packed.len(),
            p.half_n(),
        );
        Ok(packed)
    }

    /// Returns the packed bootstrapping-key plaintext `sigma_t` as separate
    /// real and imaginary slot vectors in the crate's conventional encoding
    /// representation.
    ///
    /// `t` must be in `0..4`. Taking the compiled context ensures the plan and
    /// scalar precision are exactly the ones used by evaluation. The `2*C`
    /// packing transform dispatches through `module`'s CKKS encoding plans and
    /// uses the caller's standard arena. The returned vectors each have `N/2`
    /// entries and can be passed to
    /// [`CKKSEncodingHostOps::ckks_encode_reim_into`](crate::api::CKKSEncodingHostOps::ckks_encode_reim_into).
    pub fn sigma_slots_reim<BE, F>(
        &self,
        module: &Module<BE>,
        context: &PaCoContext<BE, F>,
        t: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<(Vec<F>, Vec<F>)>
    where
        BE: Backend,
        Module<BE>: CKKSEncodingOps<BE, F>,
        F: PaCoScalar,
    {
        let plan = context.plan();
        self.check(plan)?;
        ensure!(t < 4, "PaCo sigma group t must be in [0, 4), got {t}");
        let packed = self.sigma_slots_with(plan, t, &mut |coeffs, re, im| {
            let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(coeffs.len());
            ensure!(
                scratch.available() >= required,
                "PaCo sigma packing needs {required} scratch bytes, but only {} are available",
                scratch.available()
            );
            scratch.scope(|arena| {
                let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(coeffs.len());
                copy_host_into_encoding_buffer::<BE, F, _>(&mut values, coeffs)?;
                module.ckks_coeffs_to_slots_assign(&mut values)?;
                copy_encoding_buffer_into_reim_host::<BE, F, _>(&values, re, im)
            })
        })?;
        let mut re = Vec::with_capacity(packed.len());
        let mut im = Vec::with_capacity(packed.len());
        for value in packed {
            re.push(value.re);
            im.push(value.im);
        }
        Ok((re, im))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Source {
        Source::new([42u8; 32])
    }

    #[test]
    fn sampler_invariants() {
        let p = PaCoPlan::new(9, 8, 4, 29).unwrap();
        let mut src = source();
        for _ in 0..50 {
            let key = PaCoSecretSpec::sample(&p, &mut src).unwrap();
            assert_eq!(key.u().len(), 4 * p.h());
            assert_eq!(key.d().len(), 4 * p.h());
            assert_eq!(key.u()[0], 0, "u_0 = 0");
            assert!(key.d()[0], "d_0 = 1");
            assert!(key.u().iter().all(|&u| u < p.b()), "u_v < B");
            for v in 0..p.h() {
                let picks = (0..4).filter(|&t| key.d()[t * p.h() + v]).count();
                assert_eq!(picks, 1, "exactly one selector in class v={v}");
            }
        }
    }

    #[test]
    fn debug_redacts_secret_material() {
        let p = PaCoPlan::new(9, 8, 4, 29).unwrap();
        let mut src = source();
        let key = PaCoSecretSpec::sample(&p, &mut src).unwrap();
        let printed = format!("{key:?}");
        assert!(printed.contains("redacted"), "Debug must redact: {printed}");
        // Neither coordinate vector may appear: no list syntax, no selector values.
        assert!(!printed.contains('['), "Debug leaks vector contents: {printed}");
        assert!(
            !printed.contains("true") && !printed.contains("false"),
            "Debug leaks selectors: {printed}"
        );
    }

    #[test]
    fn secret_has_weight_h_with_one_nonzero_per_block() {
        let p = PaCoPlan::new(9, 8, 4, 29).unwrap();
        let mut src = source();
        let key = PaCoSecretSpec::sample(&p, &mut src).unwrap();
        let s = key.sk_coeffs(&p).unwrap();
        assert_eq!(s.iter().filter(|&&x| x != 0).count(), p.h(), "Hamming weight h");
        // Nonzero positions are λ + u_λ·4h with one selected λ per residue
        // class mod h, so each of the h interleaved classes holds exactly one
        // nonzero (the paper's structural property).
        for class in 0..p.h() {
            let w = s.iter().skip(class).step_by(p.h()).filter(|&&x| x != 0).count();
            assert_eq!(w, 1, "residue class {class} mod h");
        }
    }

    #[test]
    fn sigma_chunks_select_each_v_exactly_once() {
        let p = PaCoPlan::new(9, 8, 4, 29).unwrap();
        let mut src = source();
        let key = PaCoSecretSpec::sample(&p, &mut src).unwrap();
        // Over all (t, r), each selected v contributes exactly one monomial with
        // exponent (u_v + r)/k = ceil(u_v / k).
        for t in 0..4 {
            for v in 0..p.h() {
                let selected = key.d()[t * p.h() + v];
                let mut hits = 0;
                for r in 0..p.k() {
                    let z = key.sigma_raw_chunk::<f64>(&p, t, r).unwrap();
                    let block = &z[v * 2 * p.c()..(v + 1) * 2 * p.c()];
                    let nz: Vec<usize> = block
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| x.abs() > 0.0)
                        .map(|(i, _)| i)
                        .collect();
                    if !nz.is_empty() {
                        hits += 1;
                        assert_eq!(nz, vec![key.u()[t * p.h() + v].div_ceil(p.k())], "exponent is ceil(u/k)");
                    }
                }
                assert_eq!(hits, usize::from(selected), "t={t} v={v}");
            }
        }
    }
}
