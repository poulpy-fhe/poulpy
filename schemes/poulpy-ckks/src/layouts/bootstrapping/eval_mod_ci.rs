//! EvalMod on the conjugate-invariant ring: the ring of a plan, and the plaintexts of the conversions there and back.

use anyhow::Result;
use poulpy_core::{
    GLWEDiagonalProd, LinearTransformationPrepared,
    layouts::{Base2K, Diagonals, GLWEToBackendRef, LinearTransformationPlan},
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSPlaintext, CKKSPlaintextAlloc, CKKSPlaintextOwned, ComplexDiagonals, DFTMatrix, Encode, EvalMod, EvalModPlan, Split,
        eval_mod::ckks_eval_mod_compile,
    },
};

/// The ring EvalMod runs on, after an S2C-first CoeffsToSlots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvalModRing {
    /// The ring of the bootstrap, once per half of the slots.
    #[default]
    Standard,
    /// The conjugate-invariant ring of the same degree, once for all `N` coefficients, reached through the standard
    /// ring of degree `2N`: by `u` before EvalMod, `pre`, and `v` after it, `post`. The bootstrap's module, of degree
    /// `2N`, serves both rings: the conversions at degree `2N` and EvalMod at degree `N`.
    ConjugateInvariant { pre: CIConversion, post: CIConversion },
}

/// How a conversion of [`EvalModRing::ConjugateInvariant`] reaches the ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CIConversion {
    /// A product by its plaintext, at scale `2^log_delta`, which consumes `log_delta` bits.
    Product { log_delta: usize },
    /// Folded into the adjacent DFT factor, which then runs twice, on the even and
    /// the odd part of the plaintext: no product and no budget. Only `u`, into
    /// the last CoeffsToSlots factor.
    Folded,
}

impl CIConversion {
    /// Budget the conversion consumes.
    pub fn consumed_bits(&self) -> usize {
        match *self {
            Self::Product { log_delta } => log_delta,
            Self::Folded => 0,
        }
    }
}

impl EvalModRing {
    /// Budget the products of the conversions consume.
    pub fn consumed_bits(&self) -> usize {
        match *self {
            Self::Standard => 0,
            Self::ConjugateInvariant { pre, post } => pre.consumed_bits() + post.consumed_bits(),
        }
    }

    pub(super) fn pre_consumed_bits(&self) -> usize {
        match *self {
            Self::Standard => 0,
            Self::ConjugateInvariant { pre, .. } => pre.consumed_bits(),
        }
    }
}

/// EvalMod on the conjugate-invariant ring of degree `N` and the plaintexts of
/// degree `2N` of the conversions there and back
/// ([`EvalModRing::ConjugateInvariant`]).
pub struct EvalModCIPrepared<BE: Backend, F> {
    /// EvalMod in plaintexts of the CI ring.
    pub(crate) eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>>,
    /// `u`, `1` on slot `j` and `−i` on slot `j + N/2`: the trace of `u·z(X²)`
    /// holds the real parts of `z_j` and `−i·z_j`, `c_j` and `c_{j+N/2}`.
    pub(crate) pre: CIPrePrepared<BE>,
    /// `v`, `2` on slot `j` and `2i` on slot `j + N/2`, which pairs them back and restores the
    /// factor 2 of the conjugate pair.
    pub(crate) post: CKKSPlaintextOwned<BE>,
}

impl<BE: Backend, F> EvalModCIPrepared<BE, F> {
    /// EvalMod in plaintexts of the CI ring.
    pub fn eval_mod(&self) -> &EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>> {
        &self.eval_mod
    }

    /// `u`, the conversion before EvalMod.
    pub fn pre(&self) -> &CIPrePrepared<BE> {
        &self.pre
    }

    /// `v`, the conversion after EvalMod.
    pub fn post(&self) -> &CKKSPlaintextOwned<BE> {
        &self.post
    }
}

/// `u` of [`EvalModCIPrepared`], as its [`CIConversion`] applies it.
pub struct CIPrePrepared<BE: Backend>(pub(crate) CIPreForm<BE>);

pub(crate) enum CIPreForm<BE: Backend> {
    Product(CKKSPlaintextOwned<BE>),
    Folded {
        even: LinearTransformationPrepared<BE>,
        odd: LinearTransformationPrepared<BE>,
        /// The schedules of `even` and `odd`.
        plans: Box<[LinearTransformationPlan; 2]>,
    },
}

impl<BE: Backend> CIPrePrepared<BE> {
    /// `u` as a plaintext of degree `2N`, for [`CIConversion::Product`].
    pub fn product(&self) -> Option<&CKKSPlaintextOwned<BE>> {
        match &self.0 {
            CIPreForm::Product(u) => Some(u),
            CIPreForm::Folded { .. } => None,
        }
    }

    /// For [`CIConversion::Folded`], the last CoeffsToSlots factor `M` times the
    /// slots of the even and odd parts of `u = u_e(X²) + X·u_o(X²)`, in that order:
    /// with `a = u_e·M(w)` and `b = u_o·M(w)` of degree `N`, `u·z(X²) = a(X²) + X·b(X²)`.
    /// The CoeffsToSlots matrix of the context then holds the other factors.
    pub fn folded(&self) -> Option<[&LinearTransformationPrepared<BE>; 2]> {
        match &self.0 {
            CIPreForm::Product(_) => None,
            CIPreForm::Folded { even, odd, .. } => Some([even, odd]),
        }
    }
}

/// [`EvalModCIPrepared`] before its folded factors are prepared.
pub(super) struct EvalModCI<BE: Backend, F> {
    eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>>,
    pub(super) pre: CIPre<BE>,
    post: CKKSPlaintextOwned<BE>,
}

/// [`CIPrePrepared`] before its folded factors are prepared.
pub(super) enum CIPre<BE: Backend> {
    Product(CKKSPlaintextOwned<BE>),
    /// The even and odd factors, in a matrix of the CoeffsToSlots plan.
    Folded(DFTMatrix<BE, Encode, Split>),
}

impl<BE: Backend, F> EvalModCI<BE, F> {
    /// Prepares the factors `u` folds into, the only part not resident yet.
    pub(super) fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> Result<EvalModCIPrepared<BE, F>>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixPreparedFactory<BE>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        Ok(EvalModCIPrepared {
            eval_mod: self.eval_mod,
            pre: match self.pre {
                CIPre::Product(pre) => CIPrePrepared(CIPreForm::Product(pre)),
                CIPre::Folded(folded) => {
                    let [even, odd]: [_; 2] = module
                        .ckks_dft_matrix_prepare_alloc(&folded, scratch)?
                        .inner
                        .factors
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("an even and an odd factor"));
                    let plans = Box::new([even.index(), odd.index()]);
                    CIPrePrepared(CIPreForm::Folded { even, odd, plans })
                }
            },
            post: self.post,
        })
    }
}

impl<BE: Backend, F: CKKSEncodingScalar> EvalModCI<BE, F> {
    /// Encodes EvalMod on the conjugate-invariant ring at degree `N`, half the degree of `module`, and `pre` and `post`
    /// at degree `2N`. Folding `pre` takes the last factor
    /// of `coeffs_to_slots`.
    pub(super) fn compile(
        module: &Module<BE>,
        base2k: Base2K,
        eval_mod: EvalModPlan,
        [pre, post]: [CIConversion; 2],
        coeffs_to_slots: &mut DFTMatrix<BE, Encode, Split>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
        Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    {
        // Slot `j` of degree `N` is slot `j` and slot `j + N/2` of degree `2N`.
        let half = module.n() / 4;
        let mut pair = |modulus: f64, second: f64, log_delta: usize| {
            let (mut re, mut im) = (vec![F::zero(); 2 * half], vec![F::zero(); 2 * half]);
            re[..half].fill(F::from_f64(modulus).unwrap());
            im[half..].fill(F::from_f64(modulus * second).unwrap());
            // Values of modulus one take two bits above the scale, of modulus two three.
            let headroom = 2 + usize::from(modulus > 1.0);
            let mut pt = module.ckks_plaintext_alloc_from_slots(2 * half, base2k, (log_delta + headroom).into());
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
                slot_layout: crate::SlotLayout::Full,
            });
            module.ckks_encode_slots_host_into(&mut pt, &re, &im, scratch).map(|_| pt)
        };
        // `v`, a product since the plan refuses to fold it, also carries the factor 2
        // of the conjugate pair, which the even coefficients of degree `2N` keep once
        // back at degree `N`.
        let post = pair(2.0, 1.0, post.consumed_bits())?;
        let pre = match pre {
            CIConversion::Product { log_delta } => CIPre::Product(pair(1.0, -1.0, log_delta)?),
            CIConversion::Folded => {
                // `u` is `1` at the roots `ω_j = ζ_{4N}^{5^j}` of slots `j` and `−i` at
                // `−ω_j`, slots `j + N/2`: `u_e = (1 − i)/2` and `u_o = (1 + i)/(2ω_j)`
                // on slot `j` of degree `N`, at the root `ω_j²`.
                let order = ConjugateInvariant::CYCLOTOMIC_ORDER_FACTOR as u64 * (module.n() / 2) as u64;
                let one_half = F::from_f64(0.5).unwrap();
                let slot_scales = |re: Vec<F>, im: Vec<F>| {
                    let mut scales = ComplexDiagonals::new(Diagonals::new(half), Diagonals::new(half));
                    scales.re.set(0, re);
                    scales.im.set(0, im);
                    scales
                };
                let (mut odd_re, mut odd_im) = (Vec::with_capacity(half), Vec::with_capacity(half));
                let mut pow5 = 1u64;
                for _ in 0..half {
                    // (1 + i)/2 times the conjugate root `1/ω_j`.
                    let (cos, sin) = F::ckks_root_of_unity(order - pow5, order.trailing_zeros());
                    odd_re.push((cos - sin) * one_half);
                    odd_im.push((cos + sin) * one_half);
                    pow5 = pow5 * 5 % order;
                }
                let scales = [slot_scales(vec![one_half], vec![-one_half]), slot_scales(odd_re, odd_im)];
                let dft = &mut coeffs_to_slots.inner;
                dft.factors.pop();
                CIPre::Folded(crate::layouts::dft::ckks_scaled_last_factors::<BE, F>(
                    module,
                    base2k,
                    &dft.plan,
                    module.log_n() - 1,
                    &scales,
                    scratch,
                )?)
            }
        };
        Ok(Self {
            eval_mod: ckks_eval_mod_compile::<BE, F, ConjugateInvariant>(base2k, eval_mod, module, scratch)?,
            pre,
            post,
        })
    }
}
