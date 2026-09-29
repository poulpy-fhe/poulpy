use poulpy_core::{
    GLWEAdd, GLWENormalize,
    layouts::{GLWEInfos, GLWESecretToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxCopy, VecZnxFillUniformSource,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRshAssign, VecZnxRshTmpBytes, VecZnxSubAssign, VecZnxZero,
    },
    layouts::{
        Backend, HostDataMut, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    source::Source,
};

use crate::{
    galois_ring::{GaloisRing, modulus_low},
    layouts::{GLWEShamirLayout, GLWEShamirPolynomialOwned, GLWEShamirShareOwned, GLWEWideSecretOwned},
};

pub trait GLWEShamirMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_shamir_polynomial_gen_reference<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos;

    fn mhe_glwe_shamir_share_gen_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_shamir_share_gen_reference(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn mhe_glwe_shamir_share_aggregate_reference(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>);

    fn mhe_glwe_shamir_share_finalize_tmp_bytes_reference(&self) -> usize;

    fn mhe_glwe_shamir_share_finalize_reference(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend<ZnxWord = i64>> GLWEShamirMHEProtocolReference<BE> for Module<BE>
where
    BE::OwnedBuf: HostDataMut + Clone,
    Self: VecZnxZero<BE>
        + VecZnxCopy<BE>
        + VecZnxAddAssign<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxRshAssign<BE>
        + VecZnxRshTmpBytes
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
{
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize {
        self.vec_znx_rsh_tmp_bytes(layout.k.as_usize().div_ceil(layout.base2k.as_usize()))
    }

    fn mhe_glwe_shamir_polynomial_gen_reference<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        assert!(
            sk.rank() == res.rank && sk.n() == res.inner.n(),
            "invalid polynomial: secret layout differs from the sharing's"
        );
        let (rank, d, t) = (res.rank.as_usize(), res.gr_degree, res.threshold);
        let (base2k, k) = (res.inner.base2k().as_usize(), res.inner.k().as_usize());
        let sk = sk.to_backend_ref();
        {
            let mut inner = GLWEToBackendMut::<BE>::to_backend_mut(&mut res.inner);
            let data = inner.data_mut();
            for r in 0..rank {
                for j in 0..d {
                    self.vec_znx_zero(data, r * d + j);
                }
                // The secret enters at weight 2^-base2k; the exact shift moves it to 2^-k.
                self.vec_znx_add_scalar_assign(data, r * d, 0, sk.data(), r);
                self.vec_znx_rsh_assign(base2k, k - base2k, data, r * d, scratch);
            }
            for col in rank * d..t * rank * d {
                self.vec_znx_fill_uniform_source(base2k, k, data, col, source_xm);
            }
        }
        res.inner.set_canonical(true);
    }

    fn mhe_glwe_shamir_share_gen_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize {
        let cols = layout.rank.as_usize() * layout.gr_degree;
        let size = layout.k.as_usize().div_ceil(layout.base2k.as_usize());
        4 * BE::scratch_aligned(BE::bytes_of_vec_znx(layout.n.as_usize(), cols, size)) + self.vec_znx_normalize_tmp_bytes()
    }

    fn mhe_glwe_shamir_share_gen_reference(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let layout = poly.layout();
        assert!(
            GLWEShamirLayout {
                threshold: layout.threshold,
                ..res.layout()
            } == layout,
            "invalid share: layouts differ"
        );
        let (rank, d, t) = (layout.rank.as_usize(), layout.gr_degree, layout.threshold);
        assert!(
            recipient >= 1 && (recipient as usize) < (1 << d),
            "invalid share: party point out of range"
        );
        let (n, base2k, k) = (layout.n.as_usize(), layout.base2k.as_usize(), layout.k.as_usize());
        let (cols, size, low) = (rank * d, k.div_ceil(base2k), modulus_low(d));
        let poly_ref = GLWEToBackendRef::<BE>::to_backend_ref(&poly.inner);
        let coeffs = poly_ref.data();

        let (mut acc, scratch_1) = scratch.borrow().take_vec_znx_scratch(n, cols, size);
        let (mut out, scratch_2) = scratch_1.take_vec_znx_scratch(n, cols, size);
        let (mut pow, scratch_3) = scratch_2.take_vec_znx_scratch(n, cols, size);
        let (mut nxt, mut scratch_4) = scratch_3.take_vec_znx_scratch(n, cols, size);
        for c in 0..cols {
            self.vec_znx_copy(&mut acc, c, coeffs, (t - 1) * cols + c);
        }
        // Horner: acc <- x_recipient * acc + c_m, with x_recipient = Sum_b bit_b y^b.
        for m in (0..t - 1).rev() {
            for c in 0..cols {
                self.vec_znx_zero(&mut out, c);
                self.vec_znx_copy(&mut pow, c, &acc.to_backend_ref(), c);
            }
            for b in 0..d {
                if recipient >> b & 1 == 1 {
                    for c in 0..cols {
                        self.vec_znx_add_assign(&mut out, c, &pow.to_backend_ref(), c);
                    }
                }
                if recipient >> (b + 1) != 0 {
                    mul_y(self, &mut nxt, &pow.to_backend_ref(), rank, d, low);
                    for c in 0..cols {
                        self.vec_znx_normalize_assign(base2k, k, 0, &mut nxt, c, &mut scratch_4);
                    }
                    std::mem::swap(&mut pow, &mut nxt);
                }
            }
            for c in 0..cols {
                self.vec_znx_copy(&mut acc, c, &out.to_backend_ref(), c);
                self.vec_znx_add_assign(&mut acc, c, coeffs, m * cols + c);
                self.vec_znx_normalize_assign(base2k, k, 0, &mut acc, c, &mut scratch_4);
            }
        }
        {
            let mut inner = GLWEToBackendMut::<BE>::to_backend_mut(&mut res.inner);
            for c in 0..cols {
                self.vec_znx_copy(inner.data_mut(), c, &acc.to_backend_ref(), c);
            }
        }
        res.threshold = t;
        res.inner.set_canonical(true);
    }

    fn mhe_glwe_shamir_share_aggregate_reference(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>) {
        assert!(res.layout() == a.layout(), "invalid aggregation: shares differ");
        self.glwe_add_assign(&mut res.inner, &a.inner);
    }

    fn mhe_glwe_shamir_share_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn mhe_glwe_shamir_share_finalize_reference(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        // The host combination below bounds its accumulator by canonical digits.
        let mut share = share.clone();
        self.glwe_normalize_assign(&mut share.inner, scratch);
        let share = &share;
        let layout = share.layout();
        assert!(
            actives.len() >= layout.threshold,
            "invalid combination: fewer active parties than the threshold"
        );
        assert!(
            actives.contains(&own),
            "invalid combination: own point not among the active parties"
        );
        assert!(
            actives.iter().enumerate().all(|(i, a)| !actives[..i].contains(a)),
            "invalid combination: duplicate active parties"
        );
        assert!(
            res.n() == layout.n && res.base2k() == layout.base2k && res.k() == layout.k && res.rank() == layout.rank,
            "invalid combination: layouts differ"
        );
        let (rank, d) = (layout.rank.as_usize(), layout.gr_degree);
        let (n, base2k, k) = (layout.n.as_usize(), layout.base2k.as_usize(), layout.k.as_usize());
        let size = k.div_ceil(base2k);
        assert!(
            2 * (base2k - 1) + (d * size).next_power_of_two().ilog2() as usize <= 125,
            "invalid combination: base too large for the accumulator"
        );
        let gr = GaloisRing::new(k, d);
        let weights: Vec<Vec<i64>> = gr
            .constant_terms(&gr.lagrange(own, actives))
            .iter()
            .map(|c| gr.digits(c, base2k, size))
            .collect();

        let src = share.inner.data();
        // Blocks keep each limb row in cache while the digit products sweep it.
        let mut acc = vec![0i128; size * COMBINE_BLOCK];
        for r in 0..rank {
            let limbs: Vec<&[i64]> = (0..d * size).map(|x| src.at(r * d + x / size, x % size)).collect();
            let dst = res.inner.data_mut();
            for start in (0..n).step_by(COMBINE_BLOCK) {
                let len = COMBINE_BLOCK.min(n - start);
                acc.fill(0);
                // Weight digit q times share limb m + q lands in limb m; a negative limb index is an integer.
                for (j, w) in weights.iter().enumerate() {
                    for m in 0..size {
                        let row = &mut acc[m * COMBINE_BLOCK..m * COMBINE_BLOCK + len];
                        for q in 0..size - m {
                            let c = w[q] as i128;
                            for (a, &x) in row.iter_mut().zip(&limbs[j * size + m + q][start..start + len]) {
                                *a += c * x as i128;
                            }
                        }
                    }
                }
                for i in 0..len {
                    let mut carry = 0i128;
                    for m in (0..size).rev() {
                        let v = acc[m * COMBINE_BLOCK + i] + carry;
                        let digit = (v << (128 - base2k)) >> (128 - base2k);
                        carry = (v - digit) >> base2k;
                        acc[m * COMBINE_BLOCK + i] = digit;
                    }
                }
                for m in 0..size {
                    let row = &acc[m * COMBINE_BLOCK..m * COMBINE_BLOCK + len];
                    for (x, &digit) in dst.at_mut(r, m)[start..start + len].iter_mut().zip(row) {
                        *x = digit as i64;
                    }
                }
            }
        }
        res.inner.set_canonical(true);
    }
}

/// `res = y * a` on every rank block of `d` GR components, reduced by `f`:
/// `y^d = -(f_0 + ... + f_{d-1} y^(d-1))`.
fn mul_y<BE: Backend>(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    rank: usize,
    d: usize,
    low: u16,
) where
    Module<BE>: VecZnxZero<BE> + VecZnxCopy<BE> + VecZnxSubAssign<BE>,
{
    for r in 0..rank {
        let base = r * d;
        module.vec_znx_zero(res, base);
        for j in 1..d {
            module.vec_znx_copy(res, base + j, a, base + j - 1);
        }
        for j in (0..d).filter(|j| low >> j & 1 == 1) {
            module.vec_znx_sub_assign(res, base + j, a, base + d - 1);
        }
    }
}

const COMBINE_BLOCK: usize = 256;
