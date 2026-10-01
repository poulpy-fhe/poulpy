use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWEDecrypt, GLWEEncryptPk, GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore,
    SmudgingInfos, VecZnxAddSmudging,
    layouts::{
        GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, prepared::GLWESecretPreparedFactory,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxCopy, VecZnxFillUniformSource,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRshAssign, VecZnxRshTmpBytes, VecZnxSubAssign, VecZnxZero,
    },
    layouts::{
        Backend, HostDataMut, MaxBase2k, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxToBackendRef, ZnxView,
        ZnxViewMut, ZnxWord,
    },
    source::Source,
};

use crate::{
    galois_ring::{GaloisRing, modulus_low},
    layouts::{
        GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned, GLWEShamirLayout, GLWEShamirPolynomialOwned,
        GLWEShamirShareOwned, GLWEWideSecretOwned, GLWEWideSecretPreparedOwned,
    },
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

pub trait GLWEWideSecretPrepareReference<BE: Backend> {
    fn mhe_glwe_wide_secret_prepare_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_wide_secret_prepare_reference(
        &self,
        res: &mut GLWEWideSecretPreparedOwned<BE>,
        sk: &GLWEWideSecretOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend<ZnxWord = i64>> GLWEWideSecretPrepareReference<BE> for Module<BE>
where
    Self: GLWESecretPreparedFactory<BE>,
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn mhe_glwe_wide_secret_prepare_tmp_bytes_reference(&self, layout: &GLWEShamirLayout) -> usize {
        BE::bytes_of_scalar_znx(layout.n.as_usize(), layout.rank.as_usize())
    }

    fn mhe_glwe_wide_secret_prepare_reference(
        &self,
        res: &mut GLWEWideSecretPreparedOwned<BE>,
        sk: &GLWEWideSecretOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let (base2k, k) = (sk.base2k().as_usize(), sk.k().as_usize());
        let size = k.div_ceil(base2k);
        assert!(
            res.digits.len() == size && res.base2k == sk.base2k() && res.k == sk.k() && res.rank() == sk.rank(),
            "invalid preparation: layouts differ"
        );
        let pad = size * base2k - k;
        let src = sk.data();
        for (l, digit) in res.digits.iter_mut().enumerate() {
            let (mut tmp, _) = scratch.borrow().take_glwe_secret_scratch(sk.n(), sk.rank());
            // The last limb holds its digit times 2^pad.
            let shift = if l + 1 == size { pad } else { 0 };
            for r in 0..sk.rank().as_usize() {
                for (x, &y) in tmp.data_mut().at_mut(r, 0).iter_mut().zip(src.at(r, l)) {
                    *x = y >> shift;
                }
            }
            self.glwe_secret_prepare(digit, &tmp);
        }
    }
}

fn wide_mask_product_tmp_bytes<BE: Backend, A: GLWEInfos>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GLWEBytesOf<BE> + GLWENormalize<BE> + GLWEShift<BE> + GLWEDecrypt<BE>,
{
    2 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(infos))
        + BE::scratch_aligned(module.glwe_plaintext_bytes_of_from_infos(infos))
        + module
            .glwe_normalize_tmp_bytes()
            .max(module.glwe_shift_tmp_bytes(infos.size()))
            .max(module.glwe_decrypt_tmp_bytes(infos))
}

/// `res = Sum_j a_j * sk_j` on the masks `a` of `ct`, unnormalized: the sum over
/// the digits `d_l` of `sk` of `(2^(e_l) a) * d_l`, each term a decryption of
/// the shifted ciphertext minus its body.
fn wide_mask_product<BE: Backend, R, C>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    sk: &GLWEWideSecretPreparedOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWENormalize<BE> + GLWEShift<BE> + GLWEDecrypt<BE> + VecZnxZero<BE> + VecZnxSubAssign<BE> + VecZnxAddAssign<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    C: GLWEToBackendRef<BE> + GLWEInfos,
{
    let k = ct.k().as_usize();
    let (base2k, wide_k) = (sk.base2k.as_usize(), sk.k.as_usize());
    let size = wide_k.div_ceil(base2k);
    let (mut ct_norm, scratch_1) = scratch.borrow().take_glwe_scratch(ct);
    let (mut shifted, scratch_2) = scratch_1.take_glwe_scratch(ct);
    let (mut part, mut scratch_3) = scratch_2.take_glwe_plaintext_scratch(ct);
    module.glwe_normalize(&mut ct_norm, ct, &mut scratch_3);
    module.vec_znx_zero(res.to_backend_mut().data_mut(), 0);
    for (l, digit) in sk.digits.iter().enumerate() {
        let shift = if l + 1 == size { 0 } else { wide_k - (l + 1) * base2k };
        // The masks are in 2^-k Z, so 2^shift a vanishes modulo 1 from shift = k on.
        if shift >= k {
            continue;
        }
        module.glwe_lsh(&mut shifted, &ct_norm, shift, &mut scratch_3);
        module.glwe_decrypt(&shifted, &mut part, digit, &mut scratch_3);
        module.vec_znx_sub_assign(part.data_mut(), 0, shifted.to_backend_ref().data(), 0);
        module.vec_znx_add_assign(res.to_backend_mut().data_mut(), 0, part.to_backend_ref().data(), 0);
    }
}

fn assert_wide_secret<BE: Backend + MaxBase2k, C: GLWEInfos>(
    module: &Module<BE>,
    ct: &C,
    sk: &GLWEWideSecretPreparedOwned<BE>,
    failure_bits: usize,
) {
    assert!(
        ct.n().as_usize() == module.n(),
        "invalid share: ciphertext degree differs from the module's"
    );
    assert!(ct.rank() > 0, "invalid share: ciphertext rank must be positive");
    assert!(!sk.digits.is_empty(), "invalid share: wide secret has no digits");
    assert!(
        sk.digits.iter().all(|digit| digit.n() == ct.n()),
        "invalid share: wide secret degree differs from the ciphertext's"
    );
    assert!(
        sk.k() >= ct.k(),
        "invalid share: wide secret less precise than the ciphertext"
    );
    assert!(
        sk.rank() == ct.rank(),
        "invalid share: wide secret rank differs from the ciphertext's"
    );
    assert!(failure_bits > 0, "invalid share: numerical failure target must be positive");
    let (k, wide_k, base2k) = (ct.k().as_usize(), sk.k().as_usize(), sk.base2k().as_usize());
    let digits = sk
        .digits
        .iter()
        .enumerate()
        .filter(|(l, _)| {
            let shift = if l + 1 == sk.digits.len() {
                0
            } else {
                wide_k - (l + 1) * base2k
            };
            shift < k
        })
        .count();
    // Each digit contributes a difference of two canonical polynomials,
    // bounded by 2^B per coefficient. Reserve one more such difference for
    // CKS, and respect normalization's stricter word-width-minus-two limit.
    let terms = digits.checked_add(1).expect("invalid share: too many digit products");
    let headroom_bits = (usize::BITS - terms.saturating_sub(1).leading_zeros()) as usize;
    assert!(
        ct.base2k().as_usize() + headroom_bits <= BE::ZnxWord::BITS - 2,
        "invalid share: wide-secret accumulation exceeds coefficient headroom"
    );
    // Each contributing digit computes one rank-term product per ciphertext
    // limb. Allocate the caller's total failure budget with a union bound.
    let outputs = ct.size().checked_mul(digits).expect("invalid share: too many digit products");
    let union_bits = (usize::BITS - outputs.saturating_sub(1).leading_zeros()) as usize;
    let per_output_bits = failure_bits
        .checked_add(union_bits)
        .expect("invalid share: numerical failure target is too large");
    let max_base2k = Module::<BE>::max_base2k(ct.n().as_usize(), ct.rank().as_usize(), per_output_bits, false)
        .expect("invalid share: backend has no wide-secret product budget");
    assert!(
        ct.base2k().as_usize() + base2k <= 2 * max_base2k,
        "invalid share: wide-secret digit products exceed the backend budget"
    );
}

pub trait GLWEThresholdKeyswitchMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_keyswitch_share_gen_reference<C, S, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        sk_out: &S,
        flood: &E,
        failure_bits: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos;
}

impl<BE: Backend + MaxBase2k> GLWEThresholdKeyswitchMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEBytesOf<BE>
        + GLWENormalize<BE>
        + GLWEShift<BE>
        + GLWEDecrypt<BE>
        + GLWESub<BE>
        + VecZnxAddSmudging<BE>
        + VecZnxZero<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>,
{
    fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        2 * BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(infos))
            + wide_mask_product_tmp_bytes(self, infos)
                .max(self.glwe_decrypt_tmp_bytes(infos))
                .max(self.glwe_normalize_tmp_bytes())
    }

    fn mhe_glwe_threshold_keyswitch_share_gen_reference<C, S, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        sk_out: &S,
        flood: &E,
        failure_bits: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos,
    {
        let res = &mut res.inner;
        assert_wide_secret(self, ct, sk_in, failure_bits);
        assert!(
            res.n() == ct.n() && res.base2k() == ct.base2k(),
            "invalid share: share and ciphertext layouts differ"
        );
        assert!(res.rank() == 0, "invalid share: share rank differs from 0");
        assert!(
            sk_out.n() == ct.n(),
            "invalid share: output secret degree differs from the ciphertext's"
        );
        assert!(
            sk_out.rank() == ct.rank(),
            "invalid share: output secret rank differs from the ciphertext's"
        );
        let base2k = res.base2k().as_usize();
        let flood_noise = super::assert_flood::<BE, _>(base2k, res.k().as_usize(), flood);
        let (mut pt_in, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        let (mut pt_out, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(ct);
        wide_mask_product(self, &mut pt_in, ct, sk_in, &mut scratch_2);
        {
            // Reuse the wide-product scratch so the body subtraction has
            // canonical operands even when the input ciphertext is lazy.
            let (mut ct_norm, mut scratch_3) = scratch_2.borrow().take_glwe_scratch(ct);
            self.glwe_normalize(&mut ct_norm, ct, &mut scratch_3);
            self.glwe_decrypt(&ct_norm, &mut pt_out, sk_out, &mut scratch_3);
            self.vec_znx_sub_assign(pt_out.data_mut(), 0, ct_norm.to_backend_ref().data(), 0);
        }
        self.glwe_sub(res, &pt_in, &pt_out);
        self.glwe_normalize_assign(res, &mut scratch_2);
        self.vec_znx_add_smudging(
            base2k,
            GLWEToBackendMut::<BE>::to_backend_mut(res).data_mut(),
            0,
            flood_noise,
            source_xe,
        );
        self.glwe_normalize_assign(res, &mut scratch_2);
    }
}

pub trait GLWEThresholdPublicKeyswitchMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes_reference<A, B, P>(
        &self,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_public_keyswitch_share_gen_reference<C, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        failure_bits: usize,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;
}

impl<BE: Backend + MaxBase2k> GLWEThresholdPublicKeyswitchMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEBytesOf<BE>
        + GLWENormalize<BE>
        + GLWEShift<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptPk<BE>
        + VecZnxAddSmudging<BE>
        + VecZnxZero<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>,
{
    fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes_reference<A, B, P>(
        &self,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + wide_mask_product_tmp_bytes(self, ct_infos)
                .max(self.glwe_normalize_tmp_bytes())
                .max(self.glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos))
    }

    fn mhe_glwe_threshold_public_keyswitch_share_gen_reference<C, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        failure_bits: usize,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos,
    {
        let res = &mut res.inner;
        assert_wide_secret(self, ct, sk_in, failure_bits);
        assert!(
            res.n() == ct.n() && res.base2k() == ct.base2k(),
            "invalid share: share and ciphertext layouts differ"
        );
        assert!(
            pk_out.n() == res.n() && pk_out.base2k() == res.base2k() && pk_out.rank() == res.rank(),
            "invalid share: public key and share layouts differ"
        );
        assert!(pk_out.k() >= res.k(), "invalid share: public key less precise than the share");
        let base2k = ct.base2k().as_usize();
        let flood_noise = super::assert_flood::<BE, _>(base2k, ct.k().as_usize(), flood);
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        wide_mask_product(self, &mut pt, ct, sk_in, &mut scratch_1);
        self.glwe_normalize_assign(&mut pt, &mut scratch_1);
        self.vec_znx_add_smudging(base2k, pt.data_mut(), 0, flood_noise, source_xe);
        self.glwe_normalize_assign(&mut pt, &mut scratch_1);
        self.glwe_encrypt_pk(res, &pt, pk_out, enc_infos, source_xu, source_xe, &mut scratch_1);
    }
}
