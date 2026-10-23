//! Modulus raise with the first `CoeffsToSlots` factor applied on the way.
//!
//! A raised ciphertext is small until its first key switch: integers of a few
//! limbs at the bottom of the Torus. A plaintext product followed by a division
//! keeps it there, at the same precision, so the first factor of the transform
//! consumes no modulus when it runs before any switch
//! ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise)).
//! Each diagonal's term is rotated with no key, which leaves it under the
//! rotated raise secret, then switched to the dense secret by its own key, a
//! switching key whose rows sit `k_offset` bits lower
//! ([`poulpy_core::layouts::GGLWEInfos::k_offset`]): the switch reads the term's
//! digits as balanced integers and lowers them to the bottom of the raised
//! width, which is the raise. The products of all terms are summed in the
//! transform domain ([`poulpy_core::GLWEKeyswitch::glwe_keyswitch_dft_add`]) and
//! rounded once. Under sparse-secret encapsulation the raise secret is the
//! sparse one, and these switches replace the sparse-to-dense one.

use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEMulPlain, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos,
        LinearTransformation, Rank, prepared::GGLWEPreparedToBackendRef,
    },
    reference::keyswitching::glwe::gglwe_product_accumulation_output_size,
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxAutomorphism, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftBytesOf,
        VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena, VecZnxDftToBackendMut, VecZnxToBackendRef, galois_element},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, CoeffsMeta, GLWEPlaintextMeta, Scale, SetCKKSInfos, ckks_bail, ckks_ensure,
    layouts::{BootstrappingKeys, CKKSPlaintextOwned},
};

/// Fewest bits the raise keeps below the raised precision while it applies
/// the factor: the bits a division by a diagonal's scale leaves below the
/// raised ciphertext's last one, rounded once with the sum of the terms.
pub const FIRST_FACTOR_GUARD_BITS: usize = 16;

/// Limbs a raise to `k_large` bits works at, and the bits its working width
/// keeps below them. The first-factor keys' rows end at that width.
pub fn first_factor_work(base2k: usize, k_large: usize) -> (usize, usize) {
    let work = (k_large + FIRST_FACTOR_GUARD_BITS).div_ceil(base2k);
    (work, work * base2k - k_large)
}

/// Bits of the small ciphertext a first-factor key switches, for a raise into
/// `k_large` bits of a ciphertext that fills `raised` bits once lifted: those,
/// the guard of the working width below them, and above them the growth of a
/// product by a diagonal of `diagonal_growth` bits over `2^log_n` terms.
pub fn first_factor_small_bits(base2k: usize, log_n: usize, k_large: usize, raised: usize, diagonal_growth: usize) -> usize {
    let (_, guard) = first_factor_work(base2k, k_large);
    raised + guard + diagonal_growth + log_n + 2
}

/// Bits a diagonal's product adds once divided by its scale: its encoded width
/// past its scale.
fn diagonal_growth<P: IntPolyInfos + CKKSInfos>(diagonal: &P) -> usize {
    diagonal.encoded_k().as_usize() - diagonal.log_delta()
}

/// Bits a product by a diagonal of a transform of `meta`, encoded at radix
/// `base2k`, adds once divided by its scale: the diagonals are allocated to
/// whole limbs.
pub fn first_factor_diagonal_growth(base2k: usize, meta: &CoeffsMeta) -> usize {
    meta.k.as_usize().div_ceil(base2k) * base2k - meta.log_delta()
}

/// Limbs of the small ciphertext a first-factor key switches: its rows sit
/// under it and end at the working width.
fn small_limbs<K: GGLWEInfos>(base2k: usize, k_large: usize, key: &K) -> Result<usize> {
    let (work, _) = first_factor_work(base2k, k_large);
    let k_offset = key.k_offset().as_usize();
    ckks_ensure!(
        key.base2k().as_usize() == base2k && k_offset.is_multiple_of(base2k) && k_offset < work * base2k,
        "a first-factor key lowers its input by {k_offset} bits, not a whole number of {base2k}-bit limbs below the raise's {work} limbs"
    );
    let small = work - k_offset / base2k;
    let layout = key.gglwe_layout();
    let reach = k_offset + layout.gadget_k().as_usize();
    ckks_ensure!(
        layout.max_input_k().as_usize() >= small * base2k && reach >= k_large,
        "a first-factor key takes {} bits of input and reaches bit {reach}; the raise switches {} bits up to bit {k_large}",
        layout.max_input_k(),
        small * base2k
    );
    Ok(small)
}

fn layout<A: GLWEInfos>(infos: &A, size: usize) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: ((size * infos.base2k().as_usize()) as u32).into(),
        rank: Rank(1),
    }
}

/// `res = phi_p(a)` column by column, with no key: `res` is under `phi_p` of
/// the secret of `a`.
pub(crate) fn glwe_automorphism_keyless<BE, R, A>(module: &Module<BE>, res: &mut R, p: i64, a: &A)
where
    BE: Backend,
    Module<BE>: VecZnxAutomorphism<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_eq!((res.size(), res.rank()), (a.size(), a.rank()));
    let a_ref = a.to_backend_ref();
    for col in 0..a.rank().as_usize() + 1 {
        module.vec_znx_automorphism(p, res.to_backend_mut().data_mut(), col, a_ref.data(), col);
    }
    res.set_canonical(a_ref.is_canonical());
}

/// The first-factor key of the rotation `rotation`.
fn first_factor_key<BE: Backend, M: Backend, K: BootstrappingKeys<BE, M>>(keys: &K, rotation: i64) -> Result<&K::SwitchingKey> {
    let Some(key) = keys.first_factor_key(rotation) else {
        ckks_bail!("no first-factor key for the rotation {rotation}");
    };
    Ok(key)
}

/// What [`check_first_factor_raise`] finds a raise with the first factor to
/// work with.
pub(crate) struct FirstFactorRaise {
    /// Layout of every first-factor key.
    pub(crate) key_layout: GGLWELayout,
    /// Limbs of the terms the keys switch.
    pub(crate) small: usize,
    /// Bits a product by any diagonal adds once divided by its scale.
    pub(crate) growth: usize,
}

/// Checks that `factor` and the first-factor keys of `keys` serve a raise into
/// `k_large` bits at radix `base2k` of a ciphertext that fills `raised` bits
/// once lifted: one diagonal per giant step and no baby step, all diagonals of
/// one scale and width, and one key per rotation, all of one layout that fits
/// the raise.
pub(crate) fn check_first_factor_raise<BE, M: Backend, K>(
    base2k: usize,
    log_n: usize,
    k_large: usize,
    raised: usize,
    factor: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    keys: &K,
) -> Result<FirstFactorRaise>
where
    BE: Backend,
    K: BootstrappingKeys<BE, M>,
    CKKSPlaintextOwned<BE>: IntPolyInfos + CKKSInfos,
{
    let Some(diagonal) = factor.first_diagonal_plaintext() else {
        ckks_bail!("the first factor has no diagonal");
    };
    let (growth, width) = (diagonal_growth(diagonal), diagonal.encoded_k());
    let key_layout = first_factor_key(keys, factor.giant_steps[0].rot)?.gglwe_layout();
    for giant in &factor.giant_steps {
        ckks_ensure!(
            giant.diagonals.len() == 1 && giant.diagonals[0].baby == 0,
            "a first factor on the raise has one diagonal per rotation and no baby step"
        );
        let diagonal = &giant.diagonals[0].plaintext;
        ckks_ensure!(
            diagonal_growth(diagonal) == growth && diagonal.encoded_k() == width,
            "the diagonals of the first factor differ in scale or width"
        );
        ckks_ensure!(
            first_factor_key(keys, giant.rot)?.gglwe_layout() == key_layout,
            "the first-factor key of the rotation {} differs in layout from the others",
            giant.rot
        );
    }
    let small = small_limbs(base2k, k_large, &key_layout)?;
    let needed = first_factor_small_bits(base2k, log_n, k_large, raised, growth);
    ckks_ensure!(
        needed <= small * base2k,
        "the first-factor keys switch {} bits, the raise of a ciphertext filling {raised} bits needs {needed}",
        small * base2k
    );
    Ok(FirstFactorRaise {
        key_layout,
        small,
        growth,
    })
}

/// Scratch of [`ckks_mod_up_first_factor_reference`]. `d2s_infos` is the
/// dense-to-sparse key under encapsulation, `key_infos` the layout of the
/// first-factor keys.
pub fn ckks_mod_up_first_factor_tmp_bytes_reference<BE, Dst, Src, D2S, K>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    factor: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    d2s_infos: Option<&D2S>,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEKeyswitch<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxIdftNormalizeTmpBytes,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    K: GGLWEInfos,
    CKKSPlaintextOwned<BE>: IntPolyInfos + CKKSInfos,
{
    let Some(diagonal) = factor.first_diagonal_plaintext() else {
        return 0;
    };
    let base2k = dst_infos.base2k().as_usize();
    let k_large = dst_infos.k().as_usize();
    let d2s = d2s_infos.map_or(0, |key| module.glwe_keyswitch_tmp_bytes(src_infos, src_infos, key));
    // A key that does not fit the raise makes it fail before it takes more.
    let Ok(small) = small_limbs(base2k, k_large, key_infos) else {
        return d2s;
    };
    let small_layout = layout(dst_infos, small);
    let n: usize = dst_infos.n().into();
    let acc_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(
        &layout(dst_infos, k_large.div_ceil(base2k)),
        &small_layout,
        key_infos,
        factor.giant_steps.len(),
    );
    let carved = 3 * BE::scratch_aligned(BE::bytes_of_vec_znx(n, 2, small))
        + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, small))
        + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, acc_size))
        + module.bytes_of_vec_znx_dft(n, 2, acc_size);
    let ops = module
        .glwe_mul_plain_tmp_bytes(&small_layout, &small_layout, diagonal)
        .max(module.glwe_keyswitch_dft_add_tmp_bytes(acc_size, &small_layout, key_infos))
        .max(module.glwe_shift_tmp_bytes(small))
        .max(module.vec_znx_normalize_tmp_bytes())
        .max(module.vec_znx_idft_normalize_tmp_bytes(dst_infos.size(), acc_size));
    d2s.max(carved + ops)
}

/// Raises `src` to the precision of `dst` and applies `factor`, the first
/// factor of the transform, on the way: `dst` is what the ordinary raise
/// followed by that factor gives, at the precision of the raise itself.
///
/// `factor` holds one giant step per diagonal, the diagonal rotated back by
/// its own index and no baby step. `keys` gives the dense-to-sparse key under
/// encapsulation and one first-factor key per rotation, all of one layout.
pub fn ckks_mod_up_first_factor_reference<BE, Dst, Src, M: Backend, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    factor: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWEMulPlain<BE>
        + VecZnxAutomorphism<BE>
        + VecZnxCopy<BE>
        + VecZnxAddAssign<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxNormalize<BE>
        + VecZnxIdftNormalize<BE>
        + CyclotomicOrder,
    K: BootstrappingKeys<BE, M>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    let base2k: usize = dst.base2k().as_usize();
    let k_large: usize = dst.k().as_usize();
    ckks_ensure!(
        dst.rank().as_usize() == 1 && src.rank().as_usize() == 1,
        "the first factor is applied on a raise of rank-1 ciphertexts only"
    );
    // The dense-to-sparse switch leaves `src.k()` as it is: every check runs
    // before any work.
    let k_small: usize = src.k().as_usize();
    ckks_ensure!(
        k_large >= k_small + scale_up,
        "the raise must widen: dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up})"
    );
    // Every key has one layout: one small width for all terms.
    let FirstFactorRaise {
        key_layout,
        small,
        growth,
    } = check_first_factor_raise(
        base2k,
        dst.n().as_usize().ilog2() as usize,
        k_large,
        k_small + scale_up,
        factor,
        keys,
    )?;

    // The raise runs under the sparse secret when the keys encapsulate it.
    if let Some((dense_to_sparse, _)) = keys.encapsulation_keys() {
        module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    }
    let k_offset = key_layout.k_offset.as_usize();
    let (_, guard) = first_factor_work(base2k, k_large);
    let small_layout = layout(dst, small);
    // The cyclotomic order of the bootstrap's degree, which the module may exceed.
    let order = module.cyclotomic_order() / (module.n() / dst.n().as_usize()) as i64;
    let meta = GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    };
    let acc_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(
        &layout(dst, k_large.div_ceil(base2k)),
        &small_layout,
        &key_layout,
        factor.giant_steps.len(),
    );
    let n = dst.n().as_usize();

    {
        let scratch = scratch.borrow();
        let (mut raised, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut product, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut rotated, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut bodies, scratch) = scratch.take_vec_znx_scratch(n, 1, small);
        let (mut lowered, scratch) = scratch.take_vec_znx_scratch(n, 1, acc_size);
        let (mut acc, mut scratch) = scratch.take_vec_znx_dft_scratch(n, 2, acc_size);

        // The raise: the ciphertext as integers, its last bit `guard` bits (and the
        // lift) above the bottom of the small width, which the keys lower to the
        // bottom of the working width.
        module.glwe_copy(&mut raised, &*src, &mut scratch.borrow());
        module.glwe_rsh(
            small * base2k - k_small - scale_up - guard,
            &mut raised,
            &mut scratch.borrow(),
        );

        for (index, giant) in factor.giant_steps.iter().enumerate() {
            // Product and division by the diagonal's scale, at the same width.
            module.glwe_mul_plain(
                growth,
                &mut product,
                &raised,
                &giant.diagonals[0].plaintext,
                &mut scratch.borrow(),
            );

            // The rotation needs no key before the switch.
            let term = if giant.rot == 0 {
                &product
            } else {
                glwe_automorphism_keyless(module, &mut rotated, galois_element(giant.rot, order), &product);
                &rotated
            };

            // The masked part of the switch joins the others in the transform domain;
            // the bodies are summed limb by limb, their integer part kept.
            let key = first_factor_key(keys, giant.rot)?.to_backend_ref();
            let term_ref = term.to_backend_ref();
            if index == 0 {
                module.glwe_keyswitch_dft(&mut acc.to_backend_mut(), term, &key, &mut scratch.borrow());
                module.vec_znx_copy(&mut bodies, 0, term_ref.data(), 0);
            } else {
                module.glwe_keyswitch_dft_add(&mut acc.to_backend_mut(), term, &key, &mut scratch.borrow());
                module.vec_znx_add_assign(&mut bodies, 0, term_ref.data(), 0);
            }
        }

        // One inverse transform for the whole factor, the bodies lowered as the keys
        // lower the masks, rounded once to the raised precision.
        module.vec_znx_normalize(
            &mut lowered,
            base2k,
            acc_size * base2k,
            -(k_offset as i64),
            0,
            &bodies.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
        dst.set_canonical(true);
        let mut dst_ref = dst.to_backend_mut();
        let lowered_ref = lowered.to_backend_ref();
        for col in 0..2 {
            let addend = (col == 0).then_some((&lowered_ref, 0));
            module.vec_znx_idft_normalize(
                dst_ref.data_mut(),
                base2k,
                k_large,
                0,
                col,
                &mut acc.to_backend_mut(),
                col,
                base2k,
                addend,
                &mut scratch.borrow(),
            );
        }
    }

    dst.set_meta(meta);
    Ok(())
}
