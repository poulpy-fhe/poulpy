//! Modulus raise with the first `CoeffsToSlots` factor applied on the way.
//!
//! A raised ciphertext is small until its first key switch: integers of a few
//! limbs at the bottom of the Torus. A plaintext product followed by a division
//! keeps it there, at the same precision, so the first factor of the transform
//! consumes no modulus when it runs before any switch
//! ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise)).
//! Each diagonal's term is rotated with no key, which leaves it under the
//! rotated raise secret, then switched to the dense secret by one product with
//! its own key ([`poulpy_core::reference::keyswitching::small`]). Under
//! sparse-secret encapsulation the raise secret is the sparse one, and these
//! switches replace the sparse-to-dense one.

use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEKeyswitch, GLWEMulPlain, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos,
        LinearTransformation, Rank,
    },
    reference::keyswitching::small::{
        glwe_automorphism_keyless, glwe_keyswitch_small_acc_size, glwe_keyswitch_small_accumulate,
        glwe_keyswitch_small_accumulate_tmp_bytes, glwe_keyswitch_small_finish, glwe_keyswitch_small_finish_tmp_bytes,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ScratchArenaTakeBasic, VecZnxAutomorphism, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxRshAdd,
        VecZnxRshTmpBytes,
    },
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena, VecZnxDftToBackendMut, galois_element},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, GLWEPlaintextMeta, Scale, SetCKKSInfos, ckks_ensure,
    layouts::{BootstrappingKeys, CKKSPlaintextOwned},
};

/// Limb counts of the raise: the small ciphertext and the width its terms are
/// summed at.
struct Widths {
    /// Limbs of the small ciphertext: the raised integers, the guard below
    /// them, and the growth of a product by a diagonal.
    small: usize,
    /// Limbs the terms are switched and summed at: one more than the raised
    /// precision, so each term keeps guard bits and the sum is rounded once.
    work: usize,
    /// Bits from the bottom of the working width to the raised precision: the
    /// small ciphertext keeps its last bit that far above its bottom.
    guard: usize,
}

fn widths(base2k: usize, log_n: usize, k_small: usize, k_large: usize, scale_up: usize, diagonal_growth: usize) -> Widths {
    let work = k_large.div_ceil(base2k) + 1;
    let guard = work * base2k - k_large;
    // The mask of a fresh ciphertext fills `k_small` bits; a product by a
    // diagonal sums up to `n` terms.
    let small = (k_small + scale_up + guard + diagonal_growth + log_n + 2).div_ceil(base2k);
    Widths { small, work, guard }
}

/// Bits a diagonal's product adds once divided by its scale: its plaintext
/// budget.
fn diagonal_growth<P: IntPolyInfos + CKKSInfos>(diagonal: &P) -> usize {
    diagonal.encoded_k().as_usize() - diagonal.log_delta()
}

fn layout<A: GLWEInfos>(infos: &A, size: usize) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: ((size * infos.base2k().as_usize()) as u32).into(),
        rank: Rank(1),
    }
}

/// Scratch of [`ckks_mod_up_first_factor_reference`]. `d2s_infos` is the
/// dense-to-sparse key under encapsulation.
pub fn ckks_mod_up_first_factor_tmp_bytes_reference<BE, Dst, Src, D2S, K>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    scale_up: usize,
    factor: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    d2s_infos: Option<&D2S>,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEKeyswitch<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxRshTmpBytes,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    K: GLWEInfos,
    CKKSPlaintextOwned<BE>: IntPolyInfos + CKKSInfos,
{
    let Some(diagonal) = factor.first_diagonal_plaintext() else {
        return 0;
    };
    let base2k = dst_infos.base2k().as_usize();
    let w = widths(
        base2k,
        dst_infos.n().as_usize().ilog2() as usize,
        src_infos.k().as_usize(),
        dst_infos.max_k().as_usize(),
        scale_up,
        diagonal_growth(diagonal),
    );
    let (small, work) = (layout(dst_infos, w.small), layout(dst_infos, w.work));
    let n: usize = dst_infos.n().into();
    let key_size = key_infos.size();
    let carved = 4 * BE::scratch_aligned(BE::bytes_of_vec_znx(n, 2, w.small))
        + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 2, w.work))
        + module.bytes_of_vec_znx_dft(n, 2, glwe_keyswitch_small_acc_size(key_size));
    let ops = module
        .glwe_mul_plain_tmp_bytes(&small, &small, diagonal)
        .max(glwe_keyswitch_small_accumulate_tmp_bytes(module, n, w.small, key_size))
        .max(glwe_keyswitch_small_finish_tmp_bytes(module, &work, key_size))
        .max(module.glwe_shift_tmp_bytes(w.small))
        .max(module.glwe_normalize_tmp_bytes());
    let d2s = d2s_infos.map_or(0, |key| module.glwe_keyswitch_tmp_bytes(src_infos, src_infos, key));
    d2s.max(carved + ops)
}

/// Raises `src` to the precision of `dst` and applies `factor`, the first
/// factor of the transform, on the way: `dst` is what the ordinary raise
/// followed by that factor gives, at the precision of the raise itself.
///
/// `factor` holds one giant step per diagonal, the diagonal rotated back by
/// its own index and no baby step. `keys` gives the dense-to-sparse key under
/// encapsulation and one first-factor key per rotation.
pub fn ckks_mod_up_first_factor_reference<BE, Dst, Src, K>(
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
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + Convolution<BE>
        + VecZnxAutomorphism<BE>
        + VecZnxCopy<BE>
        + VecZnxRshAdd<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigNormalize<BE>
        + CyclotomicOrder,
    K: BootstrappingKeys<BE>,
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
    let diagonal = factor.first_diagonal_plaintext();
    ckks_ensure!(diagonal.is_some(), "the first factor has no diagonal");
    let diagonal = diagonal.unwrap();
    let growth = diagonal_growth(diagonal);
    let divide = diagonal.encoded_k().as_usize() - growth;

    // The raise runs under the sparse secret when the keys encapsulate it.
    if let Some((dense_to_sparse, _)) = keys.encapsulation_keys() {
        module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    }
    let k_small: usize = src.k().as_usize();
    ckks_ensure!(
        k_large >= k_small + scale_up,
        "the raise must widen: dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up})"
    );
    let w = widths(
        base2k,
        dst.n().as_usize().ilog2() as usize,
        k_small,
        k_large,
        scale_up,
        growth,
    );
    ckks_ensure!(
        w.small <= w.work,
        "the raised width {k_large} leaves no room for the first factor's products ({} limbs against {})",
        w.small,
        w.work
    );
    let (small_layout, work_layout) = (layout(dst, w.small), layout(dst, w.work));
    let order = module.cyclotomic_order();
    let meta = GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    };

    // Every key has the destination's working width as its digit: one size for all.
    let first_key = keys.first_factor_key(factor.giant_steps[0].rot);
    ckks_ensure!(
        first_key.is_some(),
        "no first-factor key for the rotation {}",
        factor.giant_steps[0].rot
    );
    let key_size: usize = first_key.unwrap().size();

    {
        let scratch = scratch.borrow();
        let (mut small, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut product, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut rotated, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut bodies, scratch) = scratch.take_glwe_scratch(&small_layout);
        let (mut sum, scratch) = scratch.take_glwe_scratch(&work_layout);
        let (mut acc, mut scratch) = scratch.take_vec_znx_dft_scratch(dst.n().into(), 2, glwe_keyswitch_small_acc_size(key_size));

        // The raise: the ciphertext as integers, its last bit `guard` bits (and the
        // lift) above the bottom of the small width.
        module.glwe_copy(&mut small, &*src, &mut scratch.borrow());
        module.glwe_rsh(
            w.small * base2k - k_small - scale_up - w.guard,
            &mut small,
            &mut scratch.borrow(),
        );

        for (index, giant) in factor.giant_steps.iter().enumerate() {
            ckks_ensure!(
                giant.diagonals.len() == 1 && giant.diagonals[0].baby == 0,
                "a first factor on the raise has one diagonal per rotation and no baby step"
            );
            let diagonal = &giant.diagonals[0].plaintext;
            ckks_ensure!(
                diagonal_growth(diagonal) == growth && diagonal.encoded_k().as_usize() == divide + growth,
                "the diagonals of the first factor differ in scale or width"
            );
            // Product and division by the diagonal's scale, at the same width.
            module.glwe_mul_plain(growth, &mut product, &small, diagonal, &mut scratch.borrow());

            // The rotation needs no key before the switch.
            let small_term = if giant.rot == 0 {
                &product
            } else {
                glwe_automorphism_keyless(module, &mut rotated, galois_element(giant.rot, order), &product);
                &rotated
            };

            // The masked part of the switch joins the others in the transform domain;
            // the bodies are summed as they are.
            let key = keys.first_factor_key(giant.rot);
            ckks_ensure!(key.is_some(), "no first-factor key for the rotation {}", giant.rot);
            glwe_keyswitch_small_accumulate(
                module,
                &mut acc.to_backend_mut(),
                index == 0,
                small_term,
                &key.unwrap(),
                &mut scratch.borrow(),
            );
            if index == 0 {
                module.glwe_copy(&mut bodies, small_term, &mut scratch.borrow());
            } else {
                module.glwe_add_assign(&mut bodies, small_term);
            }
        }

        // One inverse transform for the whole factor, then rounded once to the raised precision.
        glwe_keyswitch_small_finish(module, &mut sum, &mut acc.to_backend_mut(), &bodies, &mut scratch.borrow());
        dst.set_k(k_large.into());
        module.glwe_normalize(dst, &sum, &mut scratch.borrow());
    }

    dst.set_meta(meta);
    dst.set_k(k_large.into());
    Ok(())
}
