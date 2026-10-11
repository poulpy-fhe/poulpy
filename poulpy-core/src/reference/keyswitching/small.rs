//! Key switch of a small ciphertext.
//!
//! A ciphertext raised to a wider precision holds a small integer polynomial
//! at the bottom of the Torus: `a` here, read as an integer of `a.size()`
//! limbs. Its mask times a GLWE encryption of the input secret placed at that
//! bottom is its key switch, one product by an integer polynomial of a few
//! limbs. That encryption is the key ([`glwe_small_switching_key_encrypt_sk`]):
//! a GLWE ciphertext, with no gadget, whose precision runs past the bottom by a
//! guard. Nothing is decomposed, and the key noise is scaled by the mask, not
//! by a digit, so the guard must exceed the bits of the masks it multiplies.
//!
//! The key is prepared as the right operand of a convolution
//! ([`glwe_small_switching_key_prepare`]). The masked parts of several switches
//! are summed in the transform domain ([`glwe_keyswitch_small_accumulate`]) and
//! brought back once, with the sum of the bodies ([`glwe_keyswitch_small_finish`]).

use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, ScratchArenaTakeBasic, VecZnxAutomorphism, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecROwned, Module, PrepareHint, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftReborrowBackendRef,
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    GLWEBytesOf, GLWEEncryptSk, GLWENormalize, ScratchArenaTakeCore, glwe_prepare_right,
    layouts::{
        GLWEInfos, GLWEPlaintextLayout, GLWESecret, GLWESecretToBackendMut, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, TorusPrecision, prepared::GLWESecretPreparedToBackendRef,
    },
    reference::encryption::gadget_row_plaintext,
};

/// Scratch of [`glwe_small_switching_key_encrypt_sk`].
pub fn glwe_small_switching_key_encrypt_sk_tmp_bytes<BE, K>(module: &Module<BE>, key: &K) -> usize
where
    BE: Backend,
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalizeTmpBytes,
    K: GLWEInfos,
{
    module.glwe_plaintext_bytes_of_from_infos(key)
        + module
            .glwe_encrypt_sk_tmp_bytes(key)
            .max(module.vec_znx_normalize_tmp_bytes())
}

/// Encrypts in `res`, under `sk_out`, the small switching key of the rank-1
/// secret `sk_in`: `sk_in * 2^-k_secret`, the input secret at the bottom bit
/// of the `k_secret` bits a small switch writes. The `res.k() - k_secret` bits
/// behind it are its guard.
#[allow(clippy::too_many_arguments)]
pub fn glwe_small_switching_key_encrypt_sk<BE, R, S1, S2>(
    module: &Module<BE>,
    res: &mut R,
    k_secret: TorusPrecision,
    sk_in: &S1,
    sk_out: &S2,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    S1: GLWESecretToBackendRef<BE> + GLWEInfos,
    S2: GLWESecretPreparedToBackendRef<BE>,
{
    assert_eq!(sk_in.rank().as_usize(), 1, "rank 1 only");
    assert_eq!(sk_in.n(), res.n());
    assert!(
        k_secret <= res.k(),
        "the secret at bit {k_secret} lies below the key's precision {}",
        res.k()
    );
    let base2k: usize = res.base2k().as_usize();
    let k: usize = res.k().as_usize();
    let sk_in = sk_in.to_backend_ref();
    let scratch = scratch.borrow();
    let (mut pt, mut scratch) = scratch.take_glwe_plaintext_scratch(&GLWEPlaintextLayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
    });
    gadget_row_plaintext(
        module,
        &mut pt.to_backend_mut().data,
        0,
        base2k,
        k,
        0,
        k_secret.as_usize(),
        sk_in.data(),
        0,
        &mut scratch.borrow(),
    );
    module.glwe_encrypt_sk(res, &pt, sk_out, source_xe, source_xa, &mut scratch.borrow());
}

/// Allocates the prepared form of a small switching key, as the right operand
/// of a convolution.
pub fn glwe_small_switching_key_prepared_alloc<BE, K>(module: &Module<BE>, key: &K) -> CnvPVecROwned<BE>
where
    BE: Backend,
    Module<BE>: CnvPVecAlloc<BE>,
    K: GLWEInfos,
{
    module.cnv_pvec_right_alloc(key.n().into(), (key.rank() + 1).into(), key.size(), PrepareHint::Reuse)
}

/// Scratch of [`glwe_small_switching_key_prepare`].
pub fn glwe_small_switching_key_prepare_tmp_bytes<BE, K>(module: &Module<BE>, key: &K) -> usize
where
    BE: Backend,
    Module<BE>: Convolution<BE> + GLWENormalize<BE>,
    K: GLWEInfos,
{
    let size = key.size();
    BE::scratch_aligned(BE::bytes_of_vec_znx(key.n().into(), (key.rank() + 1).into(), size))
        + module
            .cnv_prepare_right_tmp_bytes(size, size)
            .max(module.glwe_normalize_tmp_bytes())
}

/// Prepares `key` into `prepared`.
pub fn glwe_small_switching_key_prepare<BE, K>(
    module: &Module<BE>,
    prepared: &mut CnvPVecROwned<BE>,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: Convolution<BE> + GLWENormalize<BE>,
    K: GLWEToBackendRef<BE> + GLWEInfos,
{
    glwe_prepare_right(module, prepared, key, key.k().as_usize(), scratch);
}

/// The mask of `a`, as an integer polynomial over all its limbs.
fn mask_layout<A: GLWEInfos>(a: &A) -> GLWEPlaintextLayout {
    GLWEPlaintextLayout {
        n: a.n(),
        base2k: a.base2k(),
        k: ((a.size() * a.base2k().as_usize()) as u32).into(),
    }
}

/// Limbs of the transform-domain accumulator of a small switch with a key of
/// `key_size` limbs: the product of the key by an integer of any size, from
/// the key's own top limb down.
pub fn glwe_keyswitch_small_acc_size(key_size: usize) -> usize {
    key_size + 1
}

/// Scratch of [`glwe_keyswitch_small_accumulate`] for a small ciphertext of
/// `a_size` limbs and a key of `key_size` limbs.
pub fn glwe_keyswitch_small_accumulate_tmp_bytes<BE>(module: &Module<BE>, n: usize, a_size: usize, key_size: usize) -> usize
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf,
{
    let acc_size = glwe_keyswitch_small_acc_size(key_size);
    BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, a_size))
        + module.bytes_of_cnv_pvec_left(n, 1, a_size, PrepareHint::OneShot)
        + module
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(module.cnv_apply_dft_tmp_bytes(a_size - 1, acc_size, a_size, key_size))
            .max(module.cnv_apply_dft_add_tmp_bytes(a_size - 1, acc_size, a_size, key_size))
}

/// Adds to `acc` the mask of `a` times `key`: the masked part of the switch
/// of `a`, an integer ciphertext, in the transform domain. `first` overwrites
/// `acc` instead. `acc` has two columns of [`glwe_keyswitch_small_acc_size`]
/// limbs; several terms, each with its own key, are summed there and brought
/// back once by [`glwe_keyswitch_small_finish`].
pub fn glwe_keyswitch_small_accumulate<BE, A>(
    module: &Module<BE>,
    acc: &mut VecZnxDftBackendMut<'_, BE>,
    first: bool,
    a: &A,
    key: &CnvPVecRBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: Convolution<BE> + VecZnxCopy<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_eq!(a.rank().as_usize(), 1, "rank 1 only");
    let a_ref = a.to_backend_ref();
    let size: usize = a.size();
    let scratch = scratch.borrow();
    let (mut mask, scratch) = scratch.take_glwe_plaintext_scratch(&mask_layout(a));
    module.vec_znx_copy(&mut mask.to_backend_mut().data, 0, &a_ref.data, 1);
    let (mut left, mut scratch) = scratch.take_cnv_pvec_left_scratch(a.n().into(), 1, size, PrepareHint::OneShot);
    scratch = scratch.apply_mut(|scratch| module.cnv_prepare_left(&mut left, &mask.to_backend_ref().data, scratch));

    // The offset of the product by an integer is its width: the result keeps the key's alignment.
    let offset: usize = size - 1;
    for col in 0..2 {
        if first {
            module.cnv_apply_dft(
                offset,
                acc,
                col,
                &left.to_backend_ref(),
                0,
                key,
                col,
                1,
                &mut scratch.borrow(),
            );
        } else {
            module.cnv_apply_dft_add(
                offset,
                acc,
                col,
                &left.to_backend_ref(),
                0,
                key,
                col,
                1,
                &mut scratch.borrow(),
            );
        }
    }
}

/// Scratch of [`glwe_keyswitch_small_finish`].
pub fn glwe_keyswitch_small_finish_tmp_bytes<BE, R>(module: &Module<BE>, res: &R, key_size: usize) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxIdftApplyTmpBytes + VecZnxRshTmpBytes,
    R: GLWEInfos,
{
    module.bytes_of_vec_znx_big(res.n().into(), 1, glwe_keyswitch_small_acc_size(key_size))
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes())
            .max(module.vec_znx_rsh_tmp_bytes(res.size()))
}

/// `res` = the terms accumulated in `acc`, plus `body`: the sum of the bodies
/// of the switched ciphertexts, integers of `body.size()` limbs that land on
/// the last limbs of `res`. `res` has as many limbs as the keys have digit
/// limbs; its precision label is the caller's.
pub fn glwe_keyswitch_small_finish<BE, R, A>(
    module: &Module<BE>,
    res: &mut R,
    acc: &mut VecZnxDftBackendMut<'_, BE>,
    body: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxIdftApply<BE> + VecZnxBigNormalize<BE> + VecZnxRshAdd<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let base2k: usize = res.base2k().as_usize();
    assert_eq!(body.base2k().as_usize(), base2k);
    assert_eq!((res.rank().as_usize(), body.rank().as_usize()), (1, 1), "rank 1 only");
    assert!(body.size() <= res.size());
    let res_k: usize = res.size() * base2k;
    let scratch = scratch.borrow();
    let (mut big, mut scratch) = scratch.take_vec_znx_big_scratch(res.n().into(), 1, acc.size());
    for col in 0..2 {
        module.vec_znx_idft_apply(
            &mut big.to_backend_mut(),
            0,
            &acc.reborrow_backend_ref(),
            col,
            &mut scratch.borrow(),
        );
        module.vec_znx_big_normalize(
            &mut res.to_backend_mut().data,
            base2k,
            res_k,
            0,
            col,
            &big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
    }
    module.vec_znx_rsh_add(
        base2k,
        (res.size() - body.size()) * base2k,
        &mut res.to_backend_mut().data,
        0,
        &body.to_backend_ref().data,
        0,
        &mut scratch.borrow(),
    );
    res.set_canonical(false);
}

/// `res = phi_p(a)`, column by column and with no key switch: `res` encrypts the
/// image of the plaintext under `X -> X^p`, under the image of the secret
/// ([`glwe_secret_automorphism`]). `res` and `a` have the same size.
pub fn glwe_automorphism_keyless<BE, R, A>(module: &Module<BE>, res: &mut R, p: i64, a: &A)
where
    BE: Backend,
    Module<BE>: VecZnxAutomorphism<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_eq!((res.size(), res.rank()), (a.size(), a.rank()));
    let a_ref = a.to_backend_ref();
    for col in 0..a.rank().as_usize() + 1 {
        module.vec_znx_automorphism(p, &mut res.to_backend_mut().data, col, &a_ref.data, col);
    }
    res.set_canonical(a_ref.is_canonical());
}

/// `res = phi_p(sk)`, the image of a secret under `X -> X^p`: the secret a
/// ciphertext is under once the same automorphism has been applied to it.
pub fn glwe_secret_automorphism<BE, S>(module: &Module<BE>, res: &mut GLWESecret<BE::OwnedBuf, BE::ZnxWord>, p: i64, sk: &S)
where
    BE: Backend,
    Module<BE>: VecZnxAutomorphism<BE>,
    S: GLWESecretToBackendRef<BE> + GLWEInfos,
{
    let sk = sk.to_backend_ref();
    res.dist = sk.dist;
    let mut res_mut = GLWESecretToBackendMut::<BE>::to_backend_mut(res);
    let sk_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk.data());
    let mut res_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(res_mut.data_mut());
    for i in 0..sk.rank().into() {
        module.vec_znx_automorphism(p, &mut res_backend, i, &sk_backend, i);
    }
}
