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

use poulpy_hal::{
    api::{VecZnxAutomorphism, VecZnxCopy, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    GLWEBytesOf, GLWEEncryptSk, GLWEMulPlain, ScratchArenaTakeCore,
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

/// Scratch of [`glwe_keyswitch_small`].
pub fn glwe_keyswitch_small_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, key: &K) -> usize
where
    BE: Backend,
    Module<BE>: GLWEMulPlain<BE> + VecZnxRshTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GLWEInfos,
{
    let mask = mask_layout(a);
    BE::scratch_aligned(BE::bytes_of_vec_znx(a.n().into(), 1, a.size()))
        + module
            .glwe_mul_plain_tmp_bytes(res, key, &mask)
            .max(module.vec_znx_rsh_tmp_bytes(res.size()))
}

/// The mask of `a`, as an integer polynomial over all its limbs.
fn mask_layout<A: GLWEInfos>(a: &A) -> GLWEPlaintextLayout {
    GLWEPlaintextLayout {
        n: a.n(),
        base2k: a.base2k(),
        k: ((a.size() * a.base2k().as_usize()) as u32).into(),
    }
}

/// `res = a` under the output secret of `key`, for `a` an integer ciphertext:
/// `res.body = a.body + a.mask * key.body` and `res.mask = a.mask * key.mask`,
/// with the integers of `a` at the bottom of the `res.size()` limbs, where the
/// key holds the input secret.
///
/// `res` is written over all its limbs; its precision label is the caller's.
pub fn glwe_keyswitch_small<BE, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    Module<BE>: GLWEMulPlain<BE> + VecZnxCopy<BE> + VecZnxRshAdd<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GLWEToBackendRef<BE> + GLWEInfos,
{
    let base2k: usize = res.base2k().as_usize();
    assert_eq!(a.base2k().as_usize(), base2k);
    assert_eq!(key.base2k().as_usize(), base2k);
    assert_eq!((res.rank().as_usize(), a.rank().as_usize()), (1, 1), "rank 1 only");
    assert!(key.size() >= res.size(), "the key holds the secret within res");
    assert!(a.size() <= res.size());

    let a_ref = a.to_backend_ref();
    let scratch = scratch.borrow();
    let (mut mask, mut scratch) = scratch.take_glwe_plaintext_scratch(&mask_layout(a));
    module.vec_znx_copy(&mut mask.to_backend_mut().data, 0, &a_ref.data, 1);

    // The product by the integer mask, with no shift: the offset is the mask's width.
    module.glwe_mul_plain(a.size() * base2k, res, key, &mask, &mut scratch.borrow());

    // The body, an integer of `a.size()` limbs, lands on the last limbs of `res`.
    let shift: usize = (res.size() - a.size()) * base2k;
    module.vec_znx_rsh_add(
        base2k,
        shift,
        &mut res.to_backend_mut().data,
        0,
        &a_ref.data,
        0,
        &mut scratch.borrow(),
    );
    res.set_canonical(false);
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
