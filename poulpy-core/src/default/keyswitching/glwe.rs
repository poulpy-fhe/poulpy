use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedBackendRef, GLWEBigToBackendMut, GLWEBigToBackendRef, GLWEInfos, GLWEToBackendRef, LWEInfos,
        glwe_keyswitch_big_layout, prepared::GGLWEPreparedToBackendRef,
    },
};

impl<BE: Backend> GLWEKeyswitchInternal<BE> for Module<BE> where Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE> {}

pub(crate) trait GLWEKeyswitchInternal<BE: Backend>
where
    Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE>,
{
    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let cols: usize = (a_infos.rank() + 1).into();
        let a_size: usize = a_infos.size();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols - 1, a_size);
        let lvl_1: usize = self.gglwe_product_dft_tmp_bytes_default(res_infos.size(), a_size, key_infos);
        lvl_0 + lvl_1
    }

    fn glwe_keyswitch_internal<'r, A, K>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let a = a.to_backend_ref();
        let key: GGLWEPreparedBackendRef<'_, BE> = key.to_backend_ref();
        assert_eq!(a.base2k(), key.base2k());
        assert!(
            scratch.available() >= self.glwe_keyswitch_internal_tmp_bytes(&key, &a, &key),
            "scratch.available(): {} < GLWEKeyswitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_keyswitch_internal_tmp_bytes(&key, &a, &key)
        );
        let cols: usize = (a.rank() + 1).into();
        let a_size: usize = a.size();
        scratch.scope(|scratch_phase| {
            let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(self, cols - 1, a_size);
            for col_i in 0..cols - 1 {
                self.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, &a.data, col_i + 1);
            }
            self.gglwe_product_dft_default(res, &a_dft.to_backend_ref(), &key, &mut scratch_1.borrow());
        });
    }
}

impl<BE: Backend> GGLWEProductDefault<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftCopy<BE>
{
}

pub(crate) trait GGLWEProductDefault<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftCopy<BE>,
{
    fn gglwe_product_dft_tmp_bytes_default<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        let dsize: usize = key_infos.dsize().as_usize();

        if dsize == 1 {
            let lvl_0: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                key_infos.dnum().into(),
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );
            lvl_0
        } else {
            let dnum: usize = key_infos.dnum().into();
            let a_size: usize = a_size.div_ceil(dsize).min(dnum);
            let cols_out: usize = (key_infos.rank_out() + 1).into();
            let lvl_0: usize = self.bytes_of_vec_znx_dft(key_infos.rank_in().into(), a_size);
            let lvl_1: usize = self.bytes_of_vec_znx_dft(cols_out, key_infos.size());
            let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                dnum,
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );

            lvl_0 + lvl_1 + lvl_2
        }
    }

    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let cols: usize = a.cols();
        let a_size: usize = a.size();
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key),
            "scratch.available(): {} < GGLWEProductDefault::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key)
        );

        if key.dsize() == 1 {
            self.vmp_apply_dft_to_dft(res, a, &key.data, 0, scratch);
        } else {
            let dsize: usize = key.dsize().into();
            let dnum: usize = key.dnum().into();

            // `di == 0` is the overwriting pass and must leave no limb of `res`
            // holding stale scratch, so it runs at the **full** width: the VMP
            // zeroes whatever it does not compute, which is exactly the tail the
            // narrowed view used to leave untouched. Two properties make this
            // the only workable shape, and both are easy to break:
            //
            //   - the overwrite must be `di == 0`. `vmp_apply_dft_to_dft` covers
            //     `res` fully only at `limb_offset == 0`; at any other offset it
            //     writes `0..col_max - limb_offset` and zeroes from `col_max`,
            //     leaving a gap in between. So the digits cannot be walked in
            //     reverse to get a wider first pass.
            //   - the accumulating passes may keep the narrow view, since every
            //     limb they skip was already written by `di == 0`.
            //
            // Net effect: callers hand in arbitrary scratch. Before this, the
            // top `dsize - 2` limbs were outside the overwriting view and
            // silently accumulated stale bytes, so each caller had to pre-zero —
            // an obligation invisible at the call site and, on the automorphism
            // path, not actually met.
            for di in 0..dsize {
                let (mut ai_dft, mut scratch_1) =
                    scratch
                        .borrow()
                        .take_vec_znx_dft_scratch(self, cols, ((a_size + di) / dsize).min(dnum));

                for j in 0..cols {
                    self.vec_znx_dft_copy(dsize, dsize - di - 1, &mut ai_dft, j, a, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(res, &ai_dft.to_backend_ref(), &key.data, 0, &mut scratch_1.borrow());
                } else {
                    // Accumulate directly into res, folding the per-column DFT add into the
                    // VMP save (drops the res_dft_tmp buffer + the separate add pass).
                    // Pass `di` consumes `a`'s limbs at offset `dsize - di - 1`
                    // within each digit, so its product sits `dsize - 1 - di`
                    // limbs below the top. That is the bound for a *point*
                    // contribution, and it is not the one to use: an elementary
                    // limb product has magnitude ~`2^(2*base2k + log_n)`, so it
                    // spans at least two limbs rather than landing in one, and
                    // reaches one limb further down than the naive count. Hence
                    // `- 2`, not `- 1`.
                    //
                    // Do not "tighten" this to `- 1`: the keyswitch noise sweep
                    // does not catch it (the difference measured 1e-6 bits at
                    // n=2^12, base2k=18), so a green suite is not evidence that
                    // the limb was free.
                    let res_compute_size = res.size() - ((dsize - di) as isize - 2).max(0) as usize;
                    let mut res_view = res.with_size_mut(res_compute_size);
                    self.vmp_apply_dft_to_dft_accumulate(
                        &mut res_view,
                        &ai_dft.to_backend_ref(),
                        &key.data,
                        di,
                        &mut scratch_1.borrow(),
                    );
                }
            }
        }
    }
}

// === Free-function defaults for GLWEKeyswitchDefault ===

use poulpy_hal::api::{
    VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftApply,
    VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
};

use crate::{
    default::operations::GLWENormalizeDefault,
    layouts::{GLWELayout, GLWEToBackendMut},
    oep::GLWEKeyswitchDefault,
};

#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    // Mirrors the staged composition: the big-domain accumulator is carved
    // first and stays live across both stages, then each stage's own scratch
    // peaks inside it.
    let cols: usize = res_infos.rank().as_usize() + 1;
    let res_big: usize = module.bytes_of_vec_znx_big(cols, key_infos.size());
    let into_big: usize = glwe_keyswitch_into_big_tmp_bytes_default(module, res_infos, a_infos, key_infos);
    let finalize: usize = module.vec_znx_big_normalize_tmp_bytes();

    res_big + into_big.max(finalize)
}

/// Scratch required by [`glwe_keyswitch_into_big_default`].
#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_into_big_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let from_mask: usize = glwe_keyswitch_from_mask_into_big_tmp_bytes_default(module, res_infos, a_infos, key_infos);

    if a_infos.base2k() != key_infos.base2k() {
        let a_conv_infos: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let mask: usize = module.bytes_of_vec_znx_dft(a_conv_infos.rank().as_usize(), a_conv_infos.size());
        let from_mask_conv: usize =
            glwe_keyswitch_from_mask_into_big_tmp_bytes_default(module, res_infos, &a_conv_infos, key_infos);
        module.glwe_bytes_of_from_infos(&a_conv_infos) + module.glwe_normalize_tmp_bytes_default().max(mask + from_mask_conv)
    } else {
        module.bytes_of_vec_znx_dft(a_infos.rank().as_usize(), a_infos.size()) + from_mask
    }
}

/// Transforms a GLWE's mask columns (`a`'s columns `1..`) into `res`.
///
/// A plain `VecZnxDft` with `rank` columns, not a wrapper type: the mask is an
/// operation intermediate, not a semantic object.
#[doc(hidden)]
pub fn glwe_mask_dft_apply_default<BE, M, A>(module: &M, res: &mut VecZnxDftBackendMut<'_, BE>, a: &A)
where
    BE: Backend,
    M: VecZnxDftApply<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let a_ref = a.to_backend_ref();
    let rank: usize = a.rank().as_usize();
    for col in 0..rank {
        module.vec_znx_dft_apply(1, 0, res, col, &a_ref.data, col + 1);
    }
}

/// Scratch required by [`glwe_keyswitch_from_mask_into_big_default`].
#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_from_mask_into_big_tmp_bytes_default<BE, M, R, A, K>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: ModuleN + GGLWEProductDefault<BE> + VecZnxDftBytesOf + VecZnxIdftApplyTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let cols: usize = res_infos.rank().as_usize() + 1;
    // `res_dft` is the VMP destination and then the IDFT source, so it is live
    // across both phases below.
    let res_dft: usize = module.bytes_of_vec_znx_dft(cols, key_infos.size());
    let vmp: usize = module.gglwe_product_dft_tmp_bytes_default(res_infos.size(), a_infos.size(), key_infos);
    res_dft + vmp.max(module.vec_znx_idft_apply_tmp_bytes())
}

/// Keyswitch from an **already-transformed mask**, stopping in the big domain.
///
/// This is the hoisting entry point: transform a ciphertext's mask once with
/// [`GLWEKeyswitchIntoBig::glwe_mask_dft_apply`](crate::api::GLWEKeyswitchIntoBig::glwe_mask_dft_apply), then call this for each key.
/// `a` is still required because the body is never transformed and is folded in
/// here, in the big domain.
///
/// Requires `a.base2k() == key.base2k()`: with a pre-transformed mask there is
/// no opportunity to insert the conversion, so the caller must have matched them
/// (or use [`glwe_keyswitch_into_big_default`], which handles the mismatch).
#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_from_mask_into_big_default<BE, M, R, A, K>(
    module: &M,
    res_big: &mut R,
    mask_dft: &VecZnxDftBackendRef<'_, BE>,
    a: &A,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes,
    R: GLWEBigToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    assert_eq!(
        a.base2k(),
        key.base2k(),
        "a pre-transformed mask cannot absorb a base2k conversion; normalize `a` to the key's base2k first"
    );
    // `mask_dft` holds the transformed mask columns only, so its column count is
    // the rank itself, with no `+ 1` for a body.
    assert_eq!(
        mask_dft.cols(),
        key.rank_in().as_usize(),
        "mask_dft must hold key.rank_in() columns"
    );
    assert_eq!(res_big.rank(), key.rank_out(), "res_big rank must match key.rank_out()");
    assert_eq!(res_big.base2k(), key.base2k());
    assert!(
        scratch.available() >= glwe_keyswitch_from_mask_into_big_tmp_bytes_default(module, res_big, a, key),
        "scratch.available(): {} < glwe_keyswitch_from_mask_into_big_tmp_bytes: {}",
        scratch.available(),
        glwe_keyswitch_from_mask_into_big_tmp_bytes_default(module, res_big, a, key)
    );

    let key_size: usize = key.work_size(a.k());
    let cols: usize = (res_big.rank() + 1).into();
    let key_ref: GGLWEPreparedBackendRef<'_, BE> = key.to_backend_ref();

    scratch.scope(|scratch_phase| {
        // No pre-zeroing: `gglwe_product_dft_default` overwrites every limb of
        // `res_dft` on its first digit pass, whatever `dsize`.
        let (mut res_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, cols, key_size);
        module.gglwe_product_dft_default(&mut res_dft, mask_dft, &key_ref, &mut scratch_1.borrow());

        let res_dft_ref = res_dft.to_backend_ref();
        let mut res_big_mut = res_big.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_idft_apply(&mut res_big_mut.data, col, &res_dft_ref, col, &mut scratch_1.borrow());
        }
        module.vec_znx_big_add_small_assign(&mut res_big_mut.data, 0, &a.to_backend_ref().data, 0);
    });
}

/// Keyswitch, stopping one stage early: leaves an un-normalized accumulator in
/// the big domain instead of carrying it back to the coefficient domain.
///
/// `res_big` must be carved from [`glwe_keyswitch_big_layout`]. Its `base2k`
/// and `k` describe the accumulator (see the domain contract in
/// `docs/core-domain-generic-layouts-plan.md`, Phase 0); turning it into a
/// precision claim is [`glwe_finalize_big_default`]'s job, and that reads the
/// precision from the *destination*, never from here.
#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_into_big_default<BE, M, R, A, K>(
    module: &M,
    res_big: &mut R,
    a: &A,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEBigToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let expected: GLWELayout = glwe_keyswitch_big_layout(a, key);
    assert_eq!(
        res_big.base2k(),
        expected.base2k,
        "res_big must be carved at the key's base2k (glwe_keyswitch_big_layout)"
    );
    assert_eq!(
        res_big.rank(),
        expected.rank,
        "res_big.rank(): {} != key.rank_out(): {}",
        res_big.rank(),
        expected.rank
    );
    assert_eq!(
        a.rank(),
        key.rank_in(),
        "a.rank(): {} != key.rank_in(): {}",
        a.rank(),
        key.rank_in()
    );
    assert_eq!(res_big.n(), module.n() as u32);
    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);
    assert!(
        scratch.available() >= glwe_keyswitch_into_big_tmp_bytes_default(module, res_big, a, key),
        "scratch.available(): {} < glwe_keyswitch_into_big_tmp_bytes: {}",
        scratch.available(),
        glwe_keyswitch_into_big_tmp_bytes_default(module, res_big, a, key)
    );

    let a_base2k: usize = a.base2k().into();
    let key_base2k: usize = key.base2k().into();

    // Both branches reduce to "hold the input at the key's base2k, transform its
    // mask once, then run the from-mask stage". When the widths differ, the
    // single conversion serves the mask product *and* the body fold, which the
    // previous shape re-normalized separately.
    scratch.scope(|scratch_phase| {
        if a_base2k != key_base2k {
            let (mut a_conv, mut scratch_1) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            module.glwe_normalize_default(&mut a_conv, a, &mut scratch_1.borrow());
            let (mut mask_dft, mut scratch_2) =
                scratch_1.take_vec_znx_dft_scratch(module, a_conv.rank().as_usize(), a_conv.size());
            glwe_mask_dft_apply_default(module, &mut mask_dft, &a_conv);
            glwe_keyswitch_from_mask_into_big_default(
                module,
                res_big,
                &mask_dft.to_backend_ref(),
                &a_conv,
                key,
                &mut scratch_2.borrow(),
            );
        } else {
            let (mut mask_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, a.rank().as_usize(), a.size());
            glwe_mask_dft_apply_default(module, &mut mask_dft, a);
            glwe_keyswitch_from_mask_into_big_default(
                module,
                res_big,
                &mask_dft.to_backend_ref(),
                a,
                key,
                &mut scratch_1.borrow(),
            );
        }
    });
}

/// Scratch required by [`glwe_finalize_big_default`].
#[doc(hidden)]
pub fn glwe_finalize_big_tmp_bytes_default<BE, M>(module: &M) -> usize
where
    BE: Backend,
    M: VecZnxBigNormalizeTmpBytes,
{
    let _ = std::marker::PhantomData::<BE>;
    module.vec_znx_big_normalize_tmp_bytes()
}

/// Carries a big-domain accumulator back to the coefficient domain.
///
/// The output precision is taken from `res`, never from `a_big`: this is the
/// single point where an accumulator becomes a precision claim.
#[doc(hidden)]
pub fn glwe_finalize_big_default<BE, M, R, A>(module: &M, res: &mut R, a_big: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEBigToBackendRef<BE> + GLWEInfos,
{
    assert_eq!(res.rank(), a_big.rank(), "res.rank() != a_big.rank()");
    assert!(
        scratch.available() >= module.vec_znx_big_normalize_tmp_bytes(),
        "scratch.available(): {} < vec_znx_big_normalize_tmp_bytes: {}",
        scratch.available(),
        module.vec_znx_big_normalize_tmp_bytes()
    );

    let res_base2k: usize = res.base2k().into();
    let a_base2k: usize = a_big.base2k().into();
    let cols: usize = (res.rank() + 1).into();

    let a_ref = a_big.to_backend_ref();
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            0,
            i,
            &a_ref.data,
            a_base2k,
            i,
            &mut scratch.borrow(),
        );
    }
}

#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_default<BE, M, R, A, K>(module: &M, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEKeyswitchDefault<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    assert_eq!(
        a.rank(),
        key.rank_in(),
        "a.rank(): {} != b.rank_in(): {}",
        a.rank(),
        key.rank_in()
    );
    assert_eq!(
        res.rank(),
        key.rank_out(),
        "res.rank(): {} != b.rank_out(): {}",
        res.rank(),
        key.rank_out()
    );

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes_default(res, a, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes_default(res, a, key)
    );

    // The old single-shot body is now the composition of the two staged
    // operations, so the staged path cannot drift from the ergonomic one.
    let big_layout: GLWELayout = glwe_keyswitch_big_layout(a, key);
    scratch.scope(|scratch_phase| {
        let (mut res_big, mut scratch_1) = scratch_phase.take_glwe_big_scratch(module, &big_layout);
        glwe_keyswitch_into_big_default(module, &mut res_big, a, key, &mut scratch_1.borrow());
        glwe_finalize_big_default(module, res, &res_big, &mut scratch_1.borrow());
    });
}

#[allow(private_bounds)]
#[doc(hidden)]
pub fn glwe_keyswitch_assign_default<BE, M, R, K>(module: &M, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEKeyswitchDefault<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    assert_eq!(
        res.rank(),
        key.rank_in(),
        "res.rank(): {} != a.rank_in(): {}",
        res.rank(),
        key.rank_in()
    );
    assert_eq!(
        res.rank(),
        key.rank_out(),
        "res.rank(): {} != b.rank_out(): {}",
        res.rank(),
        key.rank_out()
    );

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes_default(res, res, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes_default(res, res, key)
    );

    // Same three stages as the out-of-place path, so it is the same
    // composition with `a` aliasing the destination: `into_big` only reads
    // `res`, and `finalize` only writes it, so the in-place form is safe
    // without a second copy.
    let big_layout: GLWELayout = glwe_keyswitch_big_layout(&*res, key);
    scratch.scope(|scratch_phase| {
        let (mut res_big, mut scratch_1) = scratch_phase.take_glwe_big_scratch(module, &big_layout);
        glwe_keyswitch_into_big_default(module, &mut res_big, &*res, key, &mut scratch_1.borrow());
        glwe_finalize_big_default(module, res, &res_big, &mut scratch_1.borrow());
    });
}
