use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes, VmpApplyDftToDftTmpBytes, VmpExtractSelectedRows,
        VmpPMatBytesOf,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VmpPMatBackendRef, VmpPMatToBackendRef,
    },
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedBackendRef, GLWEInfos, GLWEToBackendRef, GadgetProductOutputSizeParams, LWEInfos,
        gadget_product_limbs, gadget_product_output_size,
    },
    oep::{GGLWEProductDigitsStridedImpl, gglwe_product_digit_output_size},
};

impl<BE: Backend> GLWEKeyswitchInternal<BE> for Module<BE> where Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE> {}

/// DFT-domain plumbing shared by the key-switch reference bodies.
///
/// Public because it appears in the `where` clause of the public
/// `glwe_keyswitch*_default` functions: a backend forwarding to them by hand,
/// rather than through [`crate::impl_glwe_keyswitch_defaults_full`], has to be
/// able to name the bound. Blanket-implemented for every `Module<BE>` that has
/// the underlying HAL ops, so there is nothing to implement.
pub trait GLWEKeyswitchInternal<BE: Backend>
where
    Self: GGLWEProductDefault<BE> + VecZnxDftApply<BE>,
{
    fn glwe_keyswitch_internal_tmp_bytes_from_sizes<K>(
        &self,
        mask_cols: usize,
        res_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        let lvl_0: usize = self.bytes_of_vec_znx_dft(mask_cols, a_size);
        let lvl_1: usize = self.gglwe_product_dft_tmp_bytes_default(res_size, a_size, key_infos);
        lvl_0 + lvl_1
    }

    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_internal_tmp_bytes_from_sizes(a_infos.rank().as_usize(), res_infos.size(), a_infos.size(), key_infos)
    }

    fn glwe_keyswitch_internal<'r, A>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
    {
        glwe_keyswitch_dft_fill(self, res, a, key, scratch);
    }
}

impl<BE: Backend> GGLWEProductDefault<BE> for Module<BE>
where
    BE: GGLWEProductDigitsStridedImpl<BE>,
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxDftCopy<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    fn gglwe_product_dft_tmp_bytes_default<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        let dsize: usize = key_infos.dsize().into();
        let dnum: usize = key_infos.dnum().into();
        let cols_in: usize = key_infos.rank_in().into();
        let cols_out: usize = (key_infos.rank_out() + 1).into();
        let key_size: usize = key_infos.size();
        let product: usize = if dsize == 1 {
            self.vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, dnum, cols_in, cols_out, key_size)
        } else {
            BE::gglwe_product_digits_strided_tmp_bytes(self, res_size, cols_in, a_size, dsize, dnum, cols_in, cols_out, key_size)
        };
        if key_infos.stride() == 1 {
            product
        } else {
            self.bytes_of_vmp_pmat(dnum, cols_in, cols_out, key_size) + product
        }
    }

    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let a_size = a.size();
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key),
            "scratch.available(): {} < GGLWEProductDefault::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, key)
        );
        // A view reading one row out of every `stride` is gathered into a dense
        // matrix first, so the kernels below never see the row map.
        let stride: usize = key.stride();
        if stride == 1 {
            gglwe_product_pmat(self, res, a, key, &key.data, term_count, scratch);
            return;
        }
        let (rows, cols_in, cols_out) = (
            key.dnum().as_usize(),
            key.rank_in().as_usize(),
            (key.rank_out() + 1).as_usize(),
        );
        let key_size: usize = key.size();
        scratch.scope(|scratch_phase| {
            let (mut dense, mut scratch_1) = scratch_phase.take_vmp_pmat_scratch(self, rows, cols_in, cols_out, key_size);
            self.vmp_extract_selected_rows(&mut dense, &key.data, stride - 1, stride);
            gglwe_product_pmat(
                self,
                res,
                a,
                key,
                &dense.to_backend_ref(),
                term_count,
                &mut scratch_1.borrow(),
            );
        });
    }
}

/// One GGLWE product against an already dense matrix, with the key's own
/// layout supplying the digit geometry.
fn gglwe_product_pmat<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    pmat: &VmpPMatBackendRef<'_, BE>,
    term_count: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + GGLWEProductDigitsStridedImpl<BE>,
    Module<BE>: VmpApplyDftToDft<BE>,
{
    // One limb per digit is a plain VMP, not a strided gather; the hook below
    // is only ever entered with `dsize >= 2`.
    let dsize: usize = key.dsize().into();
    if dsize == 1 {
        module.vmp_apply_dft_to_dft(res, a, pmat, 0, scratch);
    } else {
        let product_terms = key
            .n()
            .as_usize()
            .saturating_mul(key.dnum().as_usize())
            .saturating_mul(dsize)
            .saturating_mul(key.rank_in().as_usize().max(1))
            .saturating_mul(term_count.max(1));
        let product_limbs = gadget_product_limbs(key.base2k(), product_terms);
        BE::gglwe_product_digits_strided(module, res, a, dsize, product_limbs, pmat, scratch);
    }
}

/// Default DFT-domain gadget product used by key-switching and external products.
///
/// Public so backend forwarders can name the bound. It centralizes the
/// `dsize == 1` specialization before dispatching to the backend hook.
pub trait GGLWEProductDefault<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxDftCopy<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    fn gglwe_product_dft_tmp_bytes_default<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos;

    /// Applies one GGLWE product into a DFT accumulator that will contain
    /// `term_count` such products before normalization.
    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Scratch bound of [`gglwe_product_digits_strided_default`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gglwe_product_digits_strided_tmp_bytes_default<BE: Backend>(
    module: &Module<BE>,
    res_size: usize,
    a_cols: usize,
    a_size: usize,
    dsize: usize,
    pmat_rows: usize,
    pmat_cols_in: usize,
    pmat_cols_out: usize,
    pmat_size: usize,
) -> usize
where
    Module<BE>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VmpApplyDftToDftAccumulateTmpBytes,
{
    assert_ne!(dsize, 0);
    let digit_size = a_size.div_ceil(dsize).min(pmat_rows);
    let apply = module.vmp_apply_dft_to_dft_tmp_bytes(res_size, digit_size, pmat_rows, pmat_cols_in, pmat_cols_out, pmat_size);
    let accumulate =
        module.vmp_apply_dft_to_dft_accumulate_tmp_bytes(res_size, digit_size, pmat_rows, pmat_cols_in, pmat_cols_out, pmat_size);
    module.bytes_of_vec_znx_dft(a_cols, digit_size) + apply.max(accumulate)
}

/// Canonical GGLWE product over interleaved gadget digits: digit `di` gathers
/// the source limbs congruent to `dsize - 1 - di` modulo `dsize`. Reference
/// semantics for every backend hook.
#[doc(hidden)]
pub fn gglwe_product_digits_strided_default<BE: Backend>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    dsize: usize,
    product_limbs: usize,
    pmat: &poulpy_hal::layouts::VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxDftBytesOf + VecZnxDftCopy<BE> + VmpApplyDftToDft<BE> + VmpApplyDftToDftAccumulate<BE>,
{
    assert_ne!(dsize, 0);
    let cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();
    for di in 0..dsize {
        let digit_size = ((a_size + di) / dsize).min(dnum);
        let (mut digit, mut digit_scratch) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, digit_size);
        for col in 0..cols {
            module.vec_znx_dft_copy(dsize, dsize - di - 1, &mut digit, col, a, col);
        }
        // Digit-width contract on `GLWEKeyswitchDefault`: `di == 0` overwrites at
        // full width, the accumulating digits above it are narrowed.
        if di == 0 {
            module.vmp_apply_dft_to_dft(res, &digit.to_backend_ref(), pmat, 0, &mut digit_scratch);
        } else {
            let compute_size = gglwe_product_digit_output_size(res.size(), pmat.size(), dsize, di, product_limbs);
            let mut res_view = res.with_size_mut(compute_size);
            module.vmp_apply_dft_to_dft_accumulate(&mut res_view, &digit.to_backend_ref(), pmat, di, &mut digit_scratch);
        }
    }
}

// === Free-function defaults for GLWEKeyswitchDefault ===

use poulpy_hal::{
    api::{
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes, VecZnxNormalize,
        VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{VecZnxBigToBackendRef, VecZnxToBackendRef},
};

use crate::{
    default::operations::GLWENormalizeDefault,
    layouts::{GLWELayout, GLWEToBackendMut},
    oep::GLWEKeyswitchDefault,
};

fn glwe_keyswitch_dft_fill<'r, BE, M, A>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    A: GLWEToBackendRef<BE>,
    M: GLWEKeyswitchInternal<BE> + GGLWEProductDefault<BE> + VecZnxDftApply<BE>,
{
    let a = a.to_backend_ref();
    assert_eq!(a.base2k(), key.base2k());
    let tmp_bytes = module.glwe_keyswitch_internal_tmp_bytes_from_sizes(a.rank().as_usize(), res.size(), a.size(), key);
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEKeyswitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    let mask_cols = a.rank().as_usize();
    let a_size: usize = a.size();
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, mask_cols, a_size);
        for col_i in 0..mask_cols {
            let a_data: &VecZnxBackendRef<'_, BE> = &a.data;
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a_data, col_i + 1);
        }
        let a_dft_ref = a_dft.to_backend_ref();
        module.gglwe_product_dft_default(res, &a_dft_ref, key, 1, &mut scratch_1.borrow());
    });
}

/// Practical limb window used for an immediately normalized GGLWE/VMP product.
///
/// Any lower limb can affect rounding through a sufficiently long carry chain.
/// On exact transform backends the retained window covers the live precision
/// plus the worst-case norm growth of the signed polynomial products and VMP
/// accumulation; approximate backends retain the complete work region.
pub fn gglwe_product_output_size<BE, R, A, K>(res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    gglwe_product_accumulation_output_size::<BE, _, _, _>(res_infos, a_infos, key_infos, 1)
}

/// Number of limbs required when `term_count` GGLWE/VMP products are summed
/// before a single normalization.
///
/// Relative to one product, summing `term_count` values can amplify the tail
/// by that factor. Exact backends account for this in the product-norm window;
/// approximate backends keep the complete work region.
pub fn gglwe_product_accumulation_output_size<BE, R, A, K>(res_infos: &R, a_infos: &A, key_infos: &K, term_count: usize) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    gglwe_product_accumulation_output_size_with_tail::<BE, _, _, _>(res_infos, a_infos, key_infos, term_count, 0)
}

pub(crate) fn gglwe_product_accumulation_output_size_with_tail<BE, R, A, K>(
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    term_count: usize,
    extra_live_limbs: usize,
) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    let product_terms = key_infos
        .n()
        .as_usize()
        .saturating_mul(key_infos.dnum().as_usize())
        .saturating_mul(key_infos.dsize().as_usize())
        .saturating_mul(key_infos.rank_in().as_usize().max(1))
        .saturating_mul(term_count.max(1));
    gadget_product_output_size(GadgetProductOutputSizeParams {
        key_size: key_infos.size(),
        key_base2k: key_infos.base2k(),
        input_k: a_infos.k(),
        output_k: res_infos.k(),
        dsize: key_infos.dsize(),
        k_aux: key_infos.k_aux(),
        dft_is_exact: BE::DFT_IS_EXACT,
        product_terms,
        extra_live_limbs,
    })
}

#[allow(private_bounds)]
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
        + VecZnxIdftNormalizeConsumeTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let output_cols = res_infos.rank().as_usize() + 1;
    let mask_cols = a_infos.rank().as_usize();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res_infos, a_infos, key_infos);
    let a_dft_size = a_infos.k().div_ceil(key_infos.base2k()) as usize;
    let lvl_0: usize = module.bytes_of_vec_znx_dft(output_cols, output_size);
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(output_cols, output_size);
    let consume_tmp: usize = module.vec_znx_idft_normalize_consume_tmp_bytes(output_size, output_size);
    let lvl_1: usize = consume_tmp.max(
        lvl_1_big
            + module
                .vec_znx_idft_apply_tmp_bytes()
                .max(module.vec_znx_big_normalize_tmp_bytes()),
    );
    let lvl_2: usize = if a_infos.base2k() != key_infos.base2k() {
        let small_term_tmp: usize = BE::bytes_of_vec_znx(module.n(), 1, output_size);
        let a_conv_infos: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes_default()
            .max(module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, output_size, a_dft_size, key_infos));
        let lvl_2_2: usize = small_term_tmp
            + consume_tmp.max(
                lvl_1_big
                    + module
                        .vec_znx_idft_apply_tmp_bytes()
                        .max(module.vec_znx_big_normalize_tmp_bytes())
                        .max(module.vec_znx_normalize_tmp_bytes()),
            );
        lvl_2_0 + lvl_2_1.max(lvl_2_2)
    } else {
        lvl_1.max(module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, output_size, a_dft_size, key_infos))
    };

    lvl_0 + lvl_2
}

pub fn glwe_keyswitch_default<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEKeyswitchDefault<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeConsume<BE>
        + VecZnxNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
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

    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, a, key);

    let a_base2k: usize = a.base2k().into();
    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();

    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, output_size);

    let mut scratch = scratch_1;
    if a_base2k != key_base2k {
        scratch.scope(|scratch_phase| {
            let (mut a_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2.borrow());
            glwe_keyswitch_dft_fill(module, &mut res_dft, &a_conv, key, &mut scratch_2);
        });
    } else {
        glwe_keyswitch_dft_fill(module, &mut res_dft, a, key, &mut scratch.borrow());
    }

    let mut res_ref = res.to_backend_mut();
    if a_base2k != key_base2k {
        let (mut res_small, mut scratch_2) = scratch.borrow().take_vec_znx_scratch(module.n(), 1, output_size);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            0,
            0,
            &a.to_backend_ref().data,
            a_base2k,
            0,
            &mut scratch_2.borrow(),
        );
        let res_small_ref = res_small.to_backend_ref();
        for i in 0..cols {
            let addend = (i == 0).then_some((&res_small_ref, 0));
            module.vec_znx_idft_normalize_consume(
                &mut res_ref.data,
                res_base2k,
                i,
                &mut res_dft,
                i,
                key_base2k,
                addend,
                &mut scratch_2.borrow(),
            );
        }
    } else {
        let a_ref = a.to_backend_ref();
        for i in 0..cols {
            let addend = (i == 0).then_some((&a_ref.data, 0));
            module.vec_znx_idft_normalize_consume(
                &mut res_ref.data,
                res_base2k,
                i,
                &mut res_dft,
                i,
                key_base2k,
                addend,
                &mut scratch.borrow(),
            );
        }
    }
}

pub fn glwe_keyswitch_assign_default<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
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
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
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

    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, res, key);

    let res_base2k: usize = res.base2k().as_usize();
    let key_base2k: usize = key.base2k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, output_size);

    let (res_big, mut scratch) = if res_base2k != key_base2k {
        let scratch = scratch_1;
        let (mut res_conv, mut scratch_3) = scratch.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: key.base2k(),
            k: res.k(),
            rank: res.rank(),
        });
        module.glwe_normalize_default(&mut res_conv, res, &mut scratch_3.borrow());

        module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_3);

        let (mut res_big, mut scratch) = scratch_3.take_vec_znx_big_scratch(module, cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        let (mut res_small, mut scratch_2) = scratch.take_vec_znx_scratch(module.n(), 1, output_size);
        let res_ref = GLWEToBackendRef::<BE>::to_backend_ref(res);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            0,
            0,
            &res_ref.data,
            res_base2k,
            0,
            &mut scratch_2.borrow(),
        );
        let res_small_ref = res_small.to_backend_ref();
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_small_ref, 0);
        (res_big, scratch_2)
    } else {
        {
            let mut ks_scratch = scratch_1.borrow();
            module.glwe_keyswitch_internal(&mut res_dft, res, key, &mut ks_scratch);
        }
        let res_ref = GLWEToBackendRef::<BE>::to_backend_ref(res);
        let (mut res_big, mut scratch) = scratch_1.take_vec_znx_big_scratch(module, cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_ref.data, 0);
        (res_big, scratch)
    };
    let res_big_ref = res_big.to_backend_ref();
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            0,
            i,
            &res_big_ref,
            key_base2k,
            i,
            &mut scratch.borrow(),
        );
    }
}
