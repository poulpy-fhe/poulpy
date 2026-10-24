use crate::api::GLWEBytesOf;
use crate::api::GLWENormalize;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxGadgetDecompose,
        VecZnxGadgetDecomposeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftAddTmpBytes,
        VmpApplyDftToDftTmpBytes, VmpExtractSelectedRows, VmpPMatBytesOf,
    },
    layouts::{
        Backend, Module, PrepareHint, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VmpPMatBackendRef, VmpPMatToBackendRef, vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        Base2K, Dnum, GGLWEInfos, GGLWEPreparedBackendRef, GLWEInfos, GLWEToBackendRef, GadgetProductOutputSizeParams, LWEInfos,
        TorusPrecision, digits_aligned, gadget_product_limbs, gadget_product_output_size,
    },
    oep::{GGLWEProductDigitsStridedImpl, gglwe_product_digit_output_size},
};

impl<BE: Backend> GLWEKeyswitchInternal<BE> for Module<BE> where
    Self: GGLWEProductReference<BE>
        + VecZnxDftApply<BE>
        + GLWENormalize<BE>
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes
{
}

/// DFT-domain plumbing shared by the key-switch reference bodies.
///
/// Public because it appears in the `where` clause of the public
/// `glwe_keyswitch*_reference` functions: a backend forwarding to them by hand,
/// rather than through [`crate::impl_glwe_keyswitch_reference_full`], has to be
/// able to name the bound. Blanket-implemented for every `Module<BE>` that has
/// the underlying HAL ops, so there is nothing to implement.
pub trait GLWEKeyswitchInternal<BE: Backend>
where
    Self: GGLWEProductReference<BE>
        + VecZnxDftApply<BE>
        + GLWENormalize<BE>
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes,
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
        let canonical: usize = BE::scratch_aligned(BE::bytes_of_vec_znx(self.n(), mask_cols + 1, a_size));
        canonical
            + glwe_keyswitch_dft_fill_tmp_bytes(self, mask_cols, res_size, a_size, key_infos).max(self.glwe_normalize_tmp_bytes())
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
        assert_no_row_offset("glwe_keyswitch_internal", key);
        glwe_keyswitch_dft_canonical(self, res, a, key, scratch);
    }
}

/// The gadget product of a key switch, `a`'s mask columns decomposed and
/// multiplied by `key`, into `res`: the input once canonical.
fn glwe_keyswitch_dft_canonical<BE, M, A>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEKeyswitchInternal<BE>,
    A: GLWEToBackendRef<BE>,
{
    let (mut a_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(&a.to_backend_ref());
    let a = if a.is_canonical() {
        a.to_backend_ref()
    } else {
        module.glwe_normalize(&mut a_tmp, a, &mut scratch.borrow());
        a_tmp.to_backend_ref()
    };
    glwe_keyswitch_dft_fill(module, res, &&a, key, &mut scratch);
}

/// Scratch of [`glwe_keyswitch_dft_reference`] for an accumulator of `res_size` limbs.
pub fn glwe_keyswitch_dft_tmp_bytes_reference<BE, M, A, K>(module: &M, res_size: usize, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEKeyswitchInternal<BE>,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    operand_degree(module.n(), &[a_infos.n(), key_infos.n()]);
    module.glwe_keyswitch_internal_tmp_bytes_from_sizes(a_infos.rank().as_usize(), res_size, a_infos.size(), key_infos)
}

/// The gadget product of a key switch of `a = (b, a_1, ..., a_rank)` over the
/// `res.size()` limbs of the DFT accumulator `res`: `Σ_i decomp(a_i) · key_i`, the
/// whole switch but for the body `b`.
///
/// The caller finishes the switch with one `vec_znx_idft_normalize` per column,
/// passing the body of `a`, lowered by `key.k_offset()` bits, as the addend of
/// column 0 (see [`glwe_keyswitch_reference`]). `a` is at the key radix.
pub fn glwe_keyswitch_dft_reference<BE, M, A>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEKeyswitchInternal<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    check_keyswitch_dft(module, res, a, key, scratch);
    glwe_keyswitch_dft_canonical(module, res, a, key, scratch);
}

fn check_keyswitch_dft<BE, M, A>(
    module: &M,
    res: &VecZnxDftBackendMut<'_, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEKeyswitchInternal<BE>,
    A: GLWEInfos,
{
    assert_eq!(a.rank(), key.rank_in(), "glwe_keyswitch_dft: a.rank() != key.rank_in()");
    assert_eq!(
        res.cols(),
        (key.rank_out() + 1).as_usize(),
        "glwe_keyswitch_dft: res.cols() != key.rank_out() + 1"
    );
    assert_eq!(a.base2k(), key.base2k(), "glwe_keyswitch_dft: a is not at the key radix");
    operand_degree(module.n(), &[res.n().into(), a.n(), key.n()]);
    assert_key_covers("glwe_keyswitch_dft", a.k(), key);
    let tmp_bytes = glwe_keyswitch_dft_tmp_bytes_reference(module, res.size(), a, key);
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_dft_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
}

fn glwe_keyswitch_dft_fill_tmp_bytes<BE, M, K>(
    module: &M,
    mask_cols: usize,
    res_size: usize,
    a_size: usize,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: GLWEKeyswitchInternal<BE>,
    K: GGLWEInfos,
{
    // Sized for an input of `a_size` full limbs, an upper bound of its precision.
    let (base2k, k_digit, dnum) = (key_infos.base2k(), key_infos.k_digit(), key_infos.dnum());
    let a_k = TorusPrecision((a_size * base2k.as_usize()) as u32);
    let operand_size: usize = gadget_input_size(base2k, k_digit, dnum, a_k, a_size);
    let lvl_0: usize = module.bytes_of_vec_znx_dft(module.n(), mask_cols, operand_size);
    let lvl_1: usize = module
        .gglwe_product_dft_tmp_bytes_reference(res_size, operand_size, key_infos)
        .max(gadget_dft_apply_tmp_bytes::<BE, _>(
            module, base2k, k_digit, dnum, a_k, a_size,
        ));
    lvl_0 + lvl_1
}

impl<BE: Backend> GGLWEProductReference<BE> for Module<BE>
where
    BE: GGLWEProductDigitsStridedImpl,
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAddTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftCopy<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    fn gglwe_product_dft_tmp_bytes_reference<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
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
            // The gathered matrix is allocated with the key's hint (see the take
            // below); the layout infos do not carry it, so size for either
            // representation.
            let dense = self
                .bytes_of_vmp_pmat(self.n(), dnum, cols_in, cols_out, key_size, PrepareHint::Reuse)
                .max(self.bytes_of_vmp_pmat(self.n(), dnum, cols_in, cols_out, key_size, PrepareHint::OneShot));
            dense + product
        }
    }

    fn gglwe_product_dft_reference<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let a_size = a.size();
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_reference(res.size(), a_size, key),
            "scratch.available(): {} < GGLWEProductReference::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_reference(res.size(), a_size, key)
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
            let (mut dense, mut scratch_1) =
                scratch_phase.take_vmp_pmat_scratch(res.n(), rows, cols_in, cols_out, key_size, key.data.hint());
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
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: VmpApplyDftToDft<BE>,
{
    // One limb per digit is a plain VMP, not a strided gather; the hook below
    // is only ever entered with `dsize >= 2`.
    let dsize: usize = key.dsize().into();
    if dsize == 1 {
        module.vmp_apply_dft_to_dft(res, a, pmat, 0, scratch);
    } else {
        let product_limbs = gglwe_product_limbs(key, term_count);
        BE::gglwe_product_digits_strided(module, res, a, dsize, product_limbs, pmat, scratch);
    }
}

/// Spill width `product_limbs` of [`GGLWEProductDigitsStridedImpl::gglwe_product_digits_strided`] for products against
/// `key`, accumulated `term_count` times before normalization.
pub fn gglwe_product_limbs<K: GGLWEInfos>(key: &K, term_count: usize) -> usize {
    let product_terms = key
        .n()
        .as_usize()
        .saturating_mul(key.dnum().as_usize())
        .saturating_mul(key.dsize().as_usize())
        .saturating_mul(key.rank_in().as_usize().max(1))
        .saturating_mul(term_count.max(1));
    gadget_product_limbs(key.base2k(), product_terms)
}

/// Default DFT-domain gadget product used by key-switching and external products.
///
/// Public so backend forwarders can name the bound. It centralizes the
/// `dsize == 1` specialization before dispatching to the backend hook.
pub trait GGLWEProductReference<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAddTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftCopy<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    fn gglwe_product_dft_tmp_bytes_reference<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos;

    /// Applies one GGLWE product into a DFT accumulator that will contain
    /// `term_count` such products before normalization.
    fn gglwe_product_dft_reference<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// The HAL operations the default of `GGLWEProductDigitsStridedImpl::gglwe_product_fused` composes.
pub trait GGLWEProductFusedReference<BE: Backend>:
    GGLWEProductReference<BE>
    + poulpy_hal::api::VecZnxDftAddAssign<BE>
    + poulpy_hal::api::VecZnxDftAutomorphism<BE>
    + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes
    + poulpy_hal::api::VecZnxIdftNormalize<BE>
    + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
{
}

impl<BE: Backend, M> GGLWEProductFusedReference<BE> for M where
    M: GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxDftAddAssign<BE>
        + poulpy_hal::api::VecZnxDftAutomorphism<BE>
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
{
}

/// The gadget product with what follows it in a composite operation, through the backend's
/// `GGLWEProductDigitsStridedImpl::gglwe_product_fused`.
pub trait GGLWEProductFused<BE: Backend> {
    fn gglwe_product_fused_tmp_bytes<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_fused(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
        automorphism: Option<i64>,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn gglwe_product_idft_normalize_tmp_bytes<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    /// The product followed at once by its inverse transform and normalization, through the
    /// backend's `GGLWEProductDigitsStridedImpl::gglwe_product_idft_normalize`.
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_idft_normalize(
        &self,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, BE>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE> GGLWEProductFused<BE> for Module<BE>
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: GGLWEProductFusedReference<BE>,
{
    fn gglwe_product_fused_tmp_bytes<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        BE::gglwe_product_fused_tmp_bytes(self, res_size, a_size, key_infos)
    }

    fn gglwe_product_idft_normalize_tmp_bytes<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        BE::gglwe_product_idft_normalize_tmp_bytes(self, res_size, product_size, a_size, key_infos)
    }

    fn gglwe_product_fused(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
        automorphism: Option<i64>,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::gglwe_product_fused(
            self,
            res,
            product_size,
            a,
            key,
            term_count,
            addends,
            automorphism,
            accumulate,
            scratch,
        )
    }

    fn gglwe_product_idft_normalize(
        &self,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, BE>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::gglwe_product_idft_normalize(
            self, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, scratch,
        )
    }
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gglwe_product_idft_normalize_reference<BE, M>(
    module: &M,
    res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_dft: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    term_count: usize,
    addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
    body: Option<(&VecZnxBackendRef<'_, BE>, usize)>,
    a_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    M: GGLWEProductFusedReference<BE> + GGLWEProductFused<BE>,
{
    // The product into `res_dft`, then the addends over the limbs both have.
    module.gglwe_product_dft_reference(res_dft, a, key, term_count, &mut scratch.borrow());
    for (col, addend) in addends.iter().enumerate() {
        if let Some((addend, addend_col)) = addend {
            module.vec_znx_dft_add_assign(res_dft, col, addend, *addend_col);
        }
    }
    for col in 0..res_dft.cols() {
        module.vec_znx_idft_normalize(
            res,
            res_base2k,
            res_k,
            res_offset,
            col,
            res_dft,
            col,
            a_base2k,
            if col == 0 { body } else { None },
            &mut scratch.borrow(),
        );
    }
}

#[doc(hidden)]
pub fn gglwe_product_idft_normalize_tmp_bytes_reference<BE, M, K>(
    module: &M,
    res_size: usize,
    product_size: usize,
    a_size: usize,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: GGLWEProductFusedReference<BE>,
    K: GGLWEInfos,
{
    module
        .gglwe_product_dft_tmp_bytes_reference(product_size, a_size, key_infos)
        .max(module.vec_znx_idft_normalize_tmp_bytes(res_size, product_size))
}

#[doc(hidden)]
pub fn gglwe_product_fused_tmp_bytes_reference<BE, M, K>(module: &M, res_size: usize, a_size: usize, key_infos: &K) -> usize
where
    BE: Backend,
    M: GGLWEProductFusedReference<BE>,
    K: GGLWEInfos,
{
    let cols: usize = (key_infos.rank_out() + 1).into();
    BE::scratch_aligned(module.bytes_of_vec_znx_dft(module.n(), cols, res_size))
        + module
            .gglwe_product_dft_tmp_bytes_reference(res_size, a_size, key_infos)
            .max(module.vec_znx_dft_automorphism_add_with_plan_tmp_bytes(res_size, res_size))
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gglwe_product_fused_reference<BE, M>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    product_size: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    term_count: usize,
    addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
    automorphism: Option<i64>,
    accumulate: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGLWEProductFusedReference<BE>,
{
    use poulpy_hal::layouts::VecZnxDftToBackendRef;
    let cols = res.cols();
    assert!(addends.len() <= cols, "more addends than output columns");
    assert!(product_size <= res.size(), "product wider than its destination");
    let (mut product, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(res.n(), cols, product_size);
    module.gglwe_product_dft_reference(&mut product, a, key, term_count, &mut scratch_1.borrow());
    for (col, addend) in addends.iter().enumerate() {
        if let Some((addend, addend_col)) = addend {
            module.vec_znx_dft_add_assign(&mut product, col, addend, *addend_col);
        }
    }
    let product = product.to_backend_ref();
    match automorphism {
        Some(p) => {
            let plan = module.vec_znx_dft_automorphism_plan(res.n(), p);
            for col in 0..cols {
                if accumulate {
                    module.vec_znx_dft_automorphism_add_with_plan(&plan, res, col, &product, col, &mut scratch_1);
                } else {
                    module.vec_znx_dft_automorphism_with_plan(&plan, res, col, &product, col);
                }
            }
        }
        None => {
            for col in 0..cols {
                if accumulate {
                    module.vec_znx_dft_add_assign(res, col, &product, col);
                } else {
                    module.vec_znx_dft_copy(1, 0, res, col, &product, col);
                }
            }
        }
    }
}

/// Scratch bound of [`gglwe_product_digits_strided_reference`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gglwe_product_digits_strided_tmp_bytes_reference<BE: Backend>(
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
    Module<BE>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VmpApplyDftToDftAddTmpBytes,
{
    assert_ne!(dsize, 0);
    let digit_size = a_size.div_ceil(dsize).min(pmat_rows);
    let apply = module.vmp_apply_dft_to_dft_tmp_bytes(res_size, digit_size, pmat_rows, pmat_cols_in, pmat_cols_out, pmat_size);
    let accumulate =
        module.vmp_apply_dft_to_dft_add_tmp_bytes(res_size, digit_size, pmat_rows, pmat_cols_in, pmat_cols_out, pmat_size);
    module.bytes_of_vec_znx_dft(module.n(), a_cols, digit_size) + apply.max(accumulate)
}

/// Canonical GGLWE product over interleaved gadget digits: digit `di` gathers
/// the source limbs congruent to `dsize - 1 - di` modulo `dsize`. Reference
/// semantics for every backend hook.
///
/// This reference body requires [`Backend::DFT_LIMBS_CONTIGUOUS`] because later
/// digits accumulate into partial DFT views. Instantiating it without that
/// capability fails at compile time. Other layouts must implement
/// [`GGLWEProductDigitsStridedImpl`] instead of forwarding to this body.
#[doc(hidden)]
pub fn gglwe_product_digits_strided_reference<BE: Backend>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    dsize: usize,
    product_limbs: usize,
    pmat: &poulpy_hal::layouts::VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxDftBytesOf + VecZnxDftCopy<BE> + VmpApplyDftToDft<BE> + VmpApplyDftToDftAdd<BE>,
{
    const {
        assert!(
            BE::DFT_LIMBS_CONTIGUOUS,
            "the reference interleaved-digit product requires contiguous DFT limbs; implement GGLWEProductDigitsStridedImpl for other layouts"
        );
    }
    assert_ne!(dsize, 0);
    let cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();
    for di in 0..dsize {
        let digit_size = ((a_size + di) / dsize).min(dnum);
        let (mut digit, mut digit_scratch) = scratch.borrow().take_vec_znx_dft_scratch(res.n(), cols, digit_size);
        for col in 0..cols {
            module.vec_znx_dft_copy(dsize, dsize - di - 1, &mut digit, col, a, col);
        }
        // Digit-width contract on `GLWEKeyswitchReference`: `di == 0` overwrites at
        // full width, the accumulating digits above it are narrowed.
        if di == 0 {
            module.vmp_apply_dft_to_dft(res, &digit.to_backend_ref(), pmat, 0, &mut digit_scratch);
        } else {
            let compute_size = gglwe_product_digit_output_size(res.size(), pmat.size(), dsize, di, product_limbs);
            let mut res_view = res.with_size_mut(compute_size);
            module.vmp_apply_dft_to_dft_add(&mut res_view, &digit.to_backend_ref(), pmat, di, &mut digit_scratch);
        }
    }
}

// === Free-function defaults for GLWEKeyswitchReference ===

use poulpy_hal::{
    api::{
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes, VecZnxNormalize, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{VecZnxBigToBackendRef, VecZnxToBackendRef},
};

use crate::layouts::operand_degree;
use crate::{
    layouts::{GLWELayout, GLWEToBackendMut},
    oep::GLWEKeyswitchReference,
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
    M: GLWEKeyswitchInternal<BE>,
{
    let a = a.to_backend_ref();
    assert_eq!(a.base2k(), key.base2k());
    let tmp_bytes = glwe_keyswitch_dft_fill_tmp_bytes(module, a.rank().as_usize(), res.size(), a.size(), key);
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEKeyswitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) =
            scratch_phase.take_vec_znx_dft_scratch(res.n(), a.rank().as_usize(), glwe_keyswitch_operand_size(&a, key));
        glwe_keyswitch_operand(module, &mut a_dft, &&a, key, &mut scratch_1);
        let a_dft_ref = a_dft.to_backend_ref();
        module.gglwe_product_dft_reference(res, &a_dft_ref, key, 1, &mut scratch_1.borrow());
    });
}

/// Limbs of the gadget operand [`glwe_keyswitch_operand`] lays the mask of `a` out on.
fn glwe_keyswitch_operand_size<A: GLWEInfos, K: GGLWEInfos>(a: &A, key: &K) -> usize {
    gadget_input_size(key.base2k(), key.k_digit(), key.dnum(), a.k(), a.size())
}

/// The mask columns of `a`, laid out on the digit grid of `key` and transformed:
/// the operand of the gadget product of a key switch, into `a_dft` of
/// [`glwe_keyswitch_operand_size`] limbs.
fn glwe_keyswitch_operand<BE, M, A>(
    module: &M,
    a_dft: &mut VecZnxDftBackendMut<'_, BE>,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    A: GLWEToBackendRef<BE>,
    M: GLWEKeyswitchInternal<BE>,
{
    let a = a.to_backend_ref();
    let (base2k, k_digit, dnum) = (key.base2k(), key.k_digit(), key.dnum());
    for col_i in 0..a.rank().as_usize() {
        let a_data: &VecZnxBackendRef<'_, BE> = &a.data;
        gadget_dft_apply(
            module,
            a_dft,
            col_i,
            a_data,
            col_i + 1,
            base2k,
            k_digit,
            dnum,
            a.k(),
            &mut scratch.borrow(),
        );
    }
}

/// Practical limb window used for an immediately normalized GGLWE/VMP product.
///
/// Any lower limb can affect rounding through a sufficiently long carry chain.
/// The retained window covers the live precision plus the worst-case norm
/// growth of the signed polynomial products and VMP accumulation, on every
/// backend; see `gadget_product_output_size`.
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
/// by that factor, which the product-norm window accounts for.
pub fn gglwe_product_accumulation_output_size<BE, R, A, K>(res_infos: &R, a_infos: &A, key_infos: &K, term_count: usize) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    gglwe_product_accumulation_output_size_with_tail(res_infos, a_infos, key_infos, term_count, 0)
}

pub(crate) fn gglwe_product_accumulation_output_size_with_tail<R, A, K>(
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    term_count: usize,
    extra_live_limbs: usize,
) -> usize
where
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
        k_digit: key_infos.k_digit(),
        k_offset: key_infos.k_offset(),
        k_aux: key_infos.k_aux(),
        product_terms,
        extra_live_limbs,
    })
}

#[allow(private_bounds)]
pub fn glwe_keyswitch_tmp_bytes_reference<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + GGLWEProductFused<BE>,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let n = operand_degree(module.n(), &[res_infos.n(), a_infos.n(), key_infos.n()]);

    let output_cols = res_infos.rank().as_usize() + 1;
    let mask_cols = a_infos.rank().as_usize();
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res_infos, a_infos, key_infos);
    let a_dft_size = a_infos.k().div_ceil(key_infos.base2k()) as usize;
    let lvl_0: usize = module.bytes_of_vec_znx_dft(n, output_cols, output_size);
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(n, output_cols, output_size);
    let consume_tmp: usize = module.vec_znx_idft_normalize_tmp_bytes(output_size, output_size);
    let lvl_1: usize = consume_tmp.max(
        lvl_1_big
            + module
                .vec_znx_idft_apply_tmp_bytes()
                .max(module.vec_znx_big_normalize_tmp_bytes()),
    );
    let lvl_2: usize = if a_infos.base2k() != key_infos.base2k() || key_infos.k_offset().0 > 0 {
        let small_term_tmp: usize = BE::bytes_of_vec_znx(n, 1, output_size);
        let a_conv_infos: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes()
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
        // The gadget operand is held across the product and the normalization.
        let operand_size = gadget_input_size(
            key_infos.base2k(),
            key_infos.k_digit(),
            key_infos.dnum(),
            a_infos.k(),
            a_dft_size,
        );
        lvl_1
            .max(module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, output_size, a_dft_size, key_infos))
            .max(
                BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, mask_cols, operand_size))
                    + module.gglwe_product_idft_normalize_tmp_bytes(res_infos.size(), output_size, operand_size, key_infos),
            )
    };

    BE::scratch_aligned(module.glwe_bytes_of_from_infos(a_infos)) + (lvl_0 + lvl_2).max(module.glwe_normalize_tmp_bytes())
}

pub fn glwe_keyswitch_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEKeyswitchReference<BE>
        + GGLWEProductFused<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftNormalize<BE>
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

    operand_degree(module.n(), &[res.n(), a.n(), key.n()]);
    assert_key_covers("glwe_keyswitch", a.k(), key);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes_reference(res, a, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes_reference(res, a, key)
    );

    let k_offset: usize = key.k_offset().as_usize();

    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, a, key);

    let a_base2k: usize = a.base2k().into();
    let key_base2k: usize = key.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let res_k = res.k().as_usize();
    let cols: usize = (res.rank() + 1).into();

    let (mut res_dft, scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(res.n().as_usize(), cols, output_size);

    // The body goes back at its own place, `k_offset` bits below the input: it
    // is rebuilt at the key radix, lowered, when the key has an offset.
    if a_base2k != key_base2k || k_offset > 0 {
        let (mut a_conv, mut scratch_2) = scratch_1.take_glwe_scratch(&GLWELayout {
            n: a.n(),
            base2k: key.base2k(),
            k: a.k(),
            rank: a.rank(),
        });
        module.glwe_normalize(&mut a_conv, a, &mut scratch_2.borrow());
        glwe_keyswitch_dft_fill(module, &mut res_dft, &a_conv, key, &mut scratch_2.borrow());

        res.set_canonical(true);
        let mut res_ref = res.to_backend_mut();
        let (mut res_small, mut scratch_3) = scratch_2.take_vec_znx_scratch(a.n().as_usize(), 1, output_size);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            output_size * key_base2k,
            -(k_offset as i64),
            0,
            &a_conv.to_backend_ref().data,
            key_base2k,
            0,
            &mut scratch_3.borrow(),
        );
        let res_small_ref = res_small.to_backend_ref();
        for i in 0..cols {
            let addend = (i == 0).then_some((&res_small_ref, 0));
            module.vec_znx_idft_normalize(
                &mut res_ref.data,
                res_base2k,
                res_k,
                0,
                i,
                &mut res_dft,
                i,
                key_base2k,
                addend,
                &mut scratch_3.borrow(),
            );
        }
    } else {
        let (mut a_tmp, mut scratch_2) = scratch_1.take_glwe_scratch(a);
        let a = if a.is_canonical() {
            a.to_backend_ref()
        } else {
            module.glwe_normalize(&mut a_tmp, a, &mut scratch_2.borrow());
            a_tmp.to_backend_ref()
        };
        // The gadget operand, then the product, its inverse and the normalization as one step.
        let (mut a_dft, mut scratch_3) =
            scratch_2.take_vec_znx_dft_scratch(a.n().as_usize(), a.rank().as_usize(), glwe_keyswitch_operand_size(&a, key));
        glwe_keyswitch_operand(module, &mut a_dft, &&a, key, &mut scratch_3);

        res.set_canonical(true);
        let mut res_ref = res.to_backend_mut();
        // The body adds back only the limbs the input's precision covers.
        let a_body = precision_limbs::<BE>(&a.data, a.base2k(), a.k());
        module.gglwe_product_idft_normalize(
            &mut res_ref.data,
            res_base2k,
            res_k,
            0,
            &mut res_dft,
            &a_dft.to_backend_ref(),
            key,
            1,
            &[],
            a_body.as_ref().map(|body| (body, 0)),
            key_base2k,
            &mut scratch_3.borrow(),
        );
    }
}

/// Panics on a key with a row offset: `op` adds its input back at the input's
/// own place, which only an offset-free key's product matches.
pub fn assert_no_row_offset<K: GGLWEInfos>(op: &str, key: &K) {
    assert_eq!(
        key.k_offset().0,
        0,
        "{op}: keys with a row offset are only taken by glwe_keyswitch and glwe_keyswitch_dft"
    );
}

/// Panics unless `key` decomposes an input of precision `k` exactly: `k` is at
/// most its [`max_input_k`](crate::layouts::GGLWELayout::max_input_k). A wider
/// input is narrowed by the caller first.
pub fn assert_key_covers<K: GGLWEInfos>(op: &str, k: TorusPrecision, key: &K) {
    assert!(
        k <= key.gglwe_layout().max_input_k(),
        "{op}: the input is wider than the key's max_input_k"
    );
}

pub fn glwe_keyswitch_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEKeyswitchReference<BE>
        + ModuleN
        + GLWEKeyswitchInternal<BE>
        + GLWENormalize<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssign<BE>,
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

    operand_degree(module.n(), &[res.n(), key.n()]);
    assert_key_covers("glwe_keyswitch_assign", res.k(), key);

    assert!(
        scratch.available() >= module.glwe_keyswitch_tmp_bytes_reference(res, res, key),
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.glwe_keyswitch_tmp_bytes_reference(res, res, key)
    );

    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, res, key);

    let res_base2k: usize = res.base2k().as_usize();
    let res_k = res.k().as_usize();
    let key_base2k: usize = key.base2k().as_usize();
    if !res.is_canonical() && res_base2k == key_base2k {
        module.glwe_normalize_assign(res, scratch);
    }
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, mut scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(res.n().as_usize(), cols, output_size);

    let (res_big, mut scratch) = if res_base2k != key_base2k {
        let (mut res_conv, mut scratch_3) = scratch_1.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: key.base2k(),
            k: res.k(),
            rank: res.rank(),
        });
        module.glwe_normalize(&mut res_conv, res, &mut scratch_3.borrow());

        module.glwe_keyswitch_internal(&mut res_dft, &res_conv, key, &mut scratch_3);

        let (mut res_big, mut scratch) = scratch_3.take_vec_znx_big_scratch(res.n().as_usize(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        let (mut res_small, mut scratch_2) = scratch.take_vec_znx_scratch(res.n().as_usize(), 1, output_size);
        module.vec_znx_normalize(
            &mut res_small,
            key_base2k,
            output_size * key_base2k,
            0,
            0,
            &res_conv.to_backend_ref().data,
            key_base2k,
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
        let (mut res_big, mut scratch) = scratch_1.take_vec_znx_big_scratch(res.n().as_usize(), cols, output_size);
        let res_dft_ref = res_dft.to_backend_ref();
        for i in 0..cols {
            module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch);
        }
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_ref.data, 0);
        (res_big, scratch)
    };
    let res_big_ref = res_big.to_backend_ref();
    res.set_canonical(true);
    let mut res_ref = res.to_backend_mut();
    for i in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            res_k,
            0,
            i,
            &res_big_ref,
            key_base2k,
            i,
            &mut scratch.borrow(),
        );
    }
}

// === Gadget operand: the input limbs, or digits laid out for the interleaved product ===

/// Limbs of the DFT operand a gadget product with digits of `k_digit` bits
/// reads, for an input of precision `a_k` stored on `a_size` limbs.
///
/// Whole-limb digits read the input limbs. Other widths lay the input out as
/// `ceil(k_digit / base2k)` sub-limbs per digit, for the digits it reaches and
/// at most `dnum`.
pub fn gadget_input_size(base2k: Base2K, k_digit: TorusPrecision, dnum: Dnum, a_k: TorusPrecision, a_size: usize) -> usize {
    if digits_aligned(base2k, k_digit) {
        a_size
    } else {
        let digits: usize = a_k.as_usize().div_ceil(k_digit.as_usize()).min(dnum.as_usize());
        digits * k_digit.as_usize().div_ceil(base2k.as_usize())
    }
}

/// `a` restricted to the limbs that precision `a_k` covers at radix `base2k`,
/// or `None` when it covers none, as for an input of precision 0.
///
/// A buffer may hold more limbs than its precision, and operations sized from
/// that precision read only these: the digit layout rounds at the gadget
/// precision, which can reach past `a_k`, and the body of a key switch adds
/// the input back.
pub fn precision_limbs<'a, BE: Backend>(
    a: &'a VecZnxBackendRef<'_, BE>,
    base2k: Base2K,
    a_k: TorusPrecision,
) -> Option<VecZnxBackendRef<'a, BE>> {
    let live: usize = a_k.as_usize().div_ceil(base2k.as_usize()).min(a.size());
    (live > 0).then(|| vec_znx_backend_ref_from_ref::<BE>(a).window_limbs(0, 1, live))
}

/// Leading limbs of the gadget operand that are known to be zero for an input
/// of magnitude below `2^-(zero_bits + 1)`, so a product may skip them.
///
/// Whole-limb digits read the canonical input limbs, the top
/// `zero_bits / base2k` of which are zero. Other widths read the digits the
/// input decomposes into. The gadget rounding can carry one unit into the
/// digit that ends at bit `zero_bits`, so only the top digits ending at least
/// one bit above it are zero, `(zero_bits - 1) / k_digit` of them, each
/// spanning `ceil(k_digit / base2k)` limbs.
pub fn gadget_known_zero_prefix(base2k: Base2K, k_digit: TorusPrecision, dnum: Dnum, zero_bits: usize) -> usize {
    if digits_aligned(base2k, k_digit) {
        zero_bits / base2k.as_usize()
    } else {
        let digits: usize = (zero_bits.saturating_sub(1) / k_digit.as_usize()).min(dnum.as_usize());
        digits * k_digit.as_usize().div_ceil(base2k.as_usize())
    }
}

/// Scratch of [`gadget_dft_apply`]: none for whole-limb digits.
pub fn gadget_dft_apply_tmp_bytes<BE, M>(
    module: &M,
    base2k: Base2K,
    k_digit: TorusPrecision,
    dnum: Dnum,
    a_k: TorusPrecision,
    a_size: usize,
) -> usize
where
    BE: Backend,
    M: ModuleN + VecZnxGadgetDecomposeTmpBytes,
{
    if digits_aligned(base2k, k_digit) {
        0
    } else {
        let size: usize = gadget_input_size(base2k, k_digit, dnum, a_k, a_size);
        BE::scratch_aligned(BE::bytes_of_vec_znx(module.n(), 1, size)) + module.vec_znx_gadget_decompose_tmp_bytes()
    }
}

/// DFT of `a[a_col]`, at radix `base2k` and precision `a_k`, into `res[res_col]`
/// as the operand of a gadget product with digits of `k_digit` bits; `res`
/// holds at least [`gadget_input_size`] limbs.
///
/// Whole-limb digits transform the input as is, exactly as before. Other
/// widths first lay the limbs `a_k` covers ([`precision_limbs`]) out on
/// the digit grid with `vec_znx_gadget_decompose`, rounded at the gadget
/// precision.
#[allow(clippy::too_many_arguments)]
pub fn gadget_dft_apply<BE, M>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    base2k: Base2K,
    k_digit: TorusPrecision,
    dnum: Dnum,
    a_k: TorusPrecision,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: VecZnxDftApply<BE> + VecZnxGadgetDecompose<BE>,
{
    if digits_aligned(base2k, k_digit) {
        module.vec_znx_dft_apply(1, 0, res, res_col, a, a_col);
        return;
    }
    let size: usize = gadget_input_size(base2k, k_digit, dnum, a_k, a.size());
    assert!(
        res.size() >= size,
        "gadget operand needs {size} limbs, res has {}",
        res.size()
    );
    // An input of precision 0 has no digit, and its operand no limb.
    let Some(a_live) = precision_limbs::<BE>(a, base2k, a_k).filter(|_| size > 0) else {
        return;
    };
    let (mut digits, mut scratch_1) = scratch.borrow().take_vec_znx_scratch(a.n(), 1, size);
    module.vec_znx_gadget_decompose(
        &mut digits,
        0,
        &a_live,
        a_col,
        base2k.as_usize(),
        k_digit.as_usize(),
        &mut scratch_1.borrow(),
    );
    module.vec_znx_dft_apply(1, 0, res, res_col, &digits.to_backend_ref(), 0);
}
