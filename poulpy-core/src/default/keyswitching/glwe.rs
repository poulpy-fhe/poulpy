use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes, VmpApplyDftToDftTmpBytes, VmpExtractSelectedRows,
        VmpPMatBytesOf,
    },
    layouts::{
        Backend, DataView, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VmpPMatToBackendRef,
    },
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        Dsize, GGLWEActiveUse, GGLWEBind, GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEUse, GLWEInfos,
        GLWEToBackendRef, GadgetProductOutputSizeParams, LWEInfos, TorusPrecision, gadget_product_limbs,
        gadget_product_output_size, prepared::GGLWEPreparedToBackendRef, resolve_gglwe_key_use,
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
    /// `use_` is the key bound; zero precision needs no product at all.
    fn glwe_keyswitch_internal_tmp_bytes_from_sizes(
        &self,
        mask_cols: usize,
        res_size: usize,
        a_size: usize,
        use_: &GGLWEUse,
    ) -> usize {
        let lvl_0: usize = self.bytes_of_vec_znx_dft(mask_cols, a_size);
        let lvl_1: usize = match use_ {
            GGLWEUse::Empty => 0,
            GGLWEUse::Active(active) => self.gglwe_product_dft_tmp_bytes_default(res_size, a_size, active),
        };
        lvl_0 + lvl_1
    }

    /// Binds `key_infos` at `a_infos.k()`, the operation's exact input precision.
    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let use_: GGLWEUse = bound_for(key_infos, a_infos.k());
        self.glwe_keyswitch_internal_tmp_bytes_from_sizes(a_infos.rank().as_usize(), res_infos.size(), a_infos.size(), &use_)
    }

    /// The key is bound here, before the conversion to a concrete prepared
    /// reference drops the requested `dsize`.
    fn glwe_keyswitch_internal<'r, A, K>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        glwe_keyswitch_dft_fill(self, res, a, key, scratch);
    }
}

/// The layout every size is computed from: the key as it is bound for the
/// operation's input precision, never the physical metadata a `with_dsize`
/// wrapper still forwards.
pub fn bound_layout<K: GGLWEInfos>(key: &K, input_k: TorusPrecision) -> GGLWELayout {
    match bound_for(key, input_k) {
        GGLWEUse::Empty => key.gglwe_layout(),
        GGLWEUse::Active(active) => active.logical_layout,
    }
}

/// Binds a key for `input_k`, or fails loudly: the seam never falls back to the
/// physical decomposition.
///
/// Public because every operation that reaches a GGLWE product binds through
/// it, including the ones outside this crate.
pub fn bound_for<K: GGLWEInfos>(key_infos: &K, input_k: TorusPrecision) -> GGLWEUse {
    match key_infos.bind_for(input_k) {
        Ok(use_) => use_,
        Err(e) => panic!("{e}"),
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
        + VecZnxDftZero<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    fn gglwe_product_dft_tmp_bytes_default(&self, res_size: usize, a_size: usize, use_: &GGLWEActiveUse) -> usize {
        let logical = &use_.logical_layout;
        let dsize: usize = logical.dsize().into();
        let dnum: usize = logical.dnum().into();
        let cols_in: usize = logical.rank_in().into();
        let cols_out: usize = (logical.rank_out() + 1).into();
        let key_size: usize = use_.logical_work_size;
        let product: usize = if dsize == 1 {
            self.vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, dnum, cols_in, cols_out, key_size)
        } else {
            BE::gglwe_product_digits_strided_tmp_bytes(self, res_size, cols_in, a_size, dsize, dnum, cols_in, cols_out, key_size)
        };
        if use_.physical_row_step.get() == 1 {
            // A contiguous bound is read in place: the dense kernels already
            // stop at `min(rows, a.size())` and `min(size, res.size())`.
            return product;
        }
        // ponytail: a strided bound is gathered dense first, one extra pass over
        // the selected material. A backend addressing the row map in its own
        // kernel overrides the product and drops this.
        self.bytes_of_vmp_pmat(dnum, cols_in, cols_out, key_size) + product
    }

    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        use_: &GGLWEActiveUse,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let a_size: usize = a.size();
        assert_eq!(
            a_size,
            use_.input_size(),
            "the product input must carry ceil(input_k / base2k) limbs for input_k={}",
            use_.input_k
        );
        assert_prepared_matches_bound::<BE>(key, use_);
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, use_),
            "scratch.available(): {} < GGLWEProductDefault::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, use_)
        );

        if use_.physical_row_step.get() == 1 {
            gglwe_product_dft_dense(self, res, a, key, use_, term_count, scratch);
            return;
        }

        let (rows, cols_in, cols_out) = (
            use_.logical_layout.dnum().as_usize(),
            use_.logical_layout.rank_in().as_usize(),
            (use_.logical_layout.rank_out() + 1).as_usize(),
        );
        scratch.scope(|scratch_phase| {
            let (mut selected, mut scratch_1) =
                scratch_phase.take_vmp_pmat_scratch(self, rows, cols_in, cols_out, use_.logical_work_size);
            self.vmp_extract_selected_rows(
                &mut selected,
                &key.data,
                use_.first_physical_row,
                use_.physical_row_step.get(),
            );
            let gathered: GGLWEPreparedBackendRef<'_, BE> = GGLWEPrepared {
                data: selected.to_backend_ref(),
                k_aux: use_.logical_layout.k_aux(),
                base2k: use_.logical_layout.base2k(),
                dsize: use_.logical_layout.dsize(),
            };
            gglwe_product_dft_dense(self, res, a, &gathered, use_, term_count, &mut scratch_1.borrow());
        });
    }
}

/// Rejects a prepared key that does not match the physical shape its bound was
/// resolved from, before any of it reaches a backend.
///
/// A key reconstructed from storage carries its metadata separately from its
/// buffer, so a mismatch is a restore bug rather than a caller bug, and it would
/// otherwise surface as an out-of-range read inside a kernel.
fn assert_prepared_matches_bound<BE: Backend>(key: &GGLWEPreparedBackendRef<'_, BE>, use_: &GGLWEActiveUse) {
    let logical: &GGLWELayout = &use_.logical_layout;
    let data = &key.data;
    let (rows, cols_in, cols_out, size) = (data.rows(), data.cols_in(), data.cols_out(), data.size());

    assert_eq!(data.n(), logical.n().as_usize(), "prepared degree does not match the bound");
    assert_eq!(rows, use_.physical_rows, "prepared rows do not match the bound");
    assert_eq!(
        cols_in,
        logical.rank_in().as_usize(),
        "prepared input columns do not match the bound"
    );
    assert_eq!(
        cols_out,
        (logical.rank_out() + 1).as_usize(),
        "prepared output columns do not match the bound"
    );
    // A bound is always resolved from a complete key, never from a projection.
    assert_eq!(size, use_.physical_size, "prepared limb pitch does not match the bound");
    assert!(
        use_.logical_work_size <= size,
        "logical work size {} exceeds the stored pitch {size}",
        use_.logical_work_size
    );

    // The last selected row, computed without wrapping.
    let last_row: Option<usize> = logical
        .dnum()
        .as_usize()
        .checked_sub(1)
        .and_then(|i| i.checked_mul(use_.physical_row_step.get()))
        .and_then(|o| o.checked_add(use_.first_physical_row));
    assert!(
        last_row.is_some_and(|last| last < rows),
        "selected rows {}..={last_row:?} step {} exceed the stored {rows}",
        use_.first_physical_row,
        use_.physical_row_step
    );

    assert!(
        BE::len_bytes_ref(DataView::data(data)) >= BE::bytes_of_vmp_pmat(data.n(), rows, cols_in, cols_out, size),
        "prepared backing is shorter than its own shape requires"
    );
}

/// The product over a key whose rows are contiguous from `first_physical_row`.
///
/// The `dsize == 1` dense specialization is entered from here, where the rows
/// are known to be contiguous; the dense kernels then trim to the operands,
/// reading `min(rows, a.size())` rows and `min(size, res.size())` limbs.
fn gglwe_product_dft_dense<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    use_: &GGLWEActiveUse,
    term_count: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + GGLWEProductDigitsStridedImpl<BE>,
    Module<BE>: VmpApplyDftToDft<BE>,
{
    let logical = &use_.logical_layout;
    let dsize: usize = logical.dsize().into();
    if dsize == 1 {
        module.vmp_apply_dft_to_dft(res, a, &key.data, 0, scratch);
    } else {
        let product_terms = logical
            .n()
            .as_usize()
            .saturating_mul(logical.dnum().as_usize())
            .saturating_mul(dsize)
            .saturating_mul(logical.rank_in().as_usize().max(1))
            .saturating_mul(term_count.max(1));
        let product_limbs = gadget_product_limbs(logical.base2k(), product_terms);
        BE::gglwe_product_digits_strided(module, res, a, dsize, product_limbs, &key.data, scratch);
    }
}

/// Binds a key through an explicitly requested `dsize`, or fails loudly.
pub(crate) fn resolved_use<K: GGLWEInfos>(key_infos: &K, input_k: TorusPrecision, effective_dsize: Dsize) -> GGLWEUse {
    match resolve_gglwe_key_use(key_infos, input_k, effective_dsize) {
        Ok(Some(use_)) => use_,
        Ok(None) => panic!("key cannot realize dsize={effective_dsize} at input_k={input_k}"),
        Err(e) => panic!("{e}"),
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
        + VecZnxDftZero<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    /// Scratch bound of [`Self::gglwe_product_dft_default`] for the same bound.
    fn gglwe_product_dft_tmp_bytes_default(&self, res_size: usize, a_size: usize, use_: &GGLWEActiveUse) -> usize;

    /// Applies one GGLWE product into a DFT accumulator that will contain
    /// `term_count` such products before normalization.
    ///
    /// Reads only the rows and limb prefix `bound` resolves. Zero precision
    /// never reaches here: the caller has no bound to pass.
    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        use_: &GGLWEActiveUse,
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
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{VecZnxBigToBackendRef, VecZnxToBackendRef},
};

use crate::{
    default::operations::GLWENormalizeDefault,
    layouts::{GLWELayout, GLWEToBackendMut},
    oep::GLWEKeyswitchDefault,
};

/// One shared body: the key is bound at `a.k()` and the product reads only what
/// the bound resolves. Zero precision leaves `res` zeroed without a product.
fn glwe_keyswitch_dft_fill<'r, BE, M, A, K>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    a: &A,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    M: GLWEKeyswitchInternal<BE> + GGLWEProductDefault<BE> + VecZnxDftApply<BE> + VecZnxDftZero<BE>,
{
    let a_ref = a.to_backend_ref();
    assert_eq!(a_ref.base2k(), key.base2k());
    let input_k: TorusPrecision = a.k();
    let use_: GGLWEUse = bound_for(key, input_k);
    let tmp_bytes = module.glwe_keyswitch_internal_tmp_bytes_from_sizes(a_ref.rank().as_usize(), res.size(), a_ref.size(), &use_);
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEKeyswitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );

    let GGLWEUse::Active(active) = use_ else {
        for col in 0..res.cols() {
            module.vec_znx_dft_zero(res, col);
        }
        return;
    };

    let mask_cols = a_ref.rank().as_usize();
    let a_size: usize = a_ref.size();
    let key_ref: GGLWEPreparedBackendRef<'_, BE> = key.to_backend_ref();
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, mask_cols, a_size);
        for col_i in 0..mask_cols {
            let a_data: &VecZnxBackendRef<'_, BE> = &a_ref.data;
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a_data, col_i + 1);
        }
        let a_dft_ref = a_dft.to_backend_ref();
        module.gglwe_product_dft_default(res, &a_dft_ref, &key_ref, &active, 1, &mut scratch_1.borrow());
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
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    glwe_keyswitch_tmp_bytes_dispatch(module, res_infos, a_infos, key_infos)
}

fn glwe_keyswitch_tmp_bytes_dispatch<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
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

    // Sizing follows the bound, never the physical metadata a `with_dsize`
    // wrapper still forwards.
    let use_: GGLWEUse = bound_for(key_infos, a_infos.k());
    let layout: GGLWELayout = match &use_ {
        GGLWEUse::Empty => key_infos.gglwe_layout(),
        GGLWEUse::Active(active) => active.logical_layout,
    };
    let mask_cols = a_infos.rank().as_usize();
    let product_tmp_bytes = |res_size: usize, a_size: usize| -> usize {
        module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, res_size, a_size, &use_)
    };

    let output_cols = res_infos.rank().as_usize() + 1;
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res_infos, a_infos, &layout);
    let a_dft_size = a_infos.k().div_ceil(layout.base2k()) as usize;
    let lvl_0: usize = module.bytes_of_vec_znx_dft(output_cols, output_size);
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(output_cols, output_size);
    let lvl_1: usize = lvl_1_big
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_2: usize = if a_infos.base2k() != layout.base2k() {
        let small_term_tmp: usize = BE::bytes_of_vec_znx(module.n(), 1, output_size);
        let a_conv_infos: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: layout.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes_default()
            .max(product_tmp_bytes(output_size, a_dft_size));
        let lvl_2_2: usize = lvl_1_big
            + small_term_tmp
            + module
                .vec_znx_idft_apply_tmp_bytes()
                .max(module.vec_znx_big_normalize_tmp_bytes())
                .max(module.vec_znx_normalize_tmp_bytes());
        lvl_2_0 + lvl_2_1.max(lvl_2_2)
    } else {
        lvl_1.max(product_tmp_bytes(output_size, a_dft_size))
    };

    lvl_0 + lvl_2
}

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
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    glwe_keyswitch_dispatch(module, res, a, key, scratch)
}

/// Shared body: the key is bound at `a.k()`, so a stored key and a `with_dsize`
/// wrapper take the same path.
fn glwe_keyswitch_dispatch<BE, M, R, A, K>(module: &M, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
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
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
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

    // Sizing follows the bound, never the physical metadata.
    let layout: GGLWELayout = match bound_for(key, a.k()) {
        GGLWEUse::Empty => key.gglwe_layout(),
        GGLWEUse::Active(active) => active.logical_layout,
    };

    let required: usize = module.glwe_keyswitch_tmp_bytes_default(res, a, key);
    assert!(
        scratch.available() >= required,
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        required
    );

    let output_size = gglwe_product_output_size::<BE, _, _, _>(res, a, &layout);

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

    let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, output_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for i in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, &mut scratch.borrow());
    }
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
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &res_small_ref, 0);
    } else {
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a.to_backend_ref().data, 0);
    }
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
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>,
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
