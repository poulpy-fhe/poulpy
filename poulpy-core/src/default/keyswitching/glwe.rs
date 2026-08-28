use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes, VmpApplyDftToDftTmpBytes, VmpExtractSelectedRows,
        VmpPMatBytesOf,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VmpPMatToBackendRef,
    },
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        Dsize, GGLWEActiveUse, GGLWEBind, GGLWEInfos, GGLWELayout, GGLWEUse, GLWEInfos, GLWEToBackendRef,
        GadgetProductOutputSizeParams, LWEInfos, TorusPrecision, gadget_product_limbs, gadget_product_output_size,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedBound, GGLWEPreparedToBackendRef},
        resolve_gglwe_key_use,
    },
    oep::{GGLWEProductBoundImpl, gglwe_product_digit_output_size},
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
        lvl_0
            .checked_add(lvl_1)
            .expect("GGLWE keyswitch scratch size overflows usize")
    }

    /// Conservative counterpart of
    /// [`Self::glwe_keyswitch_internal_tmp_bytes_from_sizes`] for proxy/bound
    /// queries. A dense representative use may become a strided, materialized
    /// use at a lower precision, so only the nested product branch is widened.
    fn glwe_keyswitch_internal_tmp_bytes_from_sizes_upper(
        &self,
        mask_cols: usize,
        res_size: usize,
        a_size: usize,
        use_: &GGLWEUse,
    ) -> usize {
        let lvl_0 = self.bytes_of_vec_znx_dft(mask_cols, a_size);
        let lvl_1 = match use_ {
            GGLWEUse::Empty => 0,
            GGLWEUse::Active(active) => self.gglwe_product_dft_tmp_bytes_upper_default(res_size, a_size, active),
        };
        lvl_0
            .checked_add(lvl_1)
            .expect("GGLWE keyswitch upper scratch size overflows usize")
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
///
/// For handing a key shape to an API that takes one. Product sizing goes
/// through [`bound_output_size`], which takes the bound itself so a physical
/// key cannot stand in for it.
pub fn bound_layout<K: GGLWEInfos>(key: &K, input_k: TorusPrecision) -> GGLWELayout {
    match bound_for(key, input_k) {
        GGLWEUse::Empty => key.gglwe_layout(),
        GGLWEUse::Active(active) => *active.logical_layout(),
    }
}

/// Binds a key for `input_k`, or fails loudly: the seam never falls back to the
/// physical decomposition.
///
/// Raw keys may intentionally stop before disposable low/auxiliary digits;
/// [`GGLWEBind::bind_for`] preserves that established product behavior. Key
/// registries and precision-aware helpers use `bind_covering_for` when complete
/// input coverage is required.
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
    BE: GGLWEProductBoundImpl<BE>,
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
        let logical = &use_.logical_layout();
        let cols_in: usize = logical.rank_in().into();
        if is_whole_dense_matrix(use_) {
            let dnum: usize = logical.dnum().into();
            let cols_out: usize = (logical.rank_out() + 1).into();
            return self.vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, dnum, cols_in, cols_out, use_.logical_work_size());
        }
        BE::gglwe_product_bound_tmp_bytes(self, res_size, cols_in, a_size, use_)
    }

    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        bound: &GGLWEPreparedBound<'_, BE>,
        term_count: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let use_: &GGLWEActiveUse = bound.use_();
        let a_size: usize = a.size();
        assert_eq!(
            a_size,
            use_.input_size(),
            "the product input must carry ceil(input_k / base2k) limbs for input_k={}",
            use_.input_k()
        );
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, use_),
            "scratch.available(): {} < GGLWEProductDefault::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes_default(res.size(), a_size, use_)
        );

        // The dense VMP specialization is kept only when the bound is the whole
        // stored matrix: every row, every limb, one digit. Anything narrower
        // goes through the bound product, which is what enforces the row map and
        // the limb prefix.
        if is_whole_dense_matrix(use_) {
            self.vmp_apply_dft_to_dft(res, a, bound.pmat(), 0, scratch);
            return;
        }
        BE::gglwe_product_bound(self, res, a, bound, product_limbs_of(use_, term_count), scratch);
    }
}

/// Whether the bound is the complete stored matrix under a single gadget digit,
/// the one shape a plain dense VMP realizes exactly.
fn is_whole_dense_matrix(use_: &GGLWEActiveUse) -> bool {
    use_.is_dense() && use_.logical_layout().dsize().as_usize() == 1
}

/// Coefficient products one accumulation sums, over the logical key.
///
/// Panics rather than saturating: a saturated count silently understates the
/// retained window, which is not noise-visible.
fn product_terms_of(use_: &GGLWEActiveUse, term_count: usize) -> usize {
    let logical: &GGLWELayout = use_.logical_layout();
    let factors: [usize; 5] = [
        logical.n().as_usize(),
        logical.dnum().as_usize(),
        logical.dsize().as_usize(),
        logical.rank_in().as_usize().max(1),
        term_count.max(1),
    ];
    factors
        .iter()
        .try_fold(1usize, |acc, f| acc.checked_mul(*f))
        .unwrap_or_else(|| panic!("product term count overflows usize for logical layout {logical:?} over {term_count} term(s)"))
}

/// Spill width of the coefficient-product accumulation for this bound.
fn product_limbs_of(use_: &GGLWEActiveUse, term_count: usize) -> usize {
    gadget_product_limbs(use_.logical_layout().base2k(), product_terms_of(use_, term_count))
}

/// Reference scratch for [`gglwe_product_bound_default`].
#[doc(hidden)]
pub fn gglwe_product_bound_tmp_bytes_default<BE: Backend>(
    module: &Module<BE>,
    res_size: usize,
    a_cols: usize,
    a_size: usize,
    use_: &GGLWEActiveUse,
) -> usize
where
    Module<BE>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VmpApplyDftToDftAccumulateTmpBytes + VmpPMatBytesOf,
{
    let logical = &use_.logical_layout();
    let (dsize, dnum) = (logical.dsize().as_usize(), logical.dnum().as_usize());
    let cols_in: usize = logical.rank_in().into();
    let cols_out: usize = logical
        .rank_out()
        .as_usize()
        .checked_add(1)
        .expect("GGLWE product output column count overflows usize");
    let key_size: usize = use_.logical_work_size();
    let product: usize = gglwe_product_digits_strided_tmp_bytes_default(
        module, res_size, a_cols, a_size, dsize, dnum, cols_in, cols_out, key_size,
    );
    if use_.is_dense() {
        return product;
    }
    // ponytail: the reference materializes the selection, one extra pass over
    // it. A backend addressing the row map in its own kernel overrides
    // `gglwe_product_bound` and drops this.
    module
        .bytes_of_vmp_pmat(dnum, cols_in, cols_out, key_size)
        .checked_add(product)
        .expect("GGLWE product scratch size overflows usize")
}

/// Runs `f` on the matrix the bound resolves.
///
/// A bound that is the whole stored matrix is handed over in place. Anything
/// narrower is materialized into the logical shape first, so `f` never sees a
/// row or a limb the bound excludes. A backend able to address the row map and
/// the limb prefix itself skips this and reads the stored matrix directly.
pub fn with_bound_pmat<BE, R>(
    module: &Module<BE>,
    bound: &GGLWEPreparedBound<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
    f: impl FnOnce(&poulpy_hal::layouts::VmpPMatBackendRef<'_, BE>, &mut ScratchArena<'_, BE>) -> R,
) -> R
where
    BE: Backend,
    Module<BE>: ModuleN + VmpExtractSelectedRows<BE> + VmpPMatBytesOf,
{
    let use_: &GGLWEActiveUse = bound.use_();
    if use_.is_dense() {
        return f(bound.pmat(), scratch);
    }
    let logical = &use_.logical_layout();
    let (rows, cols_in, cols_out) = (
        logical.dnum().as_usize(),
        logical.rank_in().as_usize(),
        (logical.rank_out() + 1).as_usize(),
    );
    scratch.scope(|scratch_phase| {
        let (mut selected, mut scratch_1) =
            scratch_phase.take_vmp_pmat_scratch(module, rows, cols_in, cols_out, use_.logical_work_size());
        module.vmp_extract_selected_rows(
            &mut selected,
            bound.pmat(),
            use_.first_physical_row(),
            use_.physical_row_step().get(),
        );
        f(&selected.to_backend_ref(), &mut scratch_1.borrow())
    })
}

/// Reference GGLWE product over a bound key.
///
/// A bound that is not the whole stored matrix is materialized into the logical
/// shape first: `vmp_extract_selected_rows` copies rows
/// `first_physical_row + i * physical_row_step` truncated to `logical_work_size`
/// limbs, so the product is exactly the product with the logical key and no
/// limb outside the prefix is read.
#[doc(hidden)]
pub fn gglwe_product_bound_default<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    bound: &GGLWEPreparedBound<'_, BE>,
    product_limbs: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VmpExtractSelectedRows<BE>
        + VmpPMatBytesOf,
{
    let dsize: usize = bound.use_().logical_layout().dsize().into();
    with_bound_pmat(module, bound, scratch, |pmat, scratch| {
        gglwe_product_digits_strided_default(module, res, a, dsize, product_limbs, pmat, scratch);
    });
}

/// Pairs a prepared key with the bound resolved from it, or fails loudly.
///
/// Public for the same reason as [`bound_for`]: every caller that reaches a
/// GGLWE product goes through it, including the ones outside this crate.
pub fn bound_prepared<'a, BE: Backend>(key: GGLWEPreparedBackendRef<'a, BE>, use_: GGLWEActiveUse) -> GGLWEPreparedBound<'a, BE> {
    match GGLWEPreparedBound::new(key, use_) {
        Ok(bound) => bound,
        Err(e) => panic!("{e}"),
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

    /// Upper bound of [`Self::gglwe_product_dft_tmp_bytes_default`] over every
    /// input precision this key can be bound at, at or below `use_`'s.
    ///
    /// The exact requirement is not monotone in the input precision: a bound
    /// that is the whole stored matrix is read in place, and any narrower one is
    /// materialized first. A query answered from a proxy operand, rather than
    /// from the ciphertext the operation will really run on, has to carry that
    /// materialization even when the proxy itself binds the whole matrix.
    fn gglwe_product_dft_tmp_bytes_upper_default(&self, res_size: usize, a_size: usize, use_: &GGLWEActiveUse) -> usize {
        let exact: usize = self.gglwe_product_dft_tmp_bytes_default(res_size, a_size, use_);
        if !use_.is_dense() {
            return exact;
        }
        let logical = &use_.logical_layout();
        let cols_out: usize = logical
            .rank_out()
            .as_usize()
            .checked_add(1)
            .expect("GGLWE product output column count overflows usize");
        exact
            .checked_add(self.bytes_of_vmp_pmat(
                logical.dnum().as_usize(),
                logical.rank_in().as_usize(),
                cols_out,
                use_.logical_work_size(),
            ))
            .expect("GGLWE product upper scratch size overflows usize")
    }

    /// Applies one GGLWE product into a DFT accumulator that will contain
    /// `term_count` such products before normalization.
    ///
    /// Reads only the rows and limb prefix the bound resolves. Zero precision
    /// never reaches here: the caller has no bound to pass.
    fn gglwe_product_dft_default<'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        bound: &GGLWEPreparedBound<'_, BE>,
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
    module
        .bytes_of_vec_znx_dft(a_cols, digit_size)
        .checked_add(apply.max(accumulate))
        .expect("GGLWE digit-product scratch size overflows usize")
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
        let digit_size = (a_size.checked_add(di).expect("GGLWE product digit offset overflows usize") / dsize).min(dnum);
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
    let bound: GGLWEPreparedBound<'_, BE> = bound_prepared(key.to_backend_ref(), active);
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft_scratch(module, mask_cols, a_size);
        for col_i in 0..mask_cols {
            let a_data: &VecZnxBackendRef<'_, BE> = &a_ref.data;
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a_data, col_i + 1);
        }
        let a_dft_ref = a_dft.to_backend_ref();
        module.gglwe_product_dft_default(res, &a_dft_ref, &bound, 1, &mut scratch_1.borrow());
    });
}

/// Practical limb window used for an immediately normalized GGLWE/VMP product.
///
/// Any lower limb can affect rounding through a sufficiently long carry chain.
/// On exact transform backends the retained window covers the live precision
/// plus the worst-case norm growth of the signed polynomial products and VMP
/// accumulation; approximate backends retain the complete work region.
pub fn bound_output_size<BE, R>(res_infos: &R, use_: &GGLWEUse) -> usize
where
    BE: Backend,
    R: LWEInfos,
{
    bound_accumulation_output_size::<BE, _>(res_infos, use_, 1)
}

/// Number of limbs required when `term_count` GGLWE/VMP products are summed
/// before a single normalization.
///
/// Relative to one product, summing `term_count` values can amplify the tail
/// by that factor. Exact backends account for this in the product-norm window;
/// approximate backends keep the complete work region.
pub fn bound_accumulation_output_size<BE, R>(res_infos: &R, use_: &GGLWEUse, term_count: usize) -> usize
where
    BE: Backend,
    R: LWEInfos,
{
    bound_accumulation_output_size_with_tail::<BE, _>(res_infos, use_, term_count, 0)
}

/// The accumulator width for one bound product into `res_infos`.
///
/// The input precision comes from the bound and nowhere else: it is the
/// precision the bound was resolved at, so sizing and execution cannot be given
/// two different answers about the same product.
pub(crate) fn bound_accumulation_output_size_with_tail<BE, R>(
    res_infos: &R,
    use_: &GGLWEUse,
    term_count: usize,
    extra_live_limbs: usize,
) -> usize
where
    BE: Backend,
    R: LWEInfos,
{
    // No row is active, so no product runs and the accumulator only has to hold
    // the destination.
    let GGLWEUse::Active(active) = use_ else {
        return res_infos.size();
    };
    let logical: &GGLWELayout = active.logical_layout();
    gadget_product_output_size(GadgetProductOutputSizeParams {
        key_size: active.logical_work_size(),
        key_base2k: logical.base2k(),
        input_k: active.input_k(),
        output_k: res_infos.k(),
        dsize: logical.dsize(),
        k_aux: logical.k_aux(),
        dft_is_exact: BE::DFT_IS_EXACT,
        product_terms: product_terms_of(active, term_count),
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
    glwe_keyswitch_tmp_bytes_dispatch(module, res_infos, a_infos, key_infos, false)
}

/// Upper bound of [`glwe_keyswitch_tmp_bytes_default`] for proxy/bound queries
/// whose representative key use may bind more narrowly at the eventual input
/// precision. Exact operation queries must keep using the exact function.
pub fn glwe_keyswitch_tmp_bytes_upper_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
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
    glwe_keyswitch_tmp_bytes_dispatch(module, res_infos, a_infos, key_infos, true)
}

fn glwe_keyswitch_tmp_bytes_dispatch<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K, upper: bool) -> usize
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
        GGLWEUse::Active(active) => *active.logical_layout(),
    };
    let mask_cols = a_infos.rank().as_usize();
    let product_tmp_bytes = |res_size: usize, a_size: usize| -> usize {
        if upper {
            module.glwe_keyswitch_internal_tmp_bytes_from_sizes_upper(mask_cols, res_size, a_size, &use_)
        } else {
            module.glwe_keyswitch_internal_tmp_bytes_from_sizes(mask_cols, res_size, a_size, &use_)
        }
    };

    let output_cols = res_infos.rank().as_usize() + 1;
    let output_size = bound_output_size::<BE, _>(res_infos, &use_);
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
    let use_: GGLWEUse = bound_for(key, a.k());
    let required: usize = module.glwe_keyswitch_tmp_bytes_default(res, a, key);
    assert!(
        scratch.available() >= required,
        "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        required
    );

    let output_size = bound_output_size::<BE, _>(res, &use_);

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

    let output_size = bound_output_size::<BE, _>(res, &bound_for(key, res.k()));

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
