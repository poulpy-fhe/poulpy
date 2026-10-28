//! Refusals of the step hooks of [`GGLWEProductDigitsStridedImpl`] and of the operand layout
//! their overrides share: each is made before the hook reads a key or writes a destination, the
//! shape checks on an arena of zero bytes.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc},
    layouts::{
        Backend, DataView, Module, ScratchOwned, VecZnx, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
        VecZnxToBackendRef,
    },
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    api::GGLWEPreparedFactory,
    gadget::gadget_dft_apply_from_dft,
    layouts::{
        Base2K, Degree, Dnum, GLWESwitchingKeyLayout, LWEInfos, Rank, TorusPrecision,
        prepared::{GGLWEPrepared, GGLWEPreparedToBackendRef},
    },
    oep::GGLWEProductDigitsStridedImpl,
    test_suite::{poisoned_scratch, refusal::assert_refusal},
};

type Dft<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, <BE as Backend>::DftWord, BE>;
type Coefficients<BE> = VecZnx<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// A zero key of rank `rank`, two rows of one limb each at `base2k`, and a row offset of
/// `k_offset` bits.
fn key<BE: Backend>(module: &Module<BE>, base2k: usize, rank: u32, k_offset: u32) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    Module<BE>: GGLWEPreparedFactory<BE>,
{
    module.gglwe_prepared_alloc_from_infos(&GLWESwitchingKeyLayout {
        n: Degree(module.n() as u32),
        base2k: Base2K(base2k as u32),
        dnum: Dnum(2),
        k_aux: TorusPrecision(base2k as u32),
        rank_in: Rank(rank),
        rank_out: Rank(rank),
        k_digit: TorusPrecision(base2k as u32),
        k_offset: TorusPrecision(k_offset),
    })
}

fn dft_bytes<BE: Backend>(vectors: &[Dft<BE>]) -> Vec<Vec<u8>> {
    vectors.iter().map(|v| BE::to_host_bytes(v.data())).collect()
}

fn coefficient_bytes<BE: Backend>(vectors: &[Coefficients<BE>]) -> Vec<Vec<u8>> {
    vectors.iter().map(|v| BE::to_host_bytes(v.data())).collect()
}

/// `gglwe_product_fused` refuses too many addends, a product wider than its destination,
/// operands of another degree, operand columns other than the key's, an addend column out of
/// range and a scratch below its query, before writing `res`.
pub fn test_gglwe_product_fused_rejects_invalid_operands<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: GGLWEPreparedFactory<BE> + VecZnxDftAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k) = (module.n(), params.base2k);
    let key = key(module, base2k, 1, 0);
    let size = key.size();
    let wide = BE::gglwe_product_fused_tmp_bytes(module, size, size, 2, &key);
    let a = module.vec_znx_dft_alloc(n, 1, 2);
    let half = module.vec_znx_dft_alloc(n / 2, 1, 2);
    let pair = module.vec_znx_dft_alloc(n, 2, 2);
    let addend = module.vec_znx_dft_alloc(n, 1, 2);
    let addend = addend.to_backend_ref();
    let cases = [
        (
            "gglwe_product_fused: more addends than output columns",
            &a,
            size,
            vec![None; 3],
            Some(0),
        ),
        (
            "gglwe_product_fused: product wider than its destination",
            &a,
            size + 1,
            vec![],
            Some(0),
        ),
        (
            "gglwe_product_fused: operands of different degrees",
            &half,
            size,
            vec![],
            Some(0),
        ),
        (
            "gglwe_product_fused: operand columns differ from the key's",
            &pair,
            size,
            vec![],
            Some(0),
        ),
        (
            "gglwe_product_fused: an addend of another degree or column",
            &a,
            size,
            vec![Some((&addend, 1))],
            Some(0),
        ),
        (
            "gglwe_product_fused: scratch below its query",
            &a,
            size,
            vec![],
            wide.checked_sub(1),
        ),
    ];
    for (message, a, product_size, addends, scratch) in cases {
        // A query of zero bytes has no scratch to refuse.
        let Some(scratch) = scratch else { continue };
        let mut res = vec![module.vec_znx_dft_alloc(n, 2, size)];
        assert_refusal(
            message,
            &mut res,
            |res| dft_bytes::<BE>(res),
            |res| {
                BE::gglwe_product_fused(
                    module,
                    &mut res[0].to_backend_mut(),
                    product_size,
                    &a.to_backend_ref(),
                    &key.to_backend_ref(),
                    1,
                    &addends,
                    1,
                    false,
                    &mut poisoned_scratch::<BE>(scratch).borrow(),
                )
            },
        );
    }
}

/// `gglwe_product_idft_normalize` refuses a destination of fewer columns than the product, a
/// body column out of range and a scratch below its query, before writing `res`.
pub fn test_gglwe_product_idft_normalize_rejects_invalid_operands<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: GGLWEPreparedFactory<BE> + VecZnxDftAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k) = (module.n(), params.base2k);
    let key = key(module, base2k, 1, 0);
    let size = key.size();
    let wide = BE::gglwe_product_idft_normalize_tmp_bytes(module, 2, size, 2, &key);
    let a = module.vec_znx_dft_alloc(n, 1, 2);
    let body = module.vec_znx_alloc(n, 1, 2);
    let body = VecZnxToBackendRef::<BE>::to_backend_ref(&body);
    let cases = [
        (
            "gglwe_product_idft_normalize: the destination differs from the product in degree or columns",
            1,
            None,
            Some(0),
        ),
        (
            "gglwe_product_idft_normalize: a body of another degree or column",
            2,
            Some(1),
            Some(0),
        ),
        (
            "gglwe_product_idft_normalize: scratch below its query",
            2,
            None,
            wide.checked_sub(1),
        ),
    ];
    for (message, cols, body_col, scratch) in cases {
        // A query of zero bytes has no scratch to refuse.
        let Some(scratch) = scratch else { continue };
        let mut state = (
            vec![module.vec_znx_alloc(n, cols, 2)],
            vec![module.vec_znx_dft_alloc(n, 2, size)],
        );
        assert_refusal(
            message,
            &mut state,
            |(res, _)| coefficient_bytes::<BE>(res),
            |(res, res_dft)| {
                BE::gglwe_product_idft_normalize(
                    module,
                    &mut vec_znx_backend_mut::<BE>(&mut res[0]),
                    base2k,
                    2 * base2k,
                    0,
                    &mut res_dft[0].to_backend_mut(),
                    &a.to_backend_ref(),
                    &key.to_backend_ref(),
                    1,
                    &[],
                    body_col.map(|col| (&body, col)),
                    base2k,
                    &mut poisoned_scratch::<BE>(scratch).borrow(),
                )
            },
        );
    }
}

/// `gglwe_products_idft_normalize` refuses items of different lengths, an invalid later item
/// and a scratch below its query, before writing the first destination.
pub fn test_gglwe_products_idft_normalize_rejects_invalid_items<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: GGLWEPreparedFactory<BE> + VecZnxDftAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k) = (module.n(), params.base2k);
    let key = key(module, base2k, 1, 0);
    let size = key.size();
    let wide = BE::gglwe_product_idft_normalize_tmp_bytes(module, 2, size, 2, &key);
    let a = module.vec_znx_dft_alloc(n, 1, 2);
    let cases = [
        (
            "gglwe_products_idft_normalize: items of different lengths",
            [2, 2],
            1,
            Some(0),
        ),
        (
            "gglwe_products_idft_normalize: the destination differs from the product in degree or columns",
            [2, 1],
            2,
            Some(0),
        ),
        (
            "gglwe_products_idft_normalize: scratch below its query",
            [2, 2],
            2,
            wide.checked_sub(1),
        ),
    ];
    for (message, cols, products, scratch) in cases {
        // A query of zero bytes has no scratch to refuse.
        let Some(scratch) = scratch else { continue };
        let mut state = (
            cols.iter().map(|&cols| module.vec_znx_alloc(n, cols, 2)).collect::<Vec<_>>(),
            (0..products)
                .map(|_| module.vec_znx_dft_alloc(n, 2, size))
                .collect::<Vec<_>>(),
        );
        assert_refusal(
            message,
            &mut state,
            |(res, _)| coefficient_bytes::<BE>(res),
            |(res, res_dft)| {
                let mut res: Vec<_> = res.iter_mut().map(|v| vec_znx_backend_mut::<BE>(v)).collect();
                let mut res_dft: Vec<_> = res_dft.iter_mut().map(|v| v.to_backend_mut()).collect();
                BE::gglwe_products_idft_normalize(
                    module,
                    &mut res,
                    base2k,
                    2 * base2k,
                    0,
                    &mut res_dft,
                    &[a.to_backend_ref(), a.to_backend_ref()],
                    &key.to_backend_ref(),
                    1,
                    &[],
                    &[None, None],
                    base2k,
                    &mut poisoned_scratch::<BE>(scratch).borrow(),
                )
            },
        );
    }
}

/// `gglwe_products_from_dft` refuses items of different lengths, a source of other columns than
/// its destination, more source limbs than it holds, a source radix other than the key's, a
/// product wider than its destination, columns other than the key's, a key with a row offset
/// and a scratch below its query, before writing a destination or a source.
pub fn test_gglwe_products_from_dft_rejects_invalid_items<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>: GGLWEPreparedFactory<BE> + VecZnxDftAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k) = (module.n(), params.base2k);
    let (rank1, rank2, offset) = (
        key(module, base2k, 1, 0),
        key(module, base2k, 2, 0),
        key(module, base2k, 1, 1),
    );
    let size = rank1.size();
    let wide = BE::gglwe_products_from_dft_tmp_bytes(module, 2, 1, 2, &rank1);
    // (message, key, columns of the second item's destination and source, its destination
    // limbs, source limbs, source radix, number of source sizes, scratch)
    let cases = [
        (
            "gglwe_products_from_dft: items of different lengths",
            &rank1,
            [2, 2],
            size,
            2,
            base2k,
            1,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: a source differs from its destination in degree or columns",
            &rank1,
            [2, 1],
            size,
            2,
            base2k,
            2,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: more source limbs than it holds",
            &rank1,
            [2, 2],
            size,
            3,
            base2k,
            2,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: the source's base2k differs from the key's",
            &rank1,
            [2, 2],
            size,
            2,
            base2k + 1,
            2,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: product wider than its destination",
            &rank1,
            [2, 2],
            size - 1,
            2,
            base2k,
            2,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: operand columns differ from the key's",
            &rank2,
            [2, 2],
            size,
            2,
            base2k,
            2,
            Some(0),
        ),
        (
            "assertion `left == right` failed: gglwe_products_from_dft: keys with a row offset are only taken by \
             glwe_keyswitch and glwe_keyswitch_dft\n  left: 1\n right: 0",
            &offset,
            [2, 2],
            size,
            2,
            base2k,
            2,
            Some(0),
        ),
        (
            "gglwe_products_from_dft: scratch below its query",
            &rank1,
            [2, 2],
            size,
            2,
            base2k,
            2,
            wide.checked_sub(1),
        ),
    ];
    for (message, key, [res_cols, source_cols], res_size, source_size, source_base2k, sizes, scratch) in cases {
        // A query of zero bytes has no scratch to refuse.
        let Some(scratch) = scratch else { continue };
        let mut state = (
            vec![
                module.vec_znx_dft_alloc(n, 2, size),
                module.vec_znx_dft_alloc(n, res_cols, res_size),
            ],
            vec![module.vec_znx_dft_alloc(n, 2, 2), module.vec_znx_dft_alloc(n, source_cols, 2)],
        );
        assert_refusal(
            message,
            &mut state,
            |(res, sources)| (dft_bytes::<BE>(res), dft_bytes::<BE>(sources)),
            |(res, sources)| {
                let mut res: Vec<_> = res.iter_mut().map(|v| v.to_backend_mut()).collect();
                let mut sources: Vec<_> = sources.iter_mut().map(|v| v.to_backend_mut()).collect();
                BE::gglwe_products_from_dft(
                    module,
                    1,
                    1,
                    &mut res,
                    size,
                    &[false, false],
                    &mut sources,
                    &[2, source_size][..sizes],
                    source_base2k,
                    &key.to_backend_ref(),
                    &mut poisoned_scratch::<BE>(scratch).borrow(),
                )
            },
        );
    }
}

/// `gadget_dft_apply_from_dft` refuses a source radix other than the key's and a scratch below
/// its query, before writing `res` or the source.
pub fn test_gadget_dft_apply_from_dft_rejects_invalid_operands<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: Backend + GGLWEProductDigitsStridedImpl,
    Module<BE>:
        GGLWEPreparedFactory<BE> + VecZnxDftAlloc<BE> + crate::reference::keyswitching::glwe::GGLWEProductFusedReference<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k) = (module.n(), params.base2k);
    let key = key(module, base2k, 1, 0);
    let cases = [
        (
            "gadget_dft_apply_from_dft: the source's base2k differs from the key's",
            base2k + 1,
        ),
        ("gadget_dft_apply_from_dft: scratch below its query", base2k),
    ];
    for (message, source_base2k) in cases {
        let mut state = (
            vec![module.vec_znx_dft_alloc(n, 1, 4)],
            vec![module.vec_znx_dft_alloc(n, 2, 2)],
        );
        assert_refusal(
            message,
            &mut state,
            |(res, source)| (dft_bytes::<BE>(res), dft_bytes::<BE>(source)),
            |(res, source)| {
                gadget_dft_apply_from_dft(
                    module,
                    &mut res[0].to_backend_mut(),
                    &mut source[0].to_backend_mut(),
                    source_base2k,
                    &key.to_backend_ref(),
                    2,
                    &mut poisoned_scratch::<BE>(0).borrow(),
                )
            },
        );
    }
}
