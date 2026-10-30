//! Refusal checks shared by the reference bodies, the derived defaults and
//! backend overrides, so that every implementation refuses the same inputs.

use crate::layouts::BabyStepOperand;
use poulpy_hal::layouts::{Backend, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef};

use crate::{
    Distribution,
    layouts::{
        DiagonalProd, GGLWEInfos, LWEInfos, LinearTransformation, LinearTransformationBabySteps, TorusPrecision,
        prepared::GGLWEPreparedBackendRef,
    },
};

/// Checks that the secret an encryption reads has been sampled, before the encryption draws or carves.
///
/// # Panics
///
/// On the `NONE` distribution.
pub(crate) fn assert_secret_set(dist: &Distribution) {
    assert!(
        *dist != Distribution::NONE,
        "glwe secret distribution is NONE (have you prepared the key?)"
    );
}

/// Checks that `key` has no row offset: `op` adds its input back at the input's
/// own place, which only an offset-free key's product matches.
///
/// # Panics
///
/// On a key with a row offset.
pub fn assert_no_row_offset<K: GGLWEInfos>(op: &str, key: &K) {
    assert_eq!(
        key.k_offset().0,
        0,
        "{op}: keys with a row offset are only taken by glwe_keyswitch and glwe_keyswitch_dft"
    );
}

/// Checks that `key` decomposes an input of precision `k` exactly: `k` is at
/// most its [`max_input_k`](crate::layouts::GGLWELayout::max_input_k). A wider
/// input is narrowed by the caller first.
///
/// # Panics
///
/// Unless `key` decomposes an input of precision `k` exactly.
pub fn assert_key_covers<K: GGLWEInfos>(op: &str, k: TorusPrecision, key: &K) {
    assert!(
        k <= key.gglwe_layout().max_input_k(),
        "{op}: the input is wider than the key's max_input_k"
    );
}

/// The checks of [`glwe_eval_linear_transformation_into_reference`](crate::reference::linear_transformation::glwe_eval_linear_transformation_into_reference), run before it reads a key or writes `res`.
pub fn assert_linear_transformation_into<BE, P, L: BabyStepOperand<BE>>(
    lhs: &LinearTransformationBabySteps<BE, L>,
    rhs: &LinearTransformation<P>,
) where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    assert_linear_transformation_giant_steps(&[lhs], &[rhs]);
}

/// The checks of [`glwe_eval_linear_transformation_sum_into_reference`](crate::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference), run before it reads a key or writes `res`.
pub fn assert_linear_transformation_sum<BE, P, L: BabyStepOperand<BE>>(
    babies: &[&LinearTransformationBabySteps<BE, L>],
    lts: &[&LinearTransformation<P>],
) where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    assert_eq!(
        babies.len(),
        lts.len(),
        "linear transformation sum: babies and lts differ in length"
    );
    let (&lhs, &rhs) = (babies.first().expect("linear transformation sum has no terms"), &lts[0]);
    // First, so that a later term's cache holds the rotations its schedule reads before its layout is read.
    assert_linear_transformation_giant_steps(babies, lts);
    let diagonals = |lt: &LinearTransformation<P>| {
        lt.first_diagonal_plaintext()
            .map(|d| (d.glwe_layout(), d.max_size(), DiagonalProd::log_scale(d)))
    };
    for (t, (&l, &r)) in babies.iter().zip(lts).enumerate().skip(1) {
        assert!(
            r.giant_steps
                .iter()
                .map(|gs| gs.rot)
                .eq(rhs.giant_steps.iter().map(|gs| gs.rot)),
            "summed linear transformation {t} does not share the first one's giant schedule"
        );
        assert!(
            diagonals(r) == diagonals(rhs),
            "summed linear transformation {t} does not share the first one's diagonal layout, width and scale"
        );
        assert!(
            (l.size(), l.cols()) == (lhs.size(), lhs.cols()),
            "summed linear transformation {t} does not share the first one's source layout"
        );
    }
}

/// Checks the giant steps of every term. The other diagonals are not checked: every diagonal of a transform shares
/// one width, scale and degree, as the encoder produces them, and preparing asserts the degree. Reads no cache
/// layout, so an empty cache fails on its first missing rotation.
///
/// # Panics
///
/// Unless the first term has a giant step, and every giant step of every term has a diagonal as wide as its
/// transform's first diagonal and reads only baby rotations its term's cache holds.
fn assert_linear_transformation_giant_steps<BE, P, L: BabyStepOperand<BE>>(
    babies: &[&LinearTransformationBabySteps<BE, L>],
    lts: &[&LinearTransformation<P>],
) where
    BE: Backend,
    P: LWEInfos,
{
    assert!(
        !lts[0].giant_steps.is_empty(),
        "linear transformation has no non-empty giant steps"
    );
    for (t, (&lhs, &lt)) in babies.iter().zip(lts).enumerate() {
        let width = lt.first_diagonal_plaintext().map(|d| d.max_size());
        for (g, gs) in lt.giant_steps.iter().enumerate() {
            let first = gs
                .diagonals
                .first()
                .unwrap_or_else(|| panic!("giant step {g} of linear transformation {t} has no diagonals"));
            assert!(
                Some(first.plaintext.max_size()) == width,
                "giant step {g} of linear transformation {t} is not as wide as the first diagonal"
            );
            for d in &gs.diagonals {
                assert!(
                    lhs.contains_baby_step(d.baby),
                    "missing prepared baby-step rotation {} of linear transformation {t}",
                    d.baby
                );
            }
        }
    }
}

/// Checks a gadget product `op` takes into the first `product_size` limbs of `res`, from `a` by
/// `key`, with `addends` on its first columns, as
/// [`GGLWEProductDigitsStridedImpl::gglwe_product_fused`](crate::oep::GGLWEProductDigitsStridedImpl::gglwe_product_fused)
/// takes it.
///
/// # Panics
///
/// On more addends than output columns, a product wider than `res`, operands or addends of
/// another degree, operand columns other than the key's or an addend column out of range.
pub fn assert_gglwe_product<BE: Backend>(
    op: &str,
    res: &VecZnxDftBackendMut<'_, BE>,
    product_size: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
) {
    assert!(addends.len() <= res.cols(), "{op}: more addends than output columns");
    assert!(product_size <= res.size(), "{op}: product wider than its destination");
    assert!(
        a.n() == res.n() && key.n().as_usize() == res.n(),
        "{op}: operands of different degrees"
    );
    assert!(
        a.cols() == key.rank_in().as_usize() && res.cols() == key.rank_out().as_usize() + 1,
        "{op}: operand columns differ from the key's"
    );
    for (addend, col) in addends.iter().flatten() {
        assert!(
            addend.n() == res.n() && *col < addend.cols(),
            "{op}: an addend of another degree or column"
        );
    }
}

/// Checks a gadget product `op` takes into `res_dft`, then inverse transforms and normalizes
/// into `res` with `body` on column 0, as
/// [`GGLWEProductDigitsStridedImpl::gglwe_product_idft_normalize`](crate::oep::GGLWEProductDigitsStridedImpl::gglwe_product_idft_normalize)
/// takes it: the checks of [`assert_gglwe_product`] on the product, then the destination and
/// the body.
///
/// # Panics
///
/// As [`assert_gglwe_product`], and on a destination of another degree or with fewer columns
/// than the product, or a body of another degree or column.
pub fn assert_gglwe_product_idft_normalize<BE: Backend>(
    op: &str,
    res: &VecZnxBackendMut<'_, BE>,
    res_dft: &VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    addends: &[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>],
    body: Option<(&VecZnxBackendRef<'_, BE>, usize)>,
) {
    assert_gglwe_product(op, res_dft, res_dft.size(), a, key, addends);
    assert!(
        res.n() == res_dft.n() && res.cols() >= res_dft.cols(),
        "{op}: the destination differs from the product in degree or columns"
    );
    if let Some((body, col)) = body {
        assert!(
            body.n() == res.n() && col < body.cols(),
            "{op}: a body of another degree or column"
        );
    }
}

/// Checks a key switch `op` takes of the first `source_size` limbs of the transform-domain
/// `source` into `res` over `output_size` limbs, as
/// [`GGLWEProductDigitsStridedImpl::gglwe_products_from_dft`](crate::oep::GGLWEProductDigitsStridedImpl::gglwe_products_from_dft)
/// takes each source.
///
/// # Panics
///
/// On a source of another degree or column count than `res`, more limbs than it holds, a
/// base2k other than the key's, a product wider than `res`, columns other than the key's or a
/// key with a row offset.
pub fn assert_gglwe_product_from_dft<BE: Backend>(
    op: &str,
    res: &VecZnxDftBackendMut<'_, BE>,
    source: &VecZnxDftBackendMut<'_, BE>,
    source_size: usize,
    source_base2k: usize,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    output_size: usize,
) {
    assert!(
        source.cols() == res.cols() && source.n() == res.n(),
        "{op}: a source differs from its destination in degree or columns"
    );
    assert!(source_size <= source.size(), "{op}: more source limbs than it holds");
    assert!(
        source_base2k == key.base2k().as_usize(),
        "{op}: the source's base2k differs from the key's"
    );
    assert!(
        output_size.min(key.size()) <= res.size(),
        "{op}: product wider than its destination"
    );
    assert!(
        key.n().as_usize() == res.n()
            && key.rank_in().as_usize() + 1 == res.cols()
            && key.rank_out().as_usize() + 1 == res.cols(),
        "{op}: operand columns differ from the key's"
    );
    assert_no_row_offset(op, key);
}
