//! Refusal checks shared by the reference bodies, the derived defaults and
//! backend overrides, so that every implementation refuses the same inputs.

use poulpy_hal::layouts::Backend;

use crate::{
    Distribution,
    layouts::{
        DiagonalProd, GGLWEInfos, GLWEInfos, LWEInfos, LinearTransformation, LinearTransformationBabySteps,
        LinearTransformationPrepared, TorusPrecision,
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
pub fn assert_linear_transformation_into<BE, P>(lhs: &LinearTransformationBabySteps<BE>, rhs: &LinearTransformation<P>)
where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    assert_linear_transformation_giant_steps(&[lhs], &[rhs]);
}

/// The checks of [`glwe_eval_linear_transformation_sum_into_reference`](crate::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference), run before it reads a key or writes `res`.
pub fn assert_linear_transformation_sum<BE: Backend>(
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformationPrepared<BE>],
) {
    assert_eq!(
        babies.len(),
        lts.len(),
        "linear transformation sum: babies and lts differ in length"
    );
    let (&lhs, &rhs) = (babies.first().expect("linear transformation sum has no terms"), &lts[0]);
    // First, so that a later term's cache holds the rotations its schedule reads before its layout is read.
    assert_linear_transformation_giant_steps(babies, lts);
    let diagonals = |lt: &LinearTransformationPrepared<BE>| {
        lt.first_diagonal_plaintext()
            .map(|d| (d.glwe_layout(), d.max_size(), d.log_scale()))
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
fn assert_linear_transformation_giant_steps<BE, P>(
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
) where
    BE: Backend,
    P: DiagonalProd<BE>,
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
