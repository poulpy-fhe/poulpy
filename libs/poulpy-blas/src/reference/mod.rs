//! Reference compositions over poulpy-ckks, the contract every override reproduces.

mod encoding;
mod inter;
mod intra;
mod jkls;
mod matmul;
mod matvec;

pub use encoding::BlasEncodingReference;
pub use inter::BlasMatInterReference;
pub use intra::BlasMatIntraReference;
#[cfg(feature = "test-utils")]
pub(crate) use intra::{sum_line, sum_line_tmp_bytes};
pub use jkls::BlasJklsReference;
pub use matmul::BlasMatMulReference;
pub use matvec::BlasMatVecReference;

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSLinearTransformationOps, LinearTransformation, LinearTransformationBabySteps, LtDiagonalMeta,
        ckks_plaintext_diagonal_layout,
    },
};
use poulpy_core::{
    DiagonalProd,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LinearTransformationPlan},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, ScratchArena},
};

use crate::layouts::ciphertext::layout_of;

/// How [`lt_tmp_bytes`] evaluates its linear transformation: alone, summed over resident terms, or summed over
/// streamed terms of plaintext diagonals.
#[derive(Clone, Copy)]
pub(crate) enum LtEval {
    Single,
    Sum,
    StreamedSum,
}

/// Scratch of the baby steps of a block of layout `a` and of one linear transformation evaluated as `eval`, from
/// blocks of layout `a` into a block of layout `res`.
pub(crate) fn lt_tmp_bytes<BE, M, K>(module: &M, res: &impl CKKSCtBounds, a: &impl CKKSCtBounds, key: &K, eval: LtEval) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    K: GGLWEInfos,
{
    // A `res` of no width is never read: its map fails the budget check before any scratch.
    let live = res.k().as_usize() > 0;
    // The maps take a key per rotation, which every plan of that kind sizes alike.
    let plan = LinearTransformationPlan::default();
    // Plaintext diagonals read no tensor key, so the rotation key stands for its layout.
    let query = |ct: &CKKSLayout| {
        let diagonal = ckks_plaintext_diagonal_layout(ct);
        match eval {
            LtEval::Single => module.ckks_eval_linear_transformation_tmp_bytes(ct, &diagonal, &plan, key, key),
            LtEval::Sum => module.ckks_eval_linear_transformation_sum_tmp_bytes(ct, &diagonal, &plan, key, key),
            LtEval::StreamedSum => module.ckks_eval_linear_transformation_sum_streamed_tmp_bytes(ct, &diagonal, &plan, key, key),
        }
    };
    let res_bytes = if live { query(&layout_of(res)) } else { 0 };
    module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(a, &plan, key)
        .max(query(&layout_of(a)).max(res_bytes))
}

/// `count` baby-step caches for sources of `src`'s layout, each covering the baby steps of every map of `maps`.
pub(crate) fn baby_caches<BE, M, S, P>(
    module: &M,
    maps: &[&LinearTransformation<P>],
    src: &S,
    count: usize,
) -> Vec<LinearTransformationBabySteps<BE>>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let rotations: Vec<i64> = maps.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
    (0..count)
        .map(|_| LinearTransformationBabySteps::alloc(module, &rotations, src))
        .collect()
}

/// The products by public matrices, which share their giant steps across their terms, one per cache of `caches`
/// ([`baby_caches`]): group `g` of `groups` holds the blocks `o` of `res` it writes, block `o` receiving poulpy-ckks's
/// summed linear transformation of the terms `(source(g, l), map(g, o, l))` in ascending `l`, then `label`, the
/// product's [`pt_label`]. The caches are filled from a group's sources and serve all its blocks. The groups run in
/// order, refilling the same caches, and the first error stops them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shared_giants<'s, 'm, BE, M, D, S, P, K>(
    module: &M,
    caches: &mut [LinearTransformationBabySteps<BE>],
    groups: Vec<Vec<&mut D>>,
    source: impl Fn(usize, usize) -> &'s S,
    map: impl Fn(usize, usize, usize) -> &'m LinearTransformation<P>,
    label: usize,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's + 'm,
    M: CKKSLinearTransformationOps<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + 's,
    P: DiagonalProd<BE> + LtDiagonalMeta + 'm,
    K: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    let inner = caches.len();
    let (mut srcs, mut lts) = (Vec::with_capacity(inner), Vec::with_capacity(inner));
    for (g, blocks) in groups.into_iter().enumerate() {
        for (l, cache) in caches.iter_mut().enumerate() {
            module.ckks_prepare_linear_transformation_baby_steps(cache, source(g, l), keys, scratch)?;
        }
        let babies: Vec<_> = caches.iter().collect();
        srcs.clear();
        srcs.extend((0..inner).map(|l| source(g, l)));
        for (o, dst) in blocks.into_iter().enumerate() {
            lts.clear();
            lts.extend((0..inner).map(|l| map(g, o, l)));
            module.ckks_eval_linear_transformation_sum_into(&mut *dst, &srcs, &babies, &lts, keys, scratch)?;
            dst.set_log_sparsity(label);
        }
    }
    Ok(())
}
