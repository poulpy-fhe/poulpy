use std::marker::PhantomData;

use criterion::{
    Criterion,
    measurement::{Measurement, WallTime},
};
use poulpy_blas::oep::{BlasMatIntraImpl, BlasMatMulImpl};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, MaxBase2k, Module, ScratchOwned, Standard},
};

use crate::{
    BenchOp, bench_ops,
    blas::{
        matmul::{self, BlasBenchModule},
        params::{BlasBenchParams, default_bench_params_blas},
    },
};

/// `blas_matmul` by a dense right factor of distinct matrices per matrix slot, kept apart from [`blas_ops`] because it
/// needs several matrices per ciphertext: at one matrix the right factor is the replicated one.
pub fn blas_dense_ops<BE, M>() -> [BenchOp<M, BlasBenchParams>; 1]
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    [BenchOp {
        layer: "blas",
        name: "blas_matmul",
        runner: matmul::runner_blas_matmul::<BE, _>,
    }]
}

/// The operations that run at every point: `blas_matmul` by a replicated right factor, the products by a public matrix
/// on either side under `Diagonal` and `Jkls`, and the transpose.
pub fn blas_ops<BE, M>() -> [BenchOp<M, BlasBenchParams>; 6]
where
    BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    [
        BenchOp {
            layer: "blas",
            name: "blas_matmul_replicated",
            runner: matmul::runner_blas_matmul_replicated::<BE, _>,
        },
        BenchOp {
            layer: "blas",
            name: "blas_matmul_pt_right_diagonal",
            runner: matmul::runner_blas_matmul_pt_right_diagonal::<BE, _>,
        },
        BenchOp {
            layer: "blas",
            name: "blas_matmul_pt_right_jkls",
            runner: matmul::runner_blas_matmul_pt_right_jkls::<BE, _>,
        },
        BenchOp {
            layer: "blas",
            name: "blas_matmul_pt_left_diagonal",
            runner: matmul::runner_blas_matmul_pt_left_diagonal::<BE, _>,
        },
        BenchOp {
            layer: "blas",
            name: "blas_matmul_pt_left_jkls",
            runner: matmul::runner_blas_matmul_pt_left_jkls::<BE, _>,
        },
        BenchOp {
            layer: "blas",
            name: "blas_transpose",
            runner: matmul::runner_blas_transpose::<BE, _>,
        },
    ]
}

/// Every operation over `points`, the dense product only where a ciphertext holds several matrices.
fn bench_points<BE>(c: &mut Criterion<WallTime>, points: Vec<BlasBenchParams>)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    bench_ops(
        PhantomData::<BE>,
        &blas_dense_ops::<BE, WallTime>(),
        points.iter().copied().filter(|p| p.batch() > 1),
        c,
    );
    bench_ops(PhantomData::<BE>, &blas_ops::<BE, WallTime>(), points, c);
}

/// Every BLAS operation over the full sweep, the full tier. Takes exactly one backend type parameter; a
/// `criterion_group!` registers it once per backend to cover.
pub fn bench_blas<BE>(c: &mut Criterion<WallTime>)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    bench_points::<BE>(c, default_bench_params_blas());
}

/// `standard` tier: every BLAS operation at `n` 2^13, 2^14 and 2^15, with the block dimension `d = n / 256`.
pub mod standard {
    use criterion::{Criterion, measurement::WallTime};
    use poulpy_blas::oep::{BlasMatIntraImpl, BlasMatMulImpl};
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{Backend, MaxBase2k, Module, ScratchOwned, Standard},
    };

    use super::{BlasBenchModule, bench_points};
    use crate::{blas::params::default_bench_params_blas, is_standard_n};

    /// `where` clause matches [`super::bench_blas`]'s own.
    pub fn bench_blas<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
        Module<BE>: BlasBenchModule<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let points = default_bench_params_blas()
            .into_iter()
            .filter(|p| is_standard_n(p.n as u64) && p.n == 256 * p.d())
            .collect();
        bench_points::<BE>(c, points);
    }
}

/// `light` tier: every BLAS operation at the single size `n` = 2^14.
pub mod light {
    use criterion::{Criterion, measurement::WallTime};
    use poulpy_blas::oep::{BlasMatIntraImpl, BlasMatMulImpl};
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{Backend, MaxBase2k, Module, ScratchOwned, Standard},
    };

    use super::{BlasBenchModule, bench_points};
    use crate::{blas::params::default_bench_params_blas, is_light_n};

    /// `where` clause matches [`super::bench_blas`]'s own.
    pub fn bench_blas<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
        Module<BE>: BlasBenchModule<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let points = default_bench_params_blas()
            .into_iter()
            .filter(|p| is_light_n(p.n as u64))
            .collect();
        bench_points::<BE>(c, points);
    }
}
