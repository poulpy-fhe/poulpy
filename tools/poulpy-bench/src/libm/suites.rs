use std::marker::PhantomData;

use criterion::{
    Criterion,
    measurement::{Measurement, WallTime},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostStaged, Module, ScratchOwned, Standard},
};

use crate::{
    BenchOp, bench_ops,
    libm::{
        functions::{self, LibmBenchModule},
        params::{LibmBenchParams, default_bench_params_libm},
    },
};

/// `ckks_exp`, `ckks_log` and `ckks_pow`.
pub fn libm_ops<BE, M>() -> [BenchOp<M, LibmBenchParams>; 3]
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    [
        BenchOp {
            layer: "libm",
            name: "ckks_exp",
            runner: functions::runner_ckks_exp::<BE, _>,
        },
        BenchOp {
            layer: "libm",
            name: "ckks_log",
            runner: functions::runner_ckks_log::<BE, _>,
        },
        BenchOp {
            layer: "libm",
            name: "ckks_pow",
            runner: functions::runner_ckks_pow::<BE, _>,
        },
    ]
}

/// Every libm operation over the full sweep, the full tier. Takes exactly one backend type parameter, and
/// a `criterion_group!` registers it once per backend to cover.
pub fn bench_libm<BE>(c: &mut Criterion<WallTime>)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    bench_ops(PhantomData::<BE>, &libm_ops::<BE, WallTime>(), default_bench_params_libm(), c);
}

/// `standard` tier: every libm operation at `n` 2^13, 2^14 and 2^15.
pub mod standard {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{Backend, HostStaged, Module, ScratchOwned, Standard},
    };

    use super::libm_ops;
    use crate::{
        bench_ops, is_standard_n,
        libm::{functions::LibmBenchModule, params::default_bench_params_libm},
    };

    /// Every libm operation over the standard degrees.
    pub fn bench_libm<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<Ring = Standard> + HostStaged + 'static,
        Module<BE>: LibmBenchModule<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let points = default_bench_params_libm().into_iter().filter(|p| is_standard_n(p.n as u64));
        bench_ops(PhantomData::<BE>, &libm_ops::<BE, WallTime>(), points, c);
    }
}

/// `light` tier: every libm operation at the single size `n` = 2^14.
pub mod light {
    use std::marker::PhantomData;

    use criterion::{Criterion, measurement::WallTime};
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{Backend, HostStaged, Module, ScratchOwned, Standard},
    };

    use super::libm_ops;
    use crate::{
        bench_ops, is_light_n,
        libm::{functions::LibmBenchModule, params::default_bench_params_libm},
    };

    /// Every libm operation at the light degree.
    pub fn bench_libm<BE>(c: &mut Criterion<WallTime>)
    where
        BE: Backend<Ring = Standard> + HostStaged + 'static,
        Module<BE>: LibmBenchModule<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let points = default_bench_params_libm().into_iter().filter(|p| is_light_n(p.n as u64));
        bench_ops(PhantomData::<BE>, &libm_ops::<BE, WallTime>(), points, c);
    }
}
