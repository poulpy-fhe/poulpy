use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_core::{
    CoeffMatrixPrepare, LWEMatrixMul,
    layouts::{Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, LWEMatrix, LWEMatrixLayout, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc},
    layouts::{Backend, Module, ScratchOwned, VecZnx, ZnxViewMut},
    source::Source,
};

pub fn bench_coeff_matmul<BE>(params: &crate::params::CoeffMatSweepParams, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + LWEMatrixMul<BE>,
    BE: Backend<OwnedBuf = Vec<u8>>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name = format!("coeff_matmul::{label}");
    let mut group = c.benchmark_group(group_name);

    fn fill_vec_znx(v: &mut VecZnx<Vec<u8>>, source: &mut Source, mask: i64) {
        for x in v.raw_mut() {
            *x = source.next_i64() & mask;
        }
    }

    fn runner<BE>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + LWEMatrixMul<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let module: Module<BE> = Module::<BE>::new(1 << sweep[0]);

        let rows_in = sweep[1];
        let lwe_n = sweep[2];
        let rows_out = sweep[3];
        let size = sweep[4];
        let base2k = Base2K(12);
        let k = TorusPrecision((base2k.0 as usize * size) as u32);

        assert!(rows_in <= module.n(), "coeff_mat benchmark rows_in exceeds N");
        assert!(rows_out <= module.n(), "coeff_mat benchmark rows_out exceeds N");

        let u_infos = CoeffMatrixLayout {
            n: Degree(rows_in as u32),
            rows_out,
            base2k,
            k,
        };
        let a_infos = LWEMatrixLayout {
            rows: rows_in,
            n: Degree(lwe_n as u32),
            base2k,
            k,
        };
        let res_infos = LWEMatrixLayout {
            rows: rows_out,
            n: Degree(lwe_n as u32),
            base2k,
            k,
        };

        let mut source = Source::new([0u8; 32]);
        let mut scratch = ScratchOwned::alloc(module.lwe_matrix_mul_tmp_bytes(&res_infos, &u_infos, &a_infos));

        let mut u: CoeffMatrix<Vec<u8>> = module.coeff_matrix_alloc_from_infos(&u_infos);
        let mut a: LWEMatrix<Vec<u8>> = module.lwe_matrix_alloc_from_infos(&a_infos);
        let mut res: LWEMatrix<Vec<u8>> = module.lwe_matrix_alloc_from_infos(&res_infos);

        let mask = (1_i64 << base2k.0) - 1;
        fill_vec_znx(u.data_mut(), &mut source, mask);
        fill_vec_znx(a.body_mut(), &mut source, mask);
        fill_vec_znx(a.mask_mut(), &mut source, mask);

        move || {
            module.lwe_matrix_mul(&mut res, &u, &a, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "N={} U={}x{} LWEMatrix=({}x{}, {}x1) size={}",
            1 << sweep[0],
            sweep[3],
            sweep[1],
            sweep[1],
            sweep[2],
            sweep[1],
            sweep[4]
        ));
        let mut runner = runner::<BE>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

/// Hot-path comparison for the "fixed `U`, variable bodies" workload: each
/// iteration represents one batch of `K` bodies, comparing
///
/// - `batched_unprepared`: a single `lwe_matrix_mul_bodies(cols = K)` call
///   (matmul packs K bodies, but the `U` panel is re-built every call).
/// - `batched_prepared`:   a single `lwe_matrix_mul_bodies_prepared(cols = K)`
///   call against an externally prepared `U` (panel built once, amortized
///   across all batches).
///
/// `U` shape and bodies shape come from the same `coeff_mat` sweep params;
/// `K = sweep[2]` here re-uses the `lwe_n` slot as the per-batch column count
/// so we don't need a separate config struct.
pub fn bench_coeff_matmul_bodies<BE>(params: &crate::params::CoeffMatSweepParams, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + LWEMatrixMul<BE> + CoeffMatrixPrepare<BE> + VecZnxAlloc<BE>,
    BE: Backend<OwnedBuf = Vec<u8>>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name = format!("coeff_matmul_bodies::{label}");
    let mut group = c.benchmark_group(group_name);

    fn fill_vec_znx(v: &mut VecZnx<Vec<u8>>, source: &mut Source, mask: i64) {
        for x in v.raw_mut() {
            *x = source.next_i64() & mask;
        }
    }

    for sweep in &params.sweeps {
        let log_n = sweep[0];
        let rows_in = sweep[1];
        let num_bodies = sweep[2].max(1);
        let rows_out = sweep[3];
        let size = sweep[4];

        let module: Module<BE> = Module::<BE>::new(1 << log_n);
        let base2k = Base2K(12);
        let k_prec = TorusPrecision((base2k.0 as usize * size) as u32);
        let bk = base2k.0 as usize;

        assert!(rows_in <= module.n(), "coeff_mat bodies bench rows_in exceeds N");
        assert!(rows_out <= module.n(), "coeff_mat bodies bench rows_out exceeds N");

        let u_infos = CoeffMatrixLayout {
            n: Degree(rows_in as u32),
            rows_out,
            base2k,
            k: k_prec,
        };

        let mut source = Source::new([0u8; 32]);
        let mut u: CoeffMatrix<Vec<u8>> = module.coeff_matrix_alloc_from_infos(&u_infos);
        let mask = (1_i64 << base2k.0) - 1;
        fill_vec_znx(u.data_mut(), &mut source, mask);

        let mut bodies: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
        for x in bodies.raw_mut() {
            *x = source.next_i64() & mask;
        }

        let id_label = format!("N={} U={}x{} K={} size={}", 1 << log_n, rows_out, rows_in, num_bodies, size);

        // Batched, `U` re-prepared every iteration.
        {
            let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
            let mut scratch =
                ScratchOwned::alloc(module.lwe_matrix_mul_bodies_tmp_bytes(&u_infos, num_bodies, size, size));
            group.bench_with_input(BenchmarkId::new("batched_unprepared", &id_label), &(), |b, _| {
                b.iter(|| {
                    module.lwe_matrix_mul_bodies(&mut res, bk, &u, &bodies, bk, &mut scratch.borrow());
                    black_box(());
                })
            });
        }

        // Batched, `U` prepared once before the loop (hot-path amortized prep).
        {
            let prepared = module.coeff_matrix_prepare(&u);
            let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
            let mut scratch = ScratchOwned::alloc(module.lwe_matrix_mul_bodies_prepared_tmp_bytes(
                &prepared, num_bodies, size, size,
            ));
            group.bench_with_input(BenchmarkId::new("batched_prepared", &id_label), &(), |b, _| {
                b.iter(|| {
                    module
                        .lwe_matrix_mul_bodies_prepared(&mut res, bk, &prepared, &bodies, bk, &mut scratch.borrow());
                    black_box(());
                })
            });
        }
    }

    group.finish();
}
