use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_core::{
    LWEMatrixMul,
    layouts::{Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, LWEMatrix, LWEMatrixLayout, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
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
