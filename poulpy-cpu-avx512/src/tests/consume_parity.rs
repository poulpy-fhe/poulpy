use poulpy_hal::{
    api::{ScratchOwnedAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes},
    layouts::{DataView, HostBytesBackend, Module, ScratchOwned, VecZnxDftToBackendMut, ZnxViewMut},
    test_suite::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

macro_rules! consume_case {
    ($name:ident, $parallel:ty, $serial:ty, $oracle:ty) => {
        #[test]
        fn $name() {
            rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap()
                .install(|| {
                    for n in [256usize, 65536] {
                        let host = Module::<HostBytesBackend>::new((2 * n) as u64);
                        let parallel = Module::<$parallel>::new((2 * n) as u64);
                        let serial = Module::<$serial>::new((2 * n) as u64);
                        let oracle = Module::<$oracle>::new((2 * n) as u64);
                        let base = 52usize;
                        let mut input = host.vec_znx_alloc(n, 2, 5);
                        let mut add = host.vec_znx_alloc(n, 2, 3);
                        let edge = [
                            0,
                            1,
                            -1,
                            (1i64 << 51) - 1,
                            -(1i64 << 51),
                            (1i64 << 51) - 2,
                            -(1i64 << 51) + 1,
                        ];
                        for limb in 0..5 {
                            for col in 0..2 {
                                for (i, value) in input.at_mut(col, limb).iter_mut().enumerate() {
                                    *value = edge[(i + 3 * limb + col) % edge.len()];
                                }
                            }
                        }
                        for limb in 0..3 {
                            for (i, value) in add.at_mut(0, limb).iter_mut().enumerate() {
                                *value = edge[(i + limb) % edge.len()];
                            }
                        }
                        for offset in [-53, -1, 0, 1, 53] {
                            for k in [0, 53, 259] {
                                for with_add in [false, true] {
                                    macro_rules! run {
                                        ($be:ty, $module:ident, $bytes:expr) => {{
                                            let a = upload_vec_znx::<$be>(&input);
                                            let add = upload_vec_znx::<$be>(&add);
                                            let mut dft = $module.vec_znx_dft_alloc(n, 2, 5);
                                            for col in 0..2 {
                                                $module.vec_znx_dft_apply(
                                                    1,
                                                    0,
                                                    &mut dft.to_backend_mut(),
                                                    col,
                                                    &vec_znx_backend_ref::<$be>(&a),
                                                    col,
                                                );
                                            }
                                            let before = dft.data().to_vec();
                                            let mut res = host.vec_znx_alloc(n, 2, 6);
                                            res.data_mut().fill(0xa5);
                                            let mut res = upload_vec_znx::<$be>(&res);
                                            let add_ref = vec_znx_backend_ref::<$be>(&add);
                                            let mut scratch = ScratchOwned::<$be>::alloc($bytes);
                                            $module.vec_znx_idft_normalize_consume(
                                                &mut vec_znx_backend_mut::<$be>(&mut res),
                                                49,
                                                k,
                                                offset,
                                                1,
                                                &mut dft.to_backend_mut(),
                                                1,
                                                base,
                                                with_add.then_some((&add_ref, 0)),
                                                &mut scratch.arena(),
                                            );
                                            for limb in 0..5 {
                                                let start = limb * 2 * n * 16;
                                                assert_eq!(
                                                    &dft.data()[start..start + n * 16],
                                                    &before[start..start + n * 16]
                                                );
                                            }
                                            download_vec_znx::<$be>(&res)
                                        }};
                                    }
                                    let expected = run!($oracle, oracle, oracle.vec_znx_idft_normalize_consume_tmp_bytes(6, 5));
                                    let serial_result =
                                        run!($serial, serial, serial.vec_znx_idft_normalize_consume_tmp_bytes(6, 5));
                                    assert_eq!(
                                        serial_result, expected,
                                        "serial n={n}, offset={offset}, k={k}, add={with_add}"
                                    );
                                    let declared = parallel.vec_znx_idft_normalize_consume_tmp_bytes(6, 5);
                                    assert_eq!(declared, (5 * 4 * 8 + 3 * 16) * parallel.n());
                                    for bytes in [4 * n * 8 + 3 * n * 16, 2 * 4 * n * 8 + 3 * n * 16, declared] {
                                        let actual = run!($parallel, parallel, bytes);
                                        assert_eq!(
                                            actual, expected,
                                            "parallel n={n}, offset={offset}, k={k}, add={with_add}, bytes={bytes}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                });
        }
        paste::paste! {
            #[test]
            fn [<$name _rejects_invalid_column>]() {
                let module = Module::<$parallel>::new(256);
                for col in [2, usize::MAX] {
                    let mut dft = module.vec_znx_dft_alloc(256, 2, 2);
                    let host = Module::<HostBytesBackend>::new(256);
                    let res = host.vec_znx_alloc(256, 1, 2);
                    let mut res = upload_vec_znx::<$parallel>(&res);
                    let mut scratch = ScratchOwned::<$parallel>::alloc(module.vec_znx_idft_normalize_consume_tmp_bytes(2, 2));
                    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        module.vec_znx_idft_normalize_consume(
                            &mut vec_znx_backend_mut::<$parallel>(&mut res), 52, 104, 0, 0,
                            &mut dft.to_backend_mut(), col, 52, None, &mut scratch.arena(),
                        );
                    })).unwrap_err();
                    assert_eq!(panic.downcast_ref::<&str>(), Some(&"input column out of bounds"));
                }
            }
        }
    };
}

consume_case!(
    avx,
    poulpy_cpu_avx::NTT4x30AvxRayon,
    poulpy_cpu_avx::NTT4x30Avx,
    poulpy_cpu_oracle::NTT4x30Oracle
);
consume_case!(
    avx_ci,
    poulpy_cpu_avx::NTT4x30CIAvxRayon,
    poulpy_cpu_avx::NTT4x30CIAvx,
    poulpy_cpu_oracle::NTT4x30CIOracle
);
consume_case!(
    avx512,
    crate::NTT4x30Avx512Rayon,
    crate::NTT4x30Avx512,
    poulpy_cpu_oracle::NTT4x30Oracle
);
consume_case!(
    avx512_ci,
    crate::NTT4x30CIAvx512Rayon,
    crate::NTT4x30CIAvx512,
    poulpy_cpu_oracle::NTT4x30CIOracle
);
