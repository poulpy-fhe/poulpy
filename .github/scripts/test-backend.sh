#!/usr/bin/env bash
# Compile separately, then run a backend's native or bounded emulation suite.
set -euo pipefail

stage=${1:?expected build or test}
backend=${2:?expected avx or avx512}
mode=${3:?expected native or emulated}
# Optional `blas`: run only the BLAS tests AVX-512 emulation keeps, with a budget of their own.
group=${4:-}
case "$stage" in build|test) ;; *) echo "Invalid stage: $stage" >&2; exit 2 ;; esac
case "$group" in ''|blas) ;; *) echo "Invalid test group: $group" >&2; exit 2 ;; esac
case "$backend" in
  avx) features=enable-avx,enable-rayon ;;
  avx512) features=enable-ifma,enable-rayon ;;
  *) echo "Invalid backend: $backend" >&2; exit 2 ;;
esac
case "$mode" in
  native) features+=,enable-ckks,enable-bin-fhe,enable-blas; unset POULPY_TEST_EMULATED ;;
  emulated)
    export POULPY_TEST_EMULATED=1
    # One AVX-512 binary serves the contracts and the `blas` group.
    if [[ "$backend" == avx512 ]]; then features+=,enable-ckks,enable-blas; fi ;;
  *) echo "Invalid execution mode: $mode" >&2; exit 2 ;;
esac

command=(cargo test -p "poulpy-cpu-$backend" --lib --profile ci --features "$features")
if [[ "$stage" == build ]]; then
  "${command[@]}" --no-run
  exit
fi

filters=()
# Native AVX retains its full unit suite. AVX-512 retains all existing contracts.
if [[ "$group" == blas ]]; then
  # Each AVX-512/IFMA type once, through its f64 parity; native runs every BLAS test with the contracts.
  filters=(blas_parity --skip _quad --skip _rank2 --skip _base2k54 --skip _n32)
elif [[ "$backend" == avx512 || "$mode" == emulated ]]; then
  filters=(test_vec_znx test_svp test_vmp test_convolution test_cnv test_word_compat
    test_transfer raw_transform_matches_contract core_parity core_encryption glwe_copy)
  if [[ "$mode" == native ]]; then
    filters+=(ckks_parity bin_fhe_parity blas_tests prepared_tensor_tests finish_matches_composition_edges determinism
      consume_parity one_pass_parity relinearize_tests)
  else
    filters+=(--skip ::ntt_n)
    if [[ "$backend" == avx512 ]]; then
      # Large-ring kernel paths have focused cases; exhaustive sweeps remain native.
      filters+=(core_emulated_tensor one_pass_parity::ifma_fallback one_pass_parity::ifma_rayon_fallback
        one_pass_parity::ntt4x30_fallback one_pass_parity::ntt4x30_rayon_fallback
        --skip core_parity_ntt4x30_fused --skip core_parity_ntt4x30_rayon_fused
        --skip core_parity_ntt3x42_ifma_fused --skip core_parity_ntt3x42_ifma_rayon_fused
        --skip ::vec_znx_dft_large --skip _streaming)
    fi
  fi
fi

echo "Running $backend $mode ${group:+$group }tests (400-second execution budget; compilation is separate)."
if timeout --signal=TERM --kill-after=10s 400s "${command[@]}" -- "${filters[@]}" \
  --test-threads=2 -Z unstable-options --report-time; then
  exit 0
else
  status=$?
  if [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "::error::Backend tests exceeded the execution budget or were forcibly terminated."
  fi
  exit "$status"
fi
