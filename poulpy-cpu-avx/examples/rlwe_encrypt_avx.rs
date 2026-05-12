use itertools::izip;

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;
#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxAddNormalSourceBackend,
        VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateAssign,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxFillUniformSourceBackend, VecZnxIdftApplyTmpA, VecZnxNormalizeAssignBackend,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnx, ScalarZnxToBackendRef, ScratchOwned, SvpPPolToBackendMut, SvpPPolToBackendRef,
        VecZnx, VecZnxBig, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
    source::Source,
};

fn main() {
    let n: usize = 16;
    let base2k: usize = 18;
    let ct_size: usize = 3;
    let msg_size: usize = 2;
    let log_scale: usize = msg_size * base2k - 5;
    let noise_infos = NoiseInfos::new(base2k * ct_size, 3.2, 6.0 * 3.2).unwrap();
    let module = Module::<BackendImpl>::new(n as u64);

    let mut scratch = ScratchOwned::<BackendImpl>::alloc(module.vec_znx_big_normalize_tmp_bytes());

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    // s <- Z_{-1, 0, 1}[X]/(X^{N}+1)
    let mut s: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    s.fill_ternary_prob(0, 0.5, &mut source);

    // Buffer to store s in the DFT domain
    let mut s_dft = module.svp_ppol_alloc(s.cols());

    // s_dft <- DFT(s)
    module.svp_prepare(
        &mut s_dft.to_backend_mut(),
        0,
        &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<BackendImpl>>::to_backend_ref(&s),
        0,
    );

    // Allocates a VecZnx with two columns: ct=(0, 0)
    let mut ct: VecZnx<Vec<u8>> = module.vec_znx_alloc(
        2,       // Number of columns
        ct_size, // Number of small poly per column
    );

    // Fill the second column with random values: ct = (0, a)
    module.vec_znx_fill_uniform_source_backend(
        base2k,
        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(&mut ct),
        1,
        &mut source,
    );

    let mut buf_dft: VecZnxDft<<BackendImpl as Backend>::OwnedBuf, BackendImpl> = module.vec_znx_dft_alloc(1, ct_size);

    let ct_backend = <VecZnx<Vec<u8>> as VecZnxToBackendRef<BackendImpl>>::to_backend_ref(&ct);
    module.vec_znx_dft_apply(1, 0, &mut buf_dft.to_backend_mut(), 0, &ct_backend, 1);

    // Applies DFT(ct[1]) * DFT(s)
    module.svp_apply_dft_to_dft_assign(
        &mut buf_dft.to_backend_mut(), // DFT(ct[1] * s)
        0,                             // Selects the first column of res
        &s_dft.to_backend_ref(),       // DFT(s)
        0,                             // Selects the first column of s_dft
    );

    // Alias scratch space (VecZnxDft<B> is always at least as big as VecZnxBig<B>)

    // BIG(ct[1] * s) <- IDFT(DFT(ct[1] * s)) (not normalized)
    let mut buf_big: VecZnxBig<<BackendImpl as Backend>::OwnedBuf, BackendImpl> = module.vec_znx_big_alloc(1, ct_size);
    module.vec_znx_idft_apply_tmpa(&mut buf_big.to_backend_mut(), 0, &mut buf_dft.to_backend_mut(), 0);

    // Creates a plaintext: VecZnx with 1 column
    let mut m = module.vec_znx_alloc(
        1,        // Number of columns
        msg_size, // Number of small polynomials
    );
    let mut want: Vec<i64> = vec![0; n];
    want.iter_mut().for_each(|x| *x = source.next_u64n(16, 15) as i64);
    m.encode_vec_i64(base2k, 0, log_scale, &want);
    module.vec_znx_normalize_assign_backend(
        base2k,
        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(&mut m),
        0,
        &mut scratch.borrow(),
    );

    // m - BIG(ct[1] * s)
    module.vec_znx_big_sub_small_negate_assign(
        &mut buf_big.to_backend_mut(),
        0, // Selects the first column of the receiver
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BackendImpl>>::to_backend_ref(&m),
        0, // Selects the first column of the message
    );

    // Normalizes back to VecZnx
    // ct[0] <- m - BIG(c1 * s)
    module.vec_znx_big_normalize(
        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(&mut ct),
        base2k,
        0,
        0, // Selects the first column of ct (ct[0])
        &<VecZnxBig<<BackendImpl as Backend>::OwnedBuf, BackendImpl> as VecZnxBigToBackendRef<BackendImpl>>::to_backend_ref(
            &buf_big,
        ),
        base2k,
        0, // Selects the first column of buf_big
        &mut scratch.borrow(),
    );

    // Add noise to ct[0]
    // ct[0] <- ct[0] + e
    module.vec_znx_add_normal_source_backend(
        base2k,
        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(&mut ct),
        0, // Selects the first column of ct (ct[0])
        noise_infos,
        &mut source,
    );

    // Final ciphertext: ct = (-a * s + m + e, a)

    // Decryption

    // DFT(ct[1] * s)
    let ct_backend = <VecZnx<Vec<u8>> as VecZnxToBackendRef<BackendImpl>>::to_backend_ref(&ct);
    module.vec_znx_dft_apply(1, 0, &mut buf_dft.to_backend_mut(), 0, &ct_backend, 1);
    module.svp_apply_dft_to_dft_assign(
        &mut buf_dft.to_backend_mut(),
        0, // Selects the first column of res.
        &s_dft.to_backend_ref(),
        0,
    );

    // BIG(c1 * s) = IDFT(DFT(c1 * s))
    module.vec_znx_idft_apply_tmpa(&mut buf_big.to_backend_mut(), 0, &mut buf_dft.to_backend_mut(), 0);

    // BIG(c1 * s) + ct[0]
    module.vec_znx_big_add_small_assign(
        &mut buf_big.to_backend_mut(),
        0,
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BackendImpl>>::to_backend_ref(&ct),
        0,
    );

    // m + e <- BIG(ct[1] * s + ct[0])
    let mut res = module.vec_znx_alloc(1, ct_size);
    module.vec_znx_big_normalize(
        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(&mut res),
        base2k,
        0,
        0,
        &<VecZnxBig<<BackendImpl as Backend>::OwnedBuf, BackendImpl> as VecZnxBigToBackendRef<BackendImpl>>::to_backend_ref(
            &buf_big,
        ),
        base2k,
        0,
        &mut scratch.borrow(),
    );

    // have = m * 2^{log_scale} + e
    let mut have: Vec<i64> = vec![i64::default(); n];
    res.decode_vec_i64(base2k, 0, ct_size * base2k, &mut have);
    let scale: f64 = (1 << (res.size() * base2k - log_scale)) as f64;
    izip!(want.iter(), have.iter()).enumerate().for_each(|(i, (a, b))| {
        println!("{}: {} {}", i, a, (*b as f64) / scale);
    });
}
