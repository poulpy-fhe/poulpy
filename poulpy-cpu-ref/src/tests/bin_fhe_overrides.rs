//! Independent bin-FHE overrides and host staging with opaque storage.
use crate::hal_impl::delegating_backend::BinFheOverrideFFT64 as OverrideBackend;
use poulpy_bin_fhe::{api::*, blind_rotation::*, oep::*};
use poulpy_core::layouts::*;
use poulpy_hal::{api::*, layouts::*, oep::HalModuleImpl};
use std::{cell::Cell, ptr::NonNull};

crate::impl_sampling_host!(OverrideBackend, fft64);
poulpy_core::impl_conversion_reference_full!(OverrideBackend);
poulpy_core::impl_decryption_reference_full!(OverrideBackend);
poulpy_core::impl_encryption_reference_full!(OverrideBackend);
poulpy_core::impl_ggsw_rotate_derived_full!(OverrideBackend);
poulpy_core::impl_glwe_mul_const_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_mul_plain_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_add_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_sub_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_negate_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_zero_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_rotate_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_shift_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_normalize_reference_full!(OverrideBackend);
poulpy_core::impl_polynomial_evaluation_derived_full!(OverrideBackend);
poulpy_core::impl_gglwe_external_product_derived_full!(OverrideBackend);
poulpy_core::impl_gglwe_keyswitch_derived_full!(OverrideBackend);
poulpy_core::impl_ggsw_external_product_derived_full!(OverrideBackend);
poulpy_core::impl_ggsw_keyswitch_derived_full!(OverrideBackend);
poulpy_core::impl_automorphism_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_external_product_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_keyswitch_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_packing_derived_full!(OverrideBackend);
poulpy_core::impl_glwe_trace_derived_full!(OverrideBackend);
poulpy_core::impl_linear_transformation_reference_full!(OverrideBackend);
poulpy_core::impl_lwe_keyswitch_reference_full!(OverrideBackend);
poulpy_core::impl_glwe_tensoring_reference!(OverrideBackend);
poulpy_core::impl_gglwe_product_digits_strided_reference!(OverrideBackend);
const COPY_PER_LIMB: usize = 4096;
fn copy_override_bytes<R: GLWEInfos, A: GLWEInfos>(res: &R, source: &A) -> usize {
    // Non-monotonic source costs detect queries that substitute a supposedly
    // worst-case precision or ignore the source layout altogether.
    COPY_PER_LIMB * res.max_size()
        + usize::from(source.max_size() == 5) * 65_536
        + usize::from(source.base2k().as_usize() == 10) * 32_768
}
unsafe impl poulpy_core::oep::GLWECopyImpl for OverrideBackend {
    fn glwe_copy_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<Self>, res: &R, a: &A) -> usize {
        use poulpy_core::reference::operations::GLWECopyReference;
        copy_override_bytes(res, a) + module.glwe_copy_tmp_bytes_reference(res, a)
    }
    fn glwe_copy<R: GLWEToBackendMut<Self>, A: GLWEToBackendRef<Self>>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        use poulpy_core::reference::operations::GLWECopyReference;
        let bytes = copy_override_bytes(&res.to_backend_mut(), &a.to_backend_ref());
        let (mut marker, mut rest) = scratch.borrow().take_region(bytes);
        Self::copy_host_to_view(&mut marker, &vec![0x39; bytes]);
        module.glwe_copy_reference(res, a, &mut rest);
        assert!(marker.iter().all(|&x| x == 0x39));
    }
}
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_key_encrypt_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_key_compressed_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_key_decompress_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_key_compressed_factory_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_lookup_table_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_circuit_bootstrapping_key_encrypt_sk_reference!(OverrideBackend, CGGI);
poulpy_bin_fhe::impl_bin_fhe_circuit_bootstrapping_key_prepared_reference!(OverrideBackend, CGGI);
poulpy_bin_fhe::impl_bin_fhe_bdd_reference!(OverrideBackend, CGGI);

thread_local! {
    static EXECUTION_CALLS: Cell<usize> = const { Cell::new(0) };
    static QUERY_CALLS: Cell<usize> = const { Cell::new(0) };
    static STAGING_CALLS: Cell<usize> = const { Cell::new(0) };
}
const EXTRA: usize = 256;
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_key_prepared_reference!(OverrideBackend);
poulpy_bin_fhe::impl_bin_fhe_blind_rotation_mod_switch_reference!(OverrideBackend);
unsafe impl BlindRotationExecuteImpl<CGGI> for OverrideBackend {
    fn blind_rotation_execute_tmp_bytes<G: GLWEInfos, K: BlindRotationKeyInfos>(
        module: &Module<Self>,
        block_size: usize,
        extension: usize,
        output: &G,
        key: &K,
    ) -> usize {
        QUERY_CALLS.set(QUERY_CALLS.get() + 1);
        EXTRA
            + poulpy_bin_fhe::reference::blind_rotation::blind_rotation_execute_tmp_bytes_ref(
                module, block_size, extension, output, key,
            )
    }
    fn blind_rotation_execute<R: GLWEToBackendMut<Self> + GLWEInfos, L: LWEToBackendRef<Self> + LWEInfos>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<Self::OwnedBuf, i64>,
        key: &BlindRotationKeyPrepared<Self::OwnedBuf, CGGI, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        EXECUTION_CALLS.set(EXECUTION_CALLS.get() + 1);
        let (mut marker, mut rest) = scratch.borrow().take_region(EXTRA);
        Self::copy_host_to_view(&mut marker, &[0x5A; EXTRA]);
        poulpy_bin_fhe::reference::blind_rotation::blind_rotation_execute_ref(module, res, lwe, lut, key, &mut rest);
        let mut marker_after = [0u8; EXTRA];
        Self::copy_view_to_host(&Self::view_ref_mut(&marker), &mut marker_after);
        assert_eq!(marker_after, [0x5A; EXTRA]);
    }
}

#[test]
fn independent_blind_rotation_override_and_query_preserve_parity() {
    let reference = Module::<crate::FFT64Ref>::new(32);
    let tested = Module::<OverrideBackend>::new(32);
    EXECUTION_CALLS.set(0);
    QUERY_CALLS.set(0);
    poulpy_bin_fhe::test_suite::parity::blind_rotation::test_blind_rotation_parity(&reference, &tested);
    assert!(EXECUTION_CALLS.get() > 0);
    assert_eq!(EXECUTION_CALLS.get(), QUERY_CALLS.get());
    // Calling the reference query never calls the backend's selected query.
    let layout = BlindRotationKeyLayout {
        n_glwe: 32usize.into(),
        n_lwe: 4usize.into(),
        base2k: 12usize.into(),
        dnum: 2usize.into(),
        k_aux: 12usize.into(),
        rank: 1usize.into(),
    };
    let before = QUERY_CALLS.get();
    let plain = poulpy_bin_fhe::reference::blind_rotation::blind_rotation_execute_tmp_bytes_ref(&tested, 1, 1, &layout, &layout);
    assert_eq!(QUERY_CALLS.get(), before);
    assert_eq!(tested.blind_rotation_execute_tmp_bytes(1, 1, &layout, &layout), plain + EXTRA);
}

// Deliberately no AsRef/AsMut implementations: neither owned buffers nor views
// satisfy HostDataRef or HostDataMut. Access is possible only through transfers.
#[derive(Default, PartialEq, Eq)]
struct OpaqueOwned(poulpy_hal::AlignedBuf);
impl CopyToHost for OpaqueOwned {
    fn len_bytes(&self) -> usize {
        self.0.len()
    }
    fn copy_to_host(&self, dst: &mut [u8]) {
        dst.copy_from_slice(&self.0);
    }
}
impl CopyFromHost for OpaqueOwned {
    fn len_bytes(&self) -> usize {
        self.0.len()
    }
    fn copy_from_host(&mut self, src: &[u8]) {
        self.0.copy_from_slice(src);
    }
}
#[derive(Default, PartialEq, Eq)]
struct OpaqueRef<'a>(&'a [u8]);
#[derive(Default, PartialEq, Eq)]
struct OpaqueMut<'a>(&'a mut [u8]);
#[derive(Default, PartialEq, Eq)]
struct OpaqueBackend;
impl Backend for OpaqueBackend {
    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type ZnxWord = i64;
    type BigWord = i64;
    type DftWord = f64;
    type OwnedBuf = OpaqueOwned;
    type BufRef<'a> = OpaqueRef<'a>;
    type BufMut<'a> = OpaqueMut<'a>;
    type Handle = ();
    type Location = Device;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        OpaqueOwned(poulpy_hal::alloc_aligned(len))
    }
    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        OpaqueOwned(poulpy_hal::AlignedBuf::from(bytes))
    }
    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.0.to_vec()
    }
    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        dst.copy_from_slice(&buf.0[..dst.len()]);
    }
    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        buf.0[..src.len()].copy_from_slice(src);
        buf.0[src.len()..].fill(0);
    }
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        dst.copy_from_slice(&buf.0[..dst.len()]);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
        buf.0[..src.len()].copy_from_slice(src);
        buf.0[src.len()..].fill(0);
    }
    fn len_bytes(buf: &Self::OwnedBuf) -> usize {
        buf.0.len()
    }
    fn len_bytes_ref(buf: &Self::BufRef<'_>) -> usize {
        buf.0.len()
    }
    fn len_bytes_mut(buf: &Self::BufMut<'_>) -> usize {
        buf.0.len()
    }
    fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
        OpaqueRef(&buf.0)
    }
    fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
        OpaqueMut(&mut buf.0)
    }
    fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        OpaqueRef(buf.0)
    }
    fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        OpaqueRef(buf.0)
    }
    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        OpaqueMut(buf.0)
    }
    fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
        OpaqueRef(&buf.0[offset..offset + len])
    }
    fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
        OpaqueMut(&mut buf.0[offset..offset + len])
    }
    fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        OpaqueRef(&buf.0[offset..offset + len])
    }
    fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        OpaqueRef(&buf.0[offset..offset + len])
    }
    fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        OpaqueMut(&mut buf.0[offset..offset + len])
    }
    unsafe fn destroy(_: NonNull<()>) {}
}
unsafe impl HalModuleImpl for OpaqueBackend {
    fn new(n: u64) -> Module<Self> {
        unsafe { Module::from_nonnull(NonNull::dangling(), n) }
    }
}
unsafe impl BlindRotationModSwitchImpl for OpaqueBackend {
    fn blind_rotation_mod_switch<L: LWEToBackendRef<Self> + LWEInfos>(
        _module: &Module<Self>,
        modulus: usize,
        res: &mut [i64],
        lwe: &L,
        direction: LookUpTableRotationDirection,
    ) {
        STAGING_CALLS.set(STAGING_CALLS.get() + 1);
        poulpy_bin_fhe::reference::blind_rotation::mod_switch_2n_ref::<Self, _>(modulus, res, lwe, direction);
    }
}

#[test]
fn modulus_switch_accepts_opaque_storage_and_dispatches_override() {
    fn requires_parity_backend<B: poulpy_bin_fhe::test_suite::parity::ParityBackend>() {}
    requires_parity_backend::<OpaqueBackend>();
    let module = Module::<OpaqueBackend>::new(32);
    let layout = LWELayout {
        n: 4usize.into(),
        base2k: 4usize.into(),
        k: 12usize.into(),
    };
    let mut lwe = module.lwe_alloc_from_infos(&layout);
    let body = [1i64, 7, -1].into_iter().flat_map(i64::to_ne_bytes).collect::<Vec<_>>();
    let mask = [0i64, 1, -1, 2, 0, -8, 7, -2, 1, -1, 0, 0]
        .into_iter()
        .flat_map(i64::to_ne_bytes)
        .collect::<Vec<_>>();
    OpaqueBackend::copy_from_host(lwe.body_mut().data_mut(), &body);
    OpaqueBackend::copy_from_host(lwe.mask_mut().data_mut(), &mask);
    STAGING_CALLS.set(0);
    for direction in [LookUpTableRotationDirection::Left, LookUpTableRotationDirection::Right] {
        let mut expected = [0i64; 5];
        let mut actual = [0i64; 5];
        poulpy_bin_fhe::reference::blind_rotation::mod_switch_2n_ref::<OpaqueBackend, _>(64, &mut expected, &lwe, direction);
        module.blind_rotation_mod_switch(64, &mut actual, &lwe, direction);
        assert_eq!(actual, expected);
    }
    assert_eq!(STAGING_CALLS.get(), 2);
}

#[test]
fn constituent_copy_override_is_covered_by_scheme_budgets() {
    let reference = Module::<crate::FFT64Ref>::new(32);
    let tested = Module::<OverrideBackend>::new(32);
    poulpy_bin_fhe::test_suite::parity::circuit_bootstrapping::test_circuit_bootstrapping_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_cmux_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_cswap_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_glwe_blind_rotation_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_glwe_blind_selection_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_glwe_blind_retrieval_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_execute_bdd_circuit_parity(&reference, &tested);
    poulpy_bin_fhe::test_suite::parity::bdd::test_ggsw_blind_rotation_parity(&reference, &tested);
}

#[test]
fn blind_rotation_rejects_invalid_shapes_before_writes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let module = Module::<crate::FFT64Ref>::new(32);
    let layout = BlindRotationKeyLayout {
        n_glwe: 32usize.into(),
        n_lwe: 4usize.into(),
        base2k: 12usize.into(),
        dnum: 2usize.into(),
        k_aux: 12usize.into(),
        rank: 1usize.into(),
    };
    let mut key = module.blind_rotation_key_prepared_alloc(&layout);
    key.set_distribution(poulpy_core::Distribution::ZERO);
    let mut output = module.glwe_alloc_from_infos(&layout);
    output.data_mut().data_mut().fill(0x55);
    let before = output.data().data().to_vec();
    let lut_layout = LookUpTableLayout {
        n: 32usize.into(),
        extension_factor: 1,
        k: 12usize.into(),
        base2k: 12usize.into(),
    };
    let lut = LookupTable::alloc(&module, &lut_layout);
    for (n_lwe, distribution) in [
        (8usize, poulpy_core::Distribution::ZERO),
        (4, poulpy_core::Distribution::BinaryBlock(3)),
    ] {
        let lwe = module.lwe_alloc_from_infos(&LWELayout {
            n: n_lwe.into(),
            base2k: 12usize.into(),
            k: 24usize.into(),
        });
        key.set_distribution(distribution);
        let mut scratch = ScratchOwned::<crate::FFT64Ref>::alloc(module.blind_rotation_execute_tmp_bytes(1, 1, &layout, &layout));
        assert!(
            catch_unwind(AssertUnwindSafe(|| module.blind_rotation_execute(
                &mut output,
                &lwe,
                &lut,
                &key,
                &mut scratch.borrow()
            )))
            .is_err()
        );
        assert_eq!(
            output.data().data().as_slice(),
            before.as_slice(),
            "invalid shape mutated output"
        );
    }
}

const CIRCUIT_EXTRA: usize = 4096;
thread_local! { static PREPARED_CIRCUIT_CALLS: Cell<usize> = const { Cell::new(0) }; }
unsafe impl CircuitBootstrappingExecuteImpl<CGGI> for OverrideBackend {
    fn circuit_bootstrapping_prepare_to_constant<R: GGSWInfos>(
        module: &Module<Self>,
        res_infos: &R,
        key: &poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, CGGI, Self>,
        log_domain: usize,
        extension_factor: usize,
    ) -> poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
        poulpy_bin_fhe::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_constant_reference(
            module,
            res_infos,
            key,
            log_domain,
            extension_factor,
        )
    }
    fn circuit_bootstrapping_prepare_to_exponent<R: GGSWInfos>(
        module: &Module<Self>,
        log_gap_out: usize,
        res_infos: &R,
        key: &poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, CGGI, Self>,
        log_domain: usize,
        extension_factor: usize,
    ) -> poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
        poulpy_bin_fhe::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_exponent_reference(
            module,
            log_gap_out,
            res_infos,
            key,
            log_domain,
            extension_factor,
        )
    }
    fn circuit_bootstrapping_execute_prepared_tmp_bytes<
        A: poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
    >(
        module: &Module<Self>,
        plan: &poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingPlanLayout,
        key: &A,
    ) -> usize {
        CIRCUIT_EXTRA
            + poulpy_bin_fhe::reference::circuit_bootstrapping::circuit_bootstrapping_execute_prepared_tmp_bytes_reference::<
                _,
                _,
                CGGI,
                Self,
            >(module, plan, key)
    }
    fn circuit_bootstrapping_execute_prepared<R, L>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        key: &poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, CGGI, Self>,
        plan: &poulpy_bin_fhe::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos,
    {
        PREPARED_CIRCUIT_CALLS.set(PREPARED_CIRCUIT_CALLS.get() + 1);
        let (mut marker, mut remaining) = scratch.borrow().take_region(CIRCUIT_EXTRA);
        Self::copy_host_to_view(&mut marker, &[0x63; CIRCUIT_EXTRA]);
        poulpy_bin_fhe::reference::circuit_bootstrapping::circuit_bootstrapping_execute_prepared_reference(
            module,
            res,
            lwe,
            key,
            plan,
            &mut remaining,
        );
        assert!(marker.iter().all(|&byte| byte == 0x63));
    }
}

#[test]
fn prepared_circuit_override_propagates_to_direct_scratch_queries() {
    PREPARED_CIRCUIT_CALLS.set(0);
    poulpy_bin_fhe::test_suite::parity::circuit_bootstrapping::test_circuit_bootstrapping_parity(
        &Module::<crate::FFT64Ref>::new(64),
        &Module::<OverrideBackend>::new(64),
    );
    assert!(PREPARED_CIRCUIT_CALLS.get() > 0);
}

thread_local! {
    static COMPRESSED_FACTORY_CALLS: Cell<usize> = const { Cell::new(0) };
    static LUT_SET_CALLS: Cell<usize> = const { Cell::new(0) };
    static LUT_ROTATE_CALLS: Cell<usize> = const { Cell::new(0) };
}

// These overrides use only storage allocation and transfers. OpaqueBackend has
// no polynomial arithmetic implementations or host-accessible buffer views.
unsafe impl BlindRotationKeyCompressedFactoryImpl<CGGI> for OpaqueBackend {
    fn blind_rotation_key_compressed_alloc<A: BlindRotationKeyInfos>(
        module: &Module<Self>,
        infos: &A,
    ) -> BlindRotationKeyCompressed<Self::OwnedBuf, CGGI, Self::ZnxWord> {
        assert_eq!(module.n(), infos.n_glwe().as_usize());
        COMPRESSED_FACTORY_CALLS.set(COMPRESSED_FACTORY_CALLS.get() + 1);
        BlindRotationKeyCompressed::from_parts(
            (0..infos.n_lwe().as_usize())
                .map(|_| module.ggsw_compressed_alloc_from_infos(infos))
                .collect(),
            poulpy_core::Distribution::NONE,
        )
    }
}

fn opaque_lut_coefficients(lut: &LookupTable<OpaqueOwned, i64>) -> Vec<i64> {
    assert_eq!(lut.extension_factor(), 1);
    assert_eq!(lut.size(), 1);
    OpaqueBackend::to_host_bytes(lut.polynomials()[0].data().data())
        .chunks_exact(size_of::<i64>())
        .map(|word| i64::from_ne_bytes(word.try_into().unwrap()))
        .collect()
}

fn write_opaque_lut(lut: &mut LookupTable<OpaqueOwned, i64>, coefficients: &[i64]) {
    let bytes: Vec<u8> = coefficients.iter().flat_map(|word| word.to_ne_bytes()).collect();
    OpaqueBackend::copy_from_host(lut.polynomials_mut()[0].data_mut().data_mut(), &bytes);
}

fn rotate_opaque_lut(lut: &mut LookupTable<OpaqueOwned, i64>, k: i64) {
    let input = opaque_lut_coefficients(lut);
    let n = input.len();
    let mut output = vec![0; n];
    for (i, value) in input.into_iter().enumerate() {
        let index = (i as i64 + k).rem_euclid(2 * n as i64) as usize;
        output[index % n] = if index < n { value } else { -value };
    }
    write_opaque_lut(lut, &output);
}

unsafe impl LookupTableFactoryImpl for OpaqueBackend {
    fn lookup_table_set(module: &Module<Self>, res: &mut LookupTable<Self::OwnedBuf, i64>, f: &[i64], k: usize) {
        // This independent test implementation supports one normalized limb.
        assert_eq!(res.extension_factor(), 1);
        assert_eq!(res.size(), 1);
        assert_eq!(res.n().as_usize(), module.n());
        assert_eq!(res.k().as_usize(), res.base2k().as_usize());
        assert!(!f.is_empty() && module.n().is_multiple_of(f.len()));
        let base2k = res.base2k().as_usize();
        assert!((1..=base2k).contains(&k));
        let scale = 1i64 << (base2k - k);
        let bound = 1i64 << (base2k - 1);
        let step = module.n() / f.len();
        let mut coefficients = vec![0; module.n()];
        for (chunk, value) in coefficients.chunks_mut(step).zip(f) {
            let encoded = value * scale;
            assert!((-bound..bound).contains(&encoded));
            chunk.fill(encoded);
        }
        LUT_SET_CALLS.set(LUT_SET_CALLS.get() + 1);
        write_opaque_lut(res, &coefficients);
        let drift = step / 2;
        rotate_opaque_lut(res, -(drift as i64));
        res.set_drift(drift);
    }

    fn lookup_table_rotate(module: &Module<Self>, k: i64, res: &mut LookupTable<Self::OwnedBuf, i64>) {
        assert_eq!(res.n().as_usize(), module.n());
        LUT_ROTATE_CALLS.set(LUT_ROTATE_CALLS.get() + 1);
        rotate_opaque_lut(res, k);
    }
}

#[test]
fn compressed_key_factory_dispatches_with_opaque_storage() {
    COMPRESSED_FACTORY_CALLS.set(0);
    let module = Module::<OpaqueBackend>::new(32);
    let layout = BlindRotationKeyLayout {
        n_glwe: 32usize.into(),
        n_lwe: 3usize.into(),
        base2k: 12usize.into(),
        dnum: 2usize.into(),
        k_aux: 12usize.into(),
        rank: 1usize.into(),
    };
    let direct = module.blind_rotation_key_compressed_alloc(&layout);
    assert_eq!(COMPRESSED_FACTORY_CALLS.get(), 1);
    let wrapped = BlindRotationKeyCompressed::<poulpy_hal::AlignedBuf, CGGI, i64>::alloc(&module, &layout);
    assert_eq!(COMPRESSED_FACTORY_CALLS.get(), 2);
    for key in [&direct, &wrapped] {
        assert_eq!(key.keys().len(), layout.n_lwe.as_usize());
        assert_eq!(key.distribution(), poulpy_core::Distribution::NONE);
        for element in key.keys() {
            assert_eq!(element.n(), layout.n_glwe);
            assert_eq!(element.base2k(), layout.base2k);
            assert_eq!(element.dnum(), layout.dnum);
            assert_eq!(element.dsize(), layout.dsize());
            assert_eq!(element.k_aux(), layout.k_aux);
            assert_eq!(element.rank(), layout.rank);
        }
    }
}

#[test]
fn lookup_table_api_dispatches_without_reference_arithmetic_bounds() {
    LUT_SET_CALLS.set(0);
    LUT_ROTATE_CALLS.set(0);
    let module = Module::<OpaqueBackend>::new(32);
    let layout = LookUpTableLayout {
        n: 32usize.into(),
        extension_factor: 1,
        k: 12usize.into(),
        base2k: 12usize.into(),
    };
    let mut direct = LookupTable::alloc(&module, &layout);
    let mut wrapped = LookupTable::alloc(&module, &layout);
    direct.set_rotation_direction(LookUpTableRotationDirection::Right);
    wrapped.set_rotation_direction(LookUpTableRotationDirection::Right);
    module.lookup_table_set(&mut direct, &[1, -1, 2, -2], 4);
    assert_eq!(LUT_SET_CALLS.get(), 1);
    wrapped.set(&module, &[1, -1, 2, -2], 4);
    assert_eq!(LUT_SET_CALLS.get(), 2);
    let expected = [vec![256; 4], vec![-256; 8], vec![512; 8], vec![-512; 8], vec![-256; 4]].concat();
    for lut in [&direct, &wrapped] {
        assert_eq!(opaque_lut_coefficients(lut), expected);
        assert_eq!(lut.drift(), 4);
        assert!(matches!(lut.rotation_direction(), LookUpTableRotationDirection::Right));
    }
    module.lookup_table_rotate(3, &mut direct);
    assert_eq!(LUT_ROTATE_CALLS.get(), 1);
    let rotated = [vec![256; 7], vec![-256; 8], vec![512; 8], vec![-512; 8], vec![-256; 1]].concat();
    assert_eq!(opaque_lut_coefficients(&direct), rotated);
    assert_eq!(opaque_lut_coefficients(&wrapped), expected);
    assert_eq!(direct.drift(), 4);
    assert!(matches!(direct.rotation_direction(), LookUpTableRotationDirection::Right));
}
