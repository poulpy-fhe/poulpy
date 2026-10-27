//! Registration of the `poulpy-ckks` extension points.
//!
//! Every homomorphic family forwards to its reference body. Encoding is
//! float work whose result must be byte-identical on every backend: it is
//! staged through the host and computed by the portable implementation, on
//! the canonical tables. No float kernel runs on the device.

use poulpy_ckks::{
    CKKSCtBounds, CKKSError, CKKSInfos, CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, CKKSResult, GLWEPlaintextInfos,
    GLWEPlaintextMeta, SetCKKSInfos, SetGLWEPlaintextInfos,
    api::{CKKSEncodingOps, CKKSEncodingScalar, PaCoScalar, ShipScalar},
    encoding::{paco_coeff_encodings_host, ship_coeff_encodings_host},
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef, CKKSEncodingBufferInfos, CKKSModuleAlloc,
        CKKSPlaintextOwned, PaCoPlan, ShipCoeffEncodings, ShipPlan, ckks_one_pt,
    },
    oep::CKKSEncodingImpl,
    reference::encoding::{CKKSSlotEmbedding, decode_coeffs_into_host, encode_coeffs_into_host},
};
use poulpy_core::{
    ComponentNoise, GLWECopy,
    layouts::{
        Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef,
        IntPolyInfos, LWEInfos, ModuleCoreAlloc, Rank, SetK, TorusPrecision,
    },
};
use poulpy_cpu_portable::ckks_encoding::{EncodingFFTTable, OwnedEncodingPlanSet};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use poulpy_hal::{
    AlignedBuf,
    api::ModuleN,
    layouts::{ConjugateInvariant, HostBytesBackend, Module, ModulePlanCache, Ring, ScratchArena, Standard},
};

use crate::{
    NTT4x30Gpu,
    ckks_add_sub::{impl_add_cached_one, impl_sub_cached_one},
    module::handle,
    runtime::{GpuMut, GpuRef, GpuRuntime, access},
};

/// The constant plaintexts `1` of a module, one per radix.
struct OnePlaintexts<BE: poulpy_hal::layouts::Backend>(Mutex<HashMap<usize, Arc<CKKSPlaintextOwned<BE>>>>);

/// The device plaintext of the constant `1.0` of the add-one and sub-one
/// operations, [`ckks_one_pt`], kept by the module: only the first request
/// for a radix uploads it.
pub(crate) fn one_plaintext<RT, R>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    base2k: Base2K,
) -> CKKSResult<Arc<CKKSPlaintextOwned<NTT4x30Gpu<RT, R>>>>
where
    RT: GpuRuntime,
    R: Ring,
    Module<NTT4x30Gpu<RT, R>>: CKKSModuleAlloc<NTT4x30Gpu<RT, R>>,
{
    // Building the constant uses the device. The access is taken before the
    // cache is locked, as everywhere else: a thread that records holds the
    // device and may ask for the constant.
    let _access = access::shared();
    handle(module)
        .plan_cache()
        .with_or_create::<OnePlaintexts<NTT4x30Gpu<RT, R>>, _, _>(
            || Ok(OnePlaintexts(Mutex::new(HashMap::new()))),
            |cache: &OnePlaintexts<NTT4x30Gpu<RT, R>>| {
                let mut cache = cache.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(one) = cache.get(&base2k.as_usize()) {
                    return Ok(one.clone());
                }
                let one = Arc::new(ckks_one_pt(module, base2k)?);
                cache.insert(base2k.as_usize(), one.clone());
                Ok(one)
            },
        )
        .map_err(CKKSError::from)?
}

/// The host backend the staged work is typed with: host bytes, `i64` limbs, the ring of the device backend.
type Host<R> = HostBytesBackend<R>;

/// A host copy of the storage of a device plaintext, with the plaintext's own
/// metadata: what the portable encoder reads and writes in its place.
struct Staged<P> {
    pt: P,
    host: GLWE<AlignedBuf, i64>,
}

/// A zeroed host container with the shape of the device storage `data`.
fn host_like<RT: GpuRuntime, R: Ring, D: poulpy_hal::layouts::Data>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    data: &GLWE<D, i64>,
) -> GLWE<AlignedBuf, i64> {
    let shape = data.data().shape();
    assert!(shape.is_dense(), "CKKS encoding is staged from dense plaintext storage");
    let base2k = LWEInfos::base2k(data);
    module.glwe_host_alloc_from_infos(&GLWELayout {
        n: Degree(shape.n() as u32),
        base2k,
        k: TorusPrecision(shape.size() as u32 * base2k.as_u32()),
        rank: Rank(shape.cols() as u32 - 1),
    })
}

macro_rules! impl_staged {
    ($reference:ty) => {
        impl<P: LWEInfos> LWEInfos for Staged<$reference> {
            fn noise(&self) -> Option<ComponentNoise> {
                self.pt.noise()
            }

            fn n(&self) -> Degree {
                self.pt.n()
            }

            fn k(&self) -> TorusPrecision {
                self.pt.k()
            }

            fn base2k(&self) -> Base2K {
                self.pt.base2k()
            }

            fn max_size(&self) -> usize {
                self.pt.max_size()
            }
        }

        impl<P: GLWEInfos> GLWEInfos for Staged<$reference> {
            fn rank(&self) -> Rank {
                self.pt.rank()
            }
        }

        impl<P: IntPolyInfos> IntPolyInfos for Staged<$reference> {
            fn encoded_k(&self) -> TorusPrecision {
                self.pt.encoded_k()
            }
        }

        impl<P: GLWEPlaintextInfos> GLWEPlaintextInfos for Staged<$reference> {
            fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
                self.pt.plaintext_meta()
            }
        }

        impl<P: CKKSInfos> CKKSInfos for Staged<$reference> {}

        impl<P, R: Ring> GLWEToBackendRef<Host<R>> for Staged<$reference> {
            fn to_backend_ref(&self) -> GLWEBackendRef<'_, Host<R>> {
                GLWEToBackendRef::<Host<R>>::to_backend_ref(&self.host)
            }
        }
    };
}

impl_staged!(&P);
impl_staged!(&mut P);

impl<P: SetGLWEPlaintextInfos> SetGLWEPlaintextInfos for Staged<&mut P> {
    fn set_plaintext_meta(&mut self, meta: Option<GLWEPlaintextMeta>) {
        self.pt.set_plaintext_meta(meta)
    }
}

impl<P: SetK> SetK for Staged<&mut P> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.pt.set_k(k)
    }
}

impl<P, R: Ring> GLWEToBackendMut<Host<R>> for Staged<&mut P> {
    fn set_noise(&mut self, _: Option<ComponentNoise>) {}

    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, Host<R>> {
        GLWEToBackendMut::<Host<R>>::to_backend_mut(&mut self.host)
    }

    fn set_canonical(&mut self, _: bool) {}
}

/// The first `len` values of a device encoding buffer, on the host.
fn download<RT: GpuRuntime, F: CKKSEncodingScalar>(data: &GpuRef<'_, RT>, len: usize) -> AlignedBuf {
    AlignedBuf::from(data.region(0, len * size_of::<F>()).to_host())
}

/// The host transform of the first `len` values of a device buffer: moved to
/// the host, transformed by the portable tables, moved back.
fn transform_assign<RT: GpuRuntime, F: CKKSEncodingScalar>(
    data: &mut GpuMut<'_, RT>,
    len: usize,
    transform: impl FnOnce(&mut [F]) -> anyhow::Result<()>,
) -> CKKSResult<()> {
    let bytes = len * size_of::<F>();
    let mut host = AlignedBuf::from(data.region(0, bytes).to_host());
    transform(bytemuck::cast_slice_mut(&mut host[..bytes])).map_err(CKKSError::from)?;
    data.region_mut(0, bytes).write_prefix(&host[..bytes]);
    Ok(())
}

unsafe impl<RT, R, F> CKKSEncodingImpl<F> for NTT4x30Gpu<RT, R>
where
    RT: GpuRuntime,
    R: Ring + CKKSSlotEmbedding,
    F: CKKSEncodingScalar,
{
    type Plans = OwnedEncodingPlanSet<F, EncodingFFTTable<F>>;

    fn ckks_encoding_plan_cache_impl(module: &Module<Self>) -> &ModulePlanCache {
        handle(module).plan_cache()
    }

    fn ckks_encoding_plans_create_impl(module: &Module<Self>) -> CKKSResult<Self::Plans> {
        OwnedEncodingPlanSet::new(2 * poulpy_ckks::CKKSModuleInfos::ckks_max_slots(module)).map_err(CKKSError::from)
    }

    fn ckks_encode_coeffs_into_impl<P>(
        module: &Module<Self>,
        pt: &mut P,
        coeffs: &CKKSEncodingBufferBackendRef<'_, Self, F>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
    {
        let len = CKKSEncodingBufferInfos::len(coeffs);
        let host_coeffs = download::<RT, F>(&coeffs.data, len);
        // The encoder writes every limb of the plaintext: nothing is read back first.
        let host = host_like(module, &GLWEToBackendRef::<Self>::to_backend_ref(pt));
        let mut staged = Staged { pt, host };
        encode_coeffs_into_host::<Host<R>, F, _>(&mut staged, &CKKSEncodingBuffer::from_data(&host_coeffs[..], len))
            .map_err(CKKSError::from)?;
        let Staged { pt, host } = staged;
        let mut device = GLWEToBackendMut::<Self>::to_backend_mut(pt);
        // Host storage is padded; the device view has the exact length.
        let storage = device.data_mut().data_mut();
        let bytes = storage.len();
        storage.write_prefix(&host.data().data()[..bytes]);
        Ok(())
    }

    fn ckks_decode_coeffs_into_impl<P>(
        module: &Module<Self>,
        pt: &P,
        coeffs: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> CKKSResult<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
    {
        let len = CKKSEncodingBufferInfos::len(coeffs);
        let bytes = len * size_of::<F>();
        let host = {
            let device = GLWEToBackendRef::<Self>::to_backend_ref(pt);
            let mut host = host_like(module, &device);
            let storage = device.data().data();
            storage.read_prefix(&mut host.data_mut().data_mut()[..storage.len()]);
            host
        };
        let mut host_coeffs = AlignedBuf::from(vec![0u8; bytes]);
        decode_coeffs_into_host::<Host<R>, F, _>(
            &Staged { pt, host },
            &mut CKKSEncodingBuffer::from_data(&mut host_coeffs[..], len),
        )
        .map_err(CKKSError::from)?;
        coeffs.data.region_mut(0, bytes).write_prefix(&host_coeffs[..bytes]);
        Ok(())
    }

    fn ckks_slots_to_coeffs_assign_impl(
        _module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> CKKSResult<()> {
        let (map, fft) = plans.for_slots(CKKSEncodingBufferInfos::len(values) / 2)?;
        let len = CKKSEncodingBufferInfos::len(values);
        transform_assign(&mut values.data, len, |values| R::slots_to_coeffs_assign(map, fft, values))
    }

    fn ckks_coeffs_to_slots_assign_impl(
        _module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> CKKSResult<()> {
        let (map, fft) = plans.for_slots(CKKSEncodingBufferInfos::len(values) / 2)?;
        let len = CKKSEncodingBufferInfos::len(values);
        transform_assign(&mut values.data, len, |values| R::coeffs_to_slots_assign(map, fft, values))
    }
}

/// A host copy of the ciphertext `ct`, whose public coefficients the PaCo and
/// SHIP coefficient encodings read.
macro_rules! staged_ciphertext {
    ($module:ident, $ct:ident, $plan:ident, $scratch:ident, $name:literal) => {{
        if $module.n() != $plan.n() {
            return Err(CKKSError::from(anyhow::anyhow!(
                "{} module degree {} does not match plan degree {}",
                $name,
                $module.n(),
                $plan.n()
            )));
        }
        let mut owned = $module.ckks_ciphertext_alloc_from_infos($ct);
        $module.glwe_copy(&mut owned, $ct, $scratch);
        owned.set_meta_checked($ct.meta())?;
        owned.to_host_owned::<BE>()
    }};
}

/// The PaCo coefficient encoding: float work on the public coefficients of a
/// ciphertext, done by the scheme's host routine on a host copy of the
/// ciphertext; the plaintexts it builds are device plaintexts.
fn paco_coeff_encodings<BE, F, Src>(
    module: &Module<BE>,
    ct: &Src,
    plan: &PaCoPlan,
    base2k: Base2K,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<[CKKSPlaintextOwned<BE>; 4]>
where
    BE: poulpy_hal::layouts::Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWECopy<BE>,
    F: PaCoScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let host = staged_ciphertext!(module, ct, plan, scratch, "PaCo");
    paco_coeff_encodings_host::<BE, _, F>(module, &host, plan, base2k).map_err(CKKSError::from)
}

/// The SHIP coefficient encoding, staged like [`paco_coeff_encodings`].
fn ship_coeff_encodings<BE, F, Src>(
    module: &Module<BE>,
    ct: &Src,
    plan: &ShipPlan,
    base2k: Base2K,
    complex: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<ShipCoeffEncodings<BE::OwnedBuf, i64>>
where
    BE: poulpy_hal::layouts::Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWECopy<BE>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let host = staged_ciphertext!(module, ct, plan, scratch, "SHIP");
    ship_coeff_encodings_host::<BE, _, F>(module, &host, plan, base2k, complex).map_err(CKKSError::from)
}

/// The families of both rings, registered through the upstream macros, which
/// take a concrete backend type.
macro_rules! impl_ckks {
    ($be:ty) => {
        ::poulpy_ckks::impl_ckks_copy_reference!($be);
        ::poulpy_ckks::impl_ckks_encryption_reference!($be);
        ::poulpy_ckks::impl_ckks_mul_reference!($be);
        ::poulpy_ckks::impl_ckks_dot_product_reference!($be);
        ::poulpy_ckks::impl_ckks_linear_transformation_reference!($be);
        ::poulpy_ckks::impl_ckks_neg_reference!($be);
        ::poulpy_ckks::impl_ckks_pow2_reference!($be);
        ::poulpy_ckks::impl_ckks_rotate_reference!($be);
        impl_add_cached_one!($be);
        impl_sub_cached_one!($be);
        ::poulpy_ckks::impl_ckks_plaintext_reference!($be);
        ::poulpy_ckks::impl_ckks_polynomial_evaluation_reference!($be);
        ::poulpy_ckks::impl_ckks_eval_mod_reference!($be);
        ::poulpy_ckks::impl_ckks_inner_sum_derived_full!($be);
    };
}

/// The families of the standard ring only, `$ci` the conjugate-invariant twin of `$be`.
macro_rules! impl_ckks_standard {
    ($be:ty, $ci:ty) => {
        ::poulpy_ckks::impl_ckks_encapsulated_mod_up_reference!($be);
        ::poulpy_ckks::impl_ckks_mod_up_first_factor_reference!($be);
        ::poulpy_ckks::impl_ckks_conjugate_reference!($be);
        ::poulpy_ckks::impl_ckks_imag_reference!($be);
        ::poulpy_ckks::impl_ckks_bootstrapping_reference!($be);
        ::poulpy_ckks::impl_ckks_bootstrapping_ci_reference!($be, $ci);
        ::poulpy_ckks::impl_ckks_fold_reference!($be);
        ::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!($be);
        ::poulpy_ckks::impl_ckks_dft_reference!($be);

        unsafe impl ::poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for $be {
            fn ckks_paco_coeff_encodings_tmp_bytes_impl<F>(_module: &Module<$be>, _plan: &PaCoPlan) -> CKKSResult<usize>
            where
                F: PaCoScalar,
                $be: CKKSEncodingImpl<F>,
            {
                Ok(0)
            }

            fn ckks_paco_coeff_encodings_impl<F, Src>(
                module: &Module<$be>,
                ct: &Src,
                plan: &PaCoPlan,
                base2k: Base2K,
                scratch: &mut ScratchArena<'_, $be>,
            ) -> CKKSResult<[CKKSPlaintextOwned<$be>; 4]>
            where
                F: PaCoScalar,
                $be: CKKSEncodingImpl<F>,
                Src: GLWEToBackendRef<$be> + CKKSCtBounds,
            {
                paco_coeff_encodings::<$be, F, Src>(module, ct, plan, base2k, scratch)
            }
        }

        unsafe impl ::poulpy_ckks::oep::CKKSShipCoeffEncodingImpl for $be {
            fn ckks_ship_coeff_encodings_tmp_bytes_impl<F>(
                _module: &Module<$be>,
                _plan: &ShipPlan,
                _base2k: Base2K,
                _complex: bool,
            ) -> CKKSResult<usize>
            where
                F: ShipScalar,
                $be: CKKSEncodingImpl<F>,
            {
                Ok(0)
            }

            fn ckks_ship_coeff_encodings_impl<F, Src>(
                module: &Module<$be>,
                ct: &Src,
                plan: &ShipPlan,
                base2k: Base2K,
                complex: bool,
                scratch: &mut ScratchArena<'_, $be>,
            ) -> CKKSResult<ShipCoeffEncodings<<$be as poulpy_hal::layouts::Backend>::OwnedBuf, i64>>
            where
                F: ShipScalar,
                $be: CKKSEncodingImpl<F>,
                Src: GLWEToBackendRef<$be> + CKKSCtBounds,
            {
                ship_coeff_encodings::<$be, F, Src>(module, ct, plan, base2k, complex, scratch)
            }
        }
    };
}

#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
mod wgpu {
    use super::*;
    impl_ckks!(crate::NTT4x30Wgpu);
    impl_ckks_standard!(crate::NTT4x30Wgpu, crate::NTT4x30Wgpu<ConjugateInvariant>);
    impl_ckks!(crate::NTT4x30Wgpu<ConjugateInvariant>);
}

#[cfg(feature = "enable-cuda")]
mod cuda {
    use super::*;
    impl_ckks!(crate::NTT4x30Cuda);
    impl_ckks_standard!(crate::NTT4x30Cuda, crate::NTT4x30Cuda<ConjugateInvariant>);
    impl_ckks!(crate::NTT4x30Cuda<ConjugateInvariant>);
}

#[cfg(feature = "enable-hip")]
mod hip {
    use super::*;
    impl_ckks!(crate::NTT4x30Hip);
    impl_ckks_standard!(crate::NTT4x30Hip, crate::NTT4x30Hip<ConjugateInvariant>);
    impl_ckks!(crate::NTT4x30Hip<ConjugateInvariant>);
}
