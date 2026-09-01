use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, HostBytesBackend, Module, ScalarZnx, ScratchArena, SvpPPolOwned},
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{GGSWPreparedFactory, LWEInfos},
};

use crate::blind_rotation::{
    BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, CGGI,
    utils::set_xai_plus_y,
};

impl<BE: Backend<ZnxWord = i64>> BlindRotationKeyPreparedFactory<CGGI, BE> for Module<BE>
where
    Self: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
{
    fn blind_rotation_key_prepared_alloc<A>(&self, infos: &A) -> BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>
    where
        A: BlindRotationKeyInfos,
    {
        BlindRotationKeyPrepared {
            data: (0..infos.n_lwe().as_usize())
                .map(|_| self.ggsw_prepared_alloc_from_infos(infos))
                .collect(),
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
        }
    }

    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        self.ggsw_prepare_tmp_bytes(infos)
    }

    fn prepare_blind_rotation_key(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.data.len(), other.keys.len());
        }

        let n: usize = other.n().as_usize();

        for (a, b) in res.data.iter_mut().zip(other.keys.iter()) {
            self.ggsw_prepare(a, b, &mut scratch.borrow());
        }

        res.dist = other.dist;

        if let Distribution::BinaryBlock(_) = other.dist {
            let mut x_pow_a: Vec<SvpPPolOwned<BE>> = Vec::with_capacity(n << 1);
            let mut buf: ScalarZnx<Vec<u8>, i64> = ScalarZnx::from_data(
                HostBytesBackend::alloc_zeroed_bytes(ScalarZnx::<Vec<u8>, i64>::bytes_of(n, 1)),
                n,
                1,
            );
            (0..n << 1).for_each(|i| {
                let mut res: SvpPPolOwned<BE> = self.svp_ppol_alloc(1);
                set_xai_plus_y(self, i, 0, &mut res, &mut buf);
                x_pow_a.push(res);
            });
            res.x_pow_a = Some(x_pow_a);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ptr::NonNull,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use poulpy_core::Distribution;
    use poulpy_hal::{
        alloc_aligned,
        layouts::{
            Backend, Device, MatZnxBackendRef, Module, ScalarZnxBackendRef, ScratchArena, ScratchOwned, SvpPPolBackendMut,
            SvpPPolBackendRef, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
            VmpPMatBackendMut, VmpPMatBackendRef,
        },
        oep::{HalModuleImpl, HalSvpImpl, HalVmpImpl},
    };

    use crate::blind_rotation::{
        BlindRotationKey, BlindRotationKeyLayout, BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, CGGI,
    };

    #[derive(Default, PartialEq, Eq)]
    struct OpaqueBuffer(Vec<u8>);

    #[derive(Default, PartialEq, Eq)]
    struct OpaqueRef<'a>(&'a [u8]);

    #[derive(Default, PartialEq, Eq)]
    struct OpaqueMut<'a>(&'a mut [u8]);

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct OpaqueDevice;

    static SVP_PREPARE_CALLS: AtomicUsize = AtomicUsize::new(0);

    impl poulpy_hal::execution::ScratchWorkers for OpaqueDevice {}

    impl Backend for OpaqueDevice {
        type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
        type ZnxWord = i64;
        type BigWord = i64;
        type DftWord = f64;
        type OwnedBuf = OpaqueBuffer;
        type BufRef<'a> = OpaqueRef<'a>;
        type BufMut<'a> = OpaqueMut<'a>;
        type Handle = ();
        type Location = Device;

        fn alloc_bytes(len: usize) -> Self::OwnedBuf {
            OpaqueBuffer(alloc_aligned(len))
        }

        fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
            let mut res = alloc_aligned(bytes.len());
            res.copy_from_slice(bytes);
            OpaqueBuffer(res)
        }

        fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
            Self::from_host_bytes(&bytes)
        }

        fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
            buf.0.clone()
        }

        fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
            dst.copy_from_slice(&buf.0[..dst.len()]);
        }

        fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
            buf.0[..src.len()].copy_from_slice(src);
            buf.0[src.len()..].fill(0);
        }

        fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
            dst.copy_from_slice(buf.0);
        }

        fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
            buf.0.copy_from_slice(src);
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

        fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
            OpaqueMut(&mut buf.0)
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

        unsafe fn destroy(_: NonNull<Self::Handle>) {}
    }

    unsafe impl HalModuleImpl<OpaqueDevice> for OpaqueDevice {
        fn new(n: u64) -> Module<OpaqueDevice> {
            assert!(n.is_power_of_two());
            unsafe { Module::from_nonnull(NonNull::dangling(), n) }
        }
    }

    unsafe impl HalSvpImpl<OpaqueDevice> for OpaqueDevice {
        fn svp_prepare(
            _: &Module<Self>,
            _: &mut SvpPPolBackendMut<'_, Self>,
            _: usize,
            a: &ScalarZnxBackendRef<'_, Self>,
            _: usize,
        ) {
            let mut bytes = vec![0u8; Self::len_bytes_ref(&a.data)];
            Self::copy_view_to_host(&a.data, &mut bytes);
            assert!(
                bytes
                    .chunks_exact(size_of::<i64>())
                    .any(|word| i64::from_ne_bytes(word.try_into().unwrap()) != 0)
            );
            SVP_PREPARE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        fn svp_ppol_copy_backend(
            _: &Module<Self>,
            _: &mut SvpPPolBackendMut<'_, Self>,
            _: usize,
            _: &SvpPPolBackendRef<'_, Self>,
            _: usize,
        ) {
            unimplemented!()
        }

        fn svp_apply_dft(
            _: &Module<Self>,
            _: &mut VecZnxDftBackendMut<'_, Self>,
            _: usize,
            _: &SvpPPolBackendRef<'_, Self>,
            _: usize,
            _: &VecZnxBackendRef<'_, Self>,
            _: usize,
        ) {
            unimplemented!()
        }

        fn svp_apply_dft_to_dft(
            _: &Module<Self>,
            _: &mut VecZnxDftBackendMut<'_, Self>,
            _: usize,
            _: &SvpPPolBackendRef<'_, Self>,
            _: usize,
            _: &VecZnxDftBackendRef<'_, Self>,
            _: usize,
        ) {
            unimplemented!()
        }

        fn svp_apply_dft_to_dft_assign(
            _: &Module<Self>,
            _: &mut VecZnxDftBackendMut<'_, Self>,
            _: usize,
            _: &SvpPPolBackendRef<'_, Self>,
            _: usize,
        ) {
            unimplemented!()
        }
    }

    unsafe impl HalVmpImpl<OpaqueDevice> for OpaqueDevice {
        fn vmp_prepare_tmp_bytes(_: &Module<Self>, _: usize, _: usize, _: usize, _: usize) -> usize {
            0
        }

        fn vmp_prepare(
            _: &Module<Self>,
            _: &mut VmpPMatBackendMut<'_, Self>,
            _: &MatZnxBackendRef<'_, Self>,
            _: &mut ScratchArena<'_, Self>,
        ) {
        }

        fn vmp_apply_dft_tmp_bytes(_: &Module<Self>, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize) -> usize {
            unimplemented!()
        }

        fn vmp_apply_dft<R>(
            _: &Module<Self>,
            _: &mut R,
            _: &VecZnxBackendRef<'_, Self>,
            _: &VmpPMatBackendRef<'_, Self>,
            _: &mut ScratchArena<'_, Self>,
        ) where
            R: VecZnxDftToBackendMut<Self>,
        {
            unimplemented!()
        }

        fn vmp_apply_dft_to_dft_tmp_bytes(_: &Module<Self>, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize) -> usize {
            unimplemented!()
        }

        fn vmp_apply_dft_to_dft(
            _: &Module<Self>,
            _: &mut VecZnxDftBackendMut<'_, Self>,
            _: &VecZnxDftBackendRef<'_, Self>,
            _: &VmpPMatBackendRef<'_, Self>,
            _: usize,
            _: &mut ScratchArena<'_, Self>,
        ) {
            unimplemented!()
        }

        fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
            _: &Module<Self>,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> usize {
            unimplemented!()
        }

        fn vmp_apply_dft_to_dft_accumulate(
            _: &Module<Self>,
            _: &mut VecZnxDftBackendMut<'_, Self>,
            _: &VecZnxDftBackendRef<'_, Self>,
            _: &VmpPMatBackendRef<'_, Self>,
            _: usize,
            _: &mut ScratchArena<'_, Self>,
        ) {
            unimplemented!()
        }

        fn vmp_extract_selected_rows(
            _: &Module<Self>,
            _: &mut VmpPMatBackendMut<'_, Self>,
            _: &VmpPMatBackendRef<'_, Self>,
            _: usize,
            _: usize,
        ) {
            unimplemented!()
        }

        fn vmp_zero(_: &Module<Self>, _: &mut VmpPMatBackendMut<'_, Self>) {
            unimplemented!()
        }
    }

    #[test]
    fn prepared_key_factory_accepts_opaque_device_buffers() {
        fn assert_factory<M: BlindRotationKeyPreparedFactory<CGGI, OpaqueDevice>>() {}
        assert_factory::<Module<OpaqueDevice>>();

        let module = Module::<OpaqueDevice>::new(8);
        let layout = BlindRotationKeyLayout {
            n_glwe: 8usize.into(),
            n_lwe: 1usize.into(),
            base2k: 4usize.into(),
            dnum: 2usize.into(),
            k_aux: 4usize.into(),
            rank: 1usize.into(),
        };
        let mut key: BlindRotationKey<OpaqueBuffer, CGGI, i64> = BlindRotationKey::alloc(&module, &layout);
        key.dist = Distribution::BinaryBlock(1);
        let mut prepared: BlindRotationKeyPrepared<OpaqueBuffer, CGGI, OpaqueDevice> =
            BlindRotationKeyPrepared::alloc(&module, &layout);
        let mut scratch = ScratchOwned::<OpaqueDevice> {
            data: OpaqueDevice::alloc_bytes(1),
            _phantom: std::marker::PhantomData,
        };

        SVP_PREPARE_CALLS.store(0, Ordering::Relaxed);
        prepared.prepare(&module, &key, &mut scratch.arena());

        assert_eq!(SVP_PREPARE_CALLS.load(Ordering::Relaxed), 2 * module.n());
        assert_eq!(prepared.x_pow_a.as_ref().unwrap().len(), 2 * module.n());
    }
}
