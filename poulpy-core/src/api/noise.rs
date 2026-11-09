use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Ring, ScalarZnx, ScratchArena, Standard, Stats};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendRef, GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendRef,
};

pub trait GLWENoise<BE: Backend, Rg: Ring = Standard> {
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        P: GLWEToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut;
}

pub trait GGLWENoise<BE: Backend, Rg: Ring = Standard> {
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64, Rg>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGLWEToBackendRef<BE, Rg> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

pub trait GGSWNoise<BE: Backend, Rg: Ring = Standard> {
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64, Rg>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGSWToBackendRef<BE, Rg> + GGSWInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;
}
