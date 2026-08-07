use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, HostDataRef, ScalarZnx, ScratchArena, Stats};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendRef, GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendRef,
};

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut;
}

pub trait GGLWENoise<BE: Backend> {
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

pub trait GGSWNoise<BE: Backend> {
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGSWToBackendRef<BE> + GGSWInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;
}
