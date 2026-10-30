use poulpy_core::{
    GGLWEDecompress, GLWEFill,
    layouts::{
        GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
        GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos,
        GLWEToBackendMut, LWEInfos,
    },
};
use poulpy_hal::{
    api::{
        MatZnxAddAssign, MatZnxNormalize, MatZnxNormalizeAssign, MatZnxNormalizeTmpBytes, VecZnxAddAssign, VecZnxNormalize,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena},
};

pub trait GLWEPatCompressedReference<BE: Backend> {
    fn glwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos;

    fn glwe_pat_compressed_finalize_tmp_bytes_reference(&self) -> usize;

    fn glwe_pat_compressed_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos;
}

impl<BE: Backend> GLWEPatCompressedReference<BE> for Module<BE>
where
    Self: VecZnxAddAssign<BE> + VecZnxNormalizeTmpBytes + VecZnxNormalize<BE> + GLWEFill<BE>,
{
    fn glwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        let metadata = crate::metadata::aggregate_metadata(res.noise(), a.noise());
        let mut res_be = res.to_backend_mut();
        let a_be = a.to_backend_ref();
        self.vec_znx_add_assign(res_be.data_mut(), 0, a_be.data(), 0);
        drop(res_be);
        res.set_noise(metadata);
    }

    fn glwe_pat_compressed_finalize_tmp_bytes_reference(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes()
    }

    fn glwe_pat_compressed_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        assert!(res.glwe_layout() == pat.glwe_layout(), "invalid finalization: layouts differ");
        assert!(
            scratch.available() >= self.glwe_pat_compressed_finalize_tmp_bytes_reference(),
            "invalid finalization: insufficient scratch"
        );
        res.set_canonical(true);
        {
            let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
            let mut res_be = res.to_backend_mut();
            let pat_be = pat.to_backend_ref();
            self.vec_znx_normalize(res_be.data_mut(), base2k, k, 0, 0, pat_be.data(), base2k, 0, scratch);
        }
        // Seeded masks are uniform digits, already canonical.
        self.glwe_fill_mask_from_seed(res, *pat.seed());
        res.set_noise(pat.noise());
    }
}

pub trait GGLWEPatCompressedReference<BE: Backend> {
    fn gglwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos;

    fn gglwe_pat_compressed_finalize_tmp_bytes_reference(&self) -> usize;

    fn gglwe_pat_compressed_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GGLWEPatCompressedReference<BE> for Module<BE>
where
    Self: MatZnxAddAssign<BE> + MatZnxNormalizeTmpBytes + MatZnxNormalizeAssign<BE> + GGLWEDecompress<BE>,
{
    fn gglwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos,
    {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        let metadata = crate::metadata::aggregate_metadata(res.noise(), a.noise());
        let mut res_be = res.to_backend_mut();
        let a_be = a.to_backend_ref();
        self.mat_znx_add_assign(res_be.data_mut(), a_be.data());
        drop(res_be);
        res.set_noise(metadata);
    }

    fn gglwe_pat_compressed_finalize_tmp_bytes_reference(&self) -> usize {
        self.mat_znx_normalize_tmp_bytes()
    }

    fn gglwe_pat_compressed_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        assert!(
            res.gglwe_layout() == pat.gglwe_layout(),
            "invalid finalization: layouts differ"
        );
        assert!(
            scratch.available() >= self.gglwe_pat_compressed_finalize_tmp_bytes_reference(),
            "invalid finalization: insufficient scratch"
        );
        self.gglwe_decompress(res, pat);
        let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
        let mut res_be = res.to_backend_mut();
        // Seeded masks are uniform digits, already canonical: only the bodies are normalized.
        self.mat_znx_normalize_assign(base2k, k, 0, res_be.data_mut(), 0, scratch);
        drop(res_be);
        res.set_noise(pat.noise());
    }
}

pub trait GGLWEPatReference<BE: Backend> {
    fn gglwe_pat_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn gglwe_pat_finalize_tmp_bytes_reference(&self) -> usize;

    fn gglwe_pat_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWEToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GGLWEPatReference<BE> for Module<BE>
where
    Self: MatZnxAddAssign<BE> + MatZnxNormalize<BE> + MatZnxNormalizeTmpBytes,
{
    fn gglwe_pat_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        let metadata = crate::metadata::aggregate_common_key_metadata(res.noise(), a.noise(), res.n().as_usize());
        let mut res_be = res.to_backend_mut();
        let a_be = a.to_backend_ref();
        self.mat_znx_add_assign(res_be.data_mut(), a_be.data());
        drop(res_be);
        res.set_noise(metadata);
    }

    fn gglwe_pat_finalize_tmp_bytes_reference(&self) -> usize {
        self.mat_znx_normalize_tmp_bytes()
    }

    fn gglwe_pat_finalize_reference<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        assert!(
            res.gglwe_layout() == pat.gglwe_layout(),
            "invalid finalization: layouts differ"
        );
        assert!(
            scratch.available() >= self.gglwe_pat_finalize_tmp_bytes_reference(),
            "invalid finalization: insufficient scratch"
        );
        let (res_base2k, res_k, pat_base2k): (usize, usize, usize) = (res.base2k().into(), res.k().into(), pat.base2k().into());
        let cols: usize = res.rank_out().as_usize() + 1;
        let mut res_be = res.to_backend_mut();
        let pat_be = pat.to_backend_ref();
        for col in 0..cols {
            self.mat_znx_normalize(
                res_be.data_mut(),
                res_base2k,
                res_k,
                0,
                col,
                pat_be.data(),
                pat_base2k,
                col,
                scratch,
            );
        }
        drop(res_be);
        res.set_noise(pat.noise());
    }
}
