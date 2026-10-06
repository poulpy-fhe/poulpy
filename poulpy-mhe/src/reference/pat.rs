use poulpy_core::{
    GLWEAdd, GLWEMaskFill, GLWENormalize,
    layouts::{
        GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEDecompress, GGLWEInfos,
        GGLWEToBackendMut, GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef,
        GLWEDecompress, GLWEInfos, GLWEToBackendMut, LWEInfos,
    },
};
use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes},
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
    Self: VecZnxAddAssign<BE> + VecZnxNormalizeTmpBytes + VecZnxNormalize<BE> + GLWEMaskFill<BE>,
{
    fn glwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        let metadata = super::aggregate_metadata(res.noise(), a.noise());
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
        res.set_canonical(true);
        {
            let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
            let mut res_be = res.to_backend_mut();
            let pat_be = pat.to_backend_ref();
            self.vec_znx_normalize(res_be.data_mut(), base2k, k, 0, 0, pat_be.data(), base2k, 0, scratch);
        }
        // Seeded masks are uniform digits, already canonical.
        self.fill_glwe_mask_from_seed(res, *pat.seed());
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
    Self: VecZnxAddAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + GLWEDecompress<Backend = BE>
        + GGLWEDecompress,
{
    fn gglwe_pat_compressed_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos,
    {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        let metadata = super::aggregate_metadata(res.noise(), a.noise());
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        let mut res_be = res.to_backend_mut();
        let a_be = a.to_backend_ref();
        for row in 0..dnum {
            for col in 0..rank_in {
                self.vec_znx_add_assign(res_be.at_view_mut(row, col).data_mut(), 0, a_be.at_view(row, col).data(), 0);
            }
        }
        drop(res_be);
        res.set_noise(metadata);
    }

    fn gglwe_pat_compressed_finalize_tmp_bytes_reference(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes()
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
        self.decompress_gglwe(res, pat);
        let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        let mut res_be = res.to_backend_mut();
        for row in 0..dnum {
            for col in 0..rank_in {
                self.vec_znx_normalize_assign(base2k, k, 0, res_be.at_view_mut(row, col).data_mut(), 0, scratch);
            }
        }
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
    Self: GLWEAdd<BE> + GLWENormalize<BE>,
{
    fn gglwe_pat_aggregate_assign_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        let metadata = super::aggregate_common_key_metadata(res.noise(), a.noise(), res.n().as_usize());
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        let mut res_be = res.to_backend_mut();
        let a_be = a.to_backend_ref();
        for row in 0..dnum {
            for col in 0..rank_in {
                self.glwe_add_assign(&mut res_be.at_view_mut(row, col), &a_be.at_view(row, col));
            }
        }
        drop(res_be);
        res.set_noise(metadata);
    }

    fn gglwe_pat_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
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
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        let mut res_be = res.to_backend_mut();
        let pat_be = pat.to_backend_ref();
        for row in 0..dnum {
            for col in 0..rank_in {
                self.glwe_normalize(&mut res_be.at_view_mut(row, col), &pat_be.at_view(row, col), scratch);
            }
        }
        drop(res_be);
        res.set_noise(pat.noise());
    }
}
