use poulpy_core::{
    GLWEAdd, GLWEMaskFill, GLWENormalize,
    layouts::{
        GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEDecompress, GGLWEInfos,
        GGLWEToBackendMut, GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef,
        GLWEDecompress, GLWEInfos, GLWEToBackendMut, LWEInfos,
    },
};
use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxCopy, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::layouts::{GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressedOwned};

pub trait PatAggregateReference<BE: Backend> {
    fn glwe_pat_compressed_aggregate_assign_reference(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        a: &GLWEPatCompressedOwned<BE>,
    );

    fn gglwe_pat_compressed_aggregate_assign_reference(
        &self,
        res: &mut GGLWEPatCompressedOwned<BE>,
        a: &GGLWEPatCompressedOwned<BE>,
    );

    fn gglwe_pat_aggregate_assign_reference(&self, res: &mut GGLWEPatOwned<BE>, a: &GGLWEPatOwned<BE>);
}

impl<BE: Backend> PatAggregateReference<BE> for Module<BE>
where
    Self: VecZnxAddAssign<BE> + GLWEAdd<BE>,
{
    fn glwe_pat_compressed_aggregate_assign_reference(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        a: &GLWEPatCompressedOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        {
            let mut res_be = <GLWEPatCompressedOwned<BE> as GLWECompressedToBackendMut<BE>>::to_backend_mut(res);
            let a_be = <GLWEPatCompressedOwned<BE> as GLWECompressedToBackendRef<BE>>::to_backend_ref(a);
            self.vec_znx_add_assign(res_be.data_mut(), 0, a_be.data(), 0);
        }
        res.canonical = false;
    }

    fn gglwe_pat_compressed_aggregate_assign_reference(
        &self,
        res: &mut GGLWEPatCompressedOwned<BE>,
        a: &GGLWEPatCompressedOwned<BE>,
    ) {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        {
            let mut res_be = <GGLWEPatCompressedOwned<BE> as GGLWECompressedToBackendMut<BE>>::to_backend_mut(res);
            let a_be = <GGLWEPatCompressedOwned<BE> as GGLWECompressedToBackendRef<BE>>::to_backend_ref(a);
            for row in 0..dnum {
                for col in 0..rank_in {
                    self.vec_znx_add_assign(res_be.at_view_mut(row, col).data_mut(), 0, a_be.at_view(row, col).data(), 0);
                }
            }
        }
        res.canonical = false;
    }

    fn gglwe_pat_aggregate_assign_reference(&self, res: &mut GGLWEPatOwned<BE>, a: &GGLWEPatOwned<BE>) {
        assert!(res.gglwe_layout() == a.gglwe_layout(), "invalid aggregation: layouts differ");
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        {
            let mut res_be = <GGLWEPatOwned<BE> as GGLWEToBackendMut<BE>>::to_backend_mut(res);
            let a_be = <GGLWEPatOwned<BE> as GGLWEToBackendRef<BE>>::to_backend_ref(a);
            for row in 0..dnum {
                for col in 0..rank_in {
                    self.glwe_add_assign(&mut res_be.at_view_mut(row, col), &a_be.at_view(row, col));
                }
            }
        }
        res.canonical = false;
    }
}

pub trait PatNormalizeReference<BE: Backend> {
    fn pat_normalize_tmp_bytes_reference(&self) -> usize;

    fn glwe_pat_compressed_normalize_assign_reference(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn gglwe_pat_compressed_normalize_assign_reference(
        &self,
        res: &mut GGLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn gglwe_pat_normalize_assign_reference(&self, res: &mut GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>);
}

impl<BE: Backend> PatNormalizeReference<BE> for Module<BE>
where
    Self: VecZnxNormalizeTmpBytes + VecZnxNormalizeAssign<BE> + GLWENormalize<BE>,
{
    fn pat_normalize_tmp_bytes_reference(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn glwe_pat_compressed_normalize_assign_reference(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        if res.canonical {
            return;
        }
        let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
        {
            let mut res_be = <GLWEPatCompressedOwned<BE> as GLWECompressedToBackendMut<BE>>::to_backend_mut(res);
            self.vec_znx_normalize_assign(base2k, k, 0, res_be.data_mut(), 0, scratch);
        }
        res.canonical = true;
    }

    fn gglwe_pat_compressed_normalize_assign_reference(
        &self,
        res: &mut GGLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        if res.canonical {
            return;
        }
        let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        {
            let mut res_be = <GGLWEPatCompressedOwned<BE> as GGLWECompressedToBackendMut<BE>>::to_backend_mut(res);
            for row in 0..dnum {
                for col in 0..rank_in {
                    self.vec_znx_normalize_assign(base2k, k, 0, res_be.at_view_mut(row, col).data_mut(), 0, scratch);
                }
            }
        }
        res.canonical = true;
    }

    fn gglwe_pat_normalize_assign_reference(&self, res: &mut GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>) {
        if res.canonical {
            return;
        }
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        {
            let mut res_be = <GGLWEPatOwned<BE> as GGLWEToBackendMut<BE>>::to_backend_mut(res);
            for row in 0..dnum {
                for col in 0..rank_in {
                    self.glwe_normalize_assign(&mut res_be.at_view_mut(row, col), scratch);
                }
            }
        }
        res.canonical = true;
    }
}

pub trait PatFinalizeReference<BE: Backend> {
    fn pat_finalize_tmp_bytes_reference(&self) -> usize;

    fn glwe_pat_compressed_finalize_reference<R>(
        &self,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn gglwe_pat_compressed_finalize_reference<R>(
        &self,
        res: &mut R,
        pat: &GGLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;

    fn gglwe_pat_finalize_reference<R>(&self, res: &mut R, pat: &GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

impl<BE: Backend> PatFinalizeReference<BE> for Module<BE>
where
    Self: VecZnxCopy<BE> + GLWEMaskFill<BE> + GLWENormalize<BE> + GLWEDecompress<Backend = BE> + GGLWEDecompress,
{
    fn pat_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn glwe_pat_compressed_finalize_reference<R>(
        &self,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        assert!(res.glwe_layout() == pat.glwe_layout(), "invalid finalization: layouts differ");
        {
            let mut res_be = res.to_backend_mut();
            let pat_be = <GLWEPatCompressedOwned<BE> as GLWECompressedToBackendRef<BE>>::to_backend_ref(pat);
            self.vec_znx_copy(res_be.data_mut(), 0, pat_be.data(), 0);
        }
        self.fill_glwe_mask_from_seed(res, *pat.seed());
        if pat.canonical {
            res.set_canonical(true);
        } else {
            self.glwe_normalize_assign(res, scratch);
        }
    }

    fn gglwe_pat_compressed_finalize_reference<R>(
        &self,
        res: &mut R,
        pat: &GGLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        assert!(
            res.gglwe_layout() == pat.gglwe_layout(),
            "invalid finalization: layouts differ"
        );
        self.decompress_gglwe(res, pat);
        if !pat.canonical {
            let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
            let mut res_be = res.to_backend_mut();
            for row in 0..dnum {
                for col in 0..rank_in {
                    self.glwe_normalize_assign(&mut res_be.at_view_mut(row, col), scratch);
                }
            }
        }
    }

    fn gglwe_pat_finalize_reference<R>(&self, res: &mut R, pat: &GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        assert!(
            res.gglwe_layout() == pat.gglwe_layout(),
            "invalid finalization: layouts differ"
        );
        let (dnum, rank_in): (usize, usize) = (res.dnum().into(), res.rank_in().into());
        let mut res_be = res.to_backend_mut();
        let pat_be = <GGLWEPatOwned<BE> as GGLWEToBackendRef<BE>>::to_backend_ref(pat);
        for row in 0..dnum {
            for col in 0..rank_in {
                self.glwe_normalize(&mut res_be.at_view_mut(row, col), &pat_be.at_view(row, col), scratch);
            }
        }
    }
}
