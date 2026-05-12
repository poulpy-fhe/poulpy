use poulpy_hal::{
    api::{
        ModuleN, VecZnxAddScalarAssignBackend, VecZnxDftBytesOf, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
        VecZnxZeroBackend,
    },
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

pub use crate::api::GGLWEEncryptSk;
use crate::{
    EncryptionInfos, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeCore,
    encryption::glwe::normalize_scratch_vec_znx,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GGLWEEncryptSkDefault<BE: Backend> {
    fn gglwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GGLWEEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxZeroBackend<BE>,
{
    fn gglwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn gglwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let pt_backend = pt.to_backend_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(
            res.rank_in(),
            pt_backend.cols() as u32,
            "res.rank_in(): {} != pt.cols(): {}",
            res.rank_in(),
            pt_backend.cols()
        );
        assert_eq!(
            res.rank_out(),
            sk_ref.rank(),
            "res.rank_out(): {} != sk.rank(): {}",
            res.rank_out(),
            sk_ref.rank()
        );
        assert_eq!(res.n(), sk_ref.n());
        assert_eq!(pt_backend.n() as u32, sk_ref.n());
        assert!(
            scratch.available() >= GGLWEEncryptSkDefault::gglwe_encrypt_sk_tmp_bytes_default(self, res),
            "scratch.available(): {} < GGLWEEncryptSk::gglwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            GGLWEEncryptSkDefault::gglwe_encrypt_sk_tmp_bytes_default(self, res)
        );
        assert!(
            res.dnum().0 * res.dsize().0 * res.base2k().0 <= res.max_k().0,
            "res.dnum() : {} * res.dsize() : {} * res.base2k() : {} = {} >= res.k() = {}",
            res.dnum(),
            res.dsize(),
            res.base2k(),
            res.dnum().0 * res.dsize().0 * res.base2k().0,
            res.max_k()
        );

        let dnum: usize = res.dnum().into();
        let dsize: usize = res.dsize().into();
        let base2k: usize = res.base2k().into();
        let rank_in: usize = res.rank_in().into();
        let cols: usize = (res.rank_out() + 1).into();
        let scratch = scratch.borrow();
        let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext_scratch(res);

        // For each input column (i.e. rank) produces a GGLWE of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s1', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                self.vec_znx_zero_backend(&mut tmp_pt.data, 0);
                {
                    let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                    self.vec_znx_add_scalar_assign_backend(
                        &mut tmp_pt_backend.data,
                        0,
                        (dsize - 1) + row_i * dsize,
                        &pt_backend,
                        col_i,
                    );
                }
                scratch_1.scope(|mut scratch| {
                    normalize_scratch_vec_znx(self, base2k, &mut tmp_pt.data, &mut scratch);
                });
                {
                    let mut scratch = scratch_1.borrow();
                    let tmp_pt_backend = tmp_pt.to_backend_ref();
                    let mut ct: GLWEViewMut<'_, BE> = res.at_view_mut(row_i, col_i);
                    <Module<BE> as GLWEEncryptSkInternal<BE>>::glwe_encrypt_sk_internal(
                        self,
                        base2k,
                        &mut ct.data,
                        cols,
                        false,
                        Some((tmp_pt_backend, 0)),
                        sk,
                        enc_infos,
                        source_xe,
                        source_xa,
                        &mut scratch,
                    );
                }
            }
        }
    }
}
