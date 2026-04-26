use poulpy_hal::{
    api::{ScratchAvailable, VecZnxAutomorphism, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes},
    layouts::{Backend, CyclotomicOrder, GaloisElement, Module, Scratch},
};

use crate::{
    GLWEKeyswitch, ScratchTakeCore,
    layouts::{GGLWE, GGLWEInfos, GGLWEPreparedToRef, GGLWEToMut, GGLWEToRef, GLWE, GetGaloisElement, SetGaloisElement},
};

pub(crate) trait GLWEAutomorphismKeyAutomorphismDefault<BE: Backend>:
    Sized
    + GaloisElement
    + GLWEKeyswitch<BE>
    + VecZnxAutomorphism
    + VecZnxAutomorphismAssign<BE>
    + VecZnxAutomorphismAssignTmpBytes
    + CyclotomicOrder
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = if res_infos.glwe_layout() == a_infos.glwe_layout() {
            self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
        } else {
            self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos)
        };
        let lvl_1: usize = self.vec_znx_automorphism_assign_tmp_bytes();

        lvl_0.max(lvl_1)
    }

    fn glwe_automorphism_key_automorphism_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        assert!(
            res.dnum().as_u32() <= a.dnum().as_u32(),
            "res dnum: {} > a dnum: {}",
            res.dnum(),
            a.dnum()
        );

        assert_eq!(res.dsize(), a.dsize(), "res dnum: {} != a dnum: {}", res.dsize(), a.dsize());

        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, a, key)
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();

        let p: i64 = a.p();
        let p_inv: i64 = self.galois_element_inv(p);

        let same_layout: bool = res.glwe_layout() == a.glwe_layout();

        {
            let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
            let a: &GGLWE<&[u8]> = &a.to_ref();

            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp: GLWE<&mut [u8]> = res.at_mut(row, col);
                    let a_ct: GLWE<&[u8]> = a.at(row, col);

                    if same_layout {
                        // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                        for i in 0..cols_out {
                            self.vec_znx_automorphism(p, res_tmp.data_mut(), i, &a_ct.data, i);
                        }

                        // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                        self.glwe_keyswitch_assign(&mut res_tmp, key, scratch);
                    } else {
                        let (mut tmp_glwe, scratch_1) = scratch.take_glwe(a);

                        // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                        for i in 0..cols_out {
                            self.vec_znx_automorphism(p, tmp_glwe.data_mut(), i, &a_ct.data, i);
                        }

                        // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                        self.glwe_keyswitch(&mut res_tmp, &tmp_glwe, key, scratch_1);
                    }

                    // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_assign(p_inv, res_tmp.data_mut(), i, scratch);
                    }
                }
            }
        }

        res.set_p((p * key.p()) % self.cyclotomic_order());
    }

    fn glwe_automorphism_key_automorphism_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(res.rank(), key.rank(), "key rank: {} != key rank: {}", res.rank(), key.rank());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, res, key)
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();
        let p: i64 = res.p();
        let p_inv: i64 = self.galois_element_inv(p);

        {
            let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp: GLWE<&mut [u8]> = res.at_mut(row, col);

                    // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_assign(p, res_tmp.data_mut(), i, scratch);
                    }

                    // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                    self.glwe_keyswitch_assign(&mut res_tmp, key, scratch);

                    // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_assign(p_inv, res_tmp.data_mut(), i, scratch);
                    }
                }
            }
        }

        res.set_p((res.p() * key.p()) % self.cyclotomic_order());
    }
}

impl<BE: Backend> GLWEAutomorphismKeyAutomorphismDefault<BE> for Module<BE>
where
    Self: GaloisElement
        + GLWEKeyswitch<BE>
        + VecZnxAutomorphism
        + VecZnxAutomorphismAssign<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + CyclotomicOrder,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
pub use crate::api::GLWEAutomorphismKeyAutomorphism;
