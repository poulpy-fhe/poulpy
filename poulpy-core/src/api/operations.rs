use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ModuleLogN, ModuleN, ScratchAvailable, VecZnxAddAssign, VecZnxAddInto, VecZnxCopy, VecZnxLsh, VecZnxLshAddInto,
        VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign, VecZnxNegate,
        VecZnxNegateAssign, VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign,
        VecZnxRotateAssignTmpBytes, VecZnxRshAssign, VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign,
        VecZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, GaloisElement, Scratch},
};

use crate::{
    ScratchTakeCore,
    api::GLWEAutomorphism,
    glwe_packer::{GLWEPacker, pack_core},
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos,
        GLWEPlaintext, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos,
    },
};

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_trace_assign<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

pub trait GLWEPackerOps<BE: Backend>
where
    Self: Sized
        + ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
{
    fn packer_add<A, K, H>(&self, packer: &mut GLWEPacker, a: Option<&A>, i: usize, auto_keys: &H, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        pack_core(self, a, &mut packer.accumulators, i, auto_keys, scratch)
    }
}

pub trait GLWEMulConst<BE: Backend> {
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const<R, A>(&self, cnv_offset: usize, res: &mut GLWE<R>, a: &GLWE<A>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
        A: DataRef;

    fn glwe_mul_const_assign<R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut;
}

pub trait GLWEMulPlain<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;
}

pub trait GLWETensoring<BE: Backend> {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        self.glwe_tensor_apply_tmp_bytes(res, a, a)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_add_assign<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;
}

pub trait GLWEAdd
where
    Self: ModuleN + VecZnxAddInto + VecZnxCopy + VecZnxAddAssign + VecZnxZero,
{
    fn glwe_add_into<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        B: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let b: &GLWE<&[u8]> = &b.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), b.base2k());
        assert_eq!(res.base2k(), b.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_add_into(res.data_mut(), i, a.data(), i, b.data(), i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, b.data(), i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_add_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_add_assign(res.data_mut(), i, a.data(), i);
        }
    }
}

pub trait GLWENegate
where
    Self: VecZnxNegate + VecZnxNegateAssign + VecZnxZero + ModuleN,
{
    fn glwe_negate<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.rank(), res.rank());
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate(res.data_mut(), i, a.data(), i);
        }
        res.base2k = a.base2k;
    }

    fn glwe_negate_assign<R>(&self, res: &mut R)
    where
        R: GLWEToMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert_eq!(res.n(), self.n() as u32);
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate_assign(res.data_mut(), i);
        }
    }
}

pub trait GLWESub
where
    Self: ModuleN + VecZnxSub + VecZnxCopy + VecZnxNegate + VecZnxZero + VecZnxSubAssign + VecZnxSubNegateAssign,
{
    fn glwe_sub<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        B: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let b: &GLWE<&[u8]> = &b.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), res.base2k());
        assert_eq!(b.base2k(), res.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_sub(res.data_mut(), i, a.data(), i, b.data(), i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_negate(res.data_mut(), i, b.data(), i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_sub_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_assign(res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_sub_negate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_negate_assign(res.data_mut(), i, a.data(), i);
        }
    }
}

pub trait GLWERotate<BE: Backend>
where
    Self: ModuleN + VecZnxRotate + VecZnxRotateAssign<BE> + VecZnxRotateAssignTmpBytes + VecZnxZero,
{
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        self.vec_znx_rotate_assign_tmp_bytes()
    }

    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let res_cols = (res.rank() + 1).into();
        let a_cols = (a.rank() + 1).into();

        for i in 0..a_cols {
            self.vec_znx_rotate(k, res.data_mut(), i, a.data(), i);
        }
        for i in a_cols..res_cols {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_rotate_tmp_bytes(),
            "scratch.available(): {} < GLWERotate::glwe_rotate_tmp_bytes: {}",
            scratch.available(),
            self.glwe_rotate_tmp_bytes()
        );

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_rotate_assign(k, res.data_mut(), i, scratch);
        }
    }
}

pub trait GGSWRotate<BE: Backend>
where
    Self: GLWERotate<BE>,
{
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        self.glwe_rotate_tmp_bytes()
    }

    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.rank(), a.rank());
        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate(k, &mut res.at_mut(row, col), &a.at(row, col));
            }
        }
    }

    fn ggsw_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
    {
        assert!(
            scratch.available() >= self.ggsw_rotate_tmp_bytes(),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_rotate_tmp_bytes()
        );
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate_assign(k, &mut res.at_mut(row, col), scratch);
            }
        }
    }
}

pub trait GLWEMulXpMinusOne<BE: Backend>
where
    Self: ModuleN + VecZnxMulXpMinusOne + VecZnxMulXpMinusOneAssign<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one(k, res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.n(), self.n() as u32);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_assign(k, res.data_mut(), i, scratch);
        }
    }
}

pub trait GLWECopy
where
    Self: ModuleN + VecZnxCopy + VecZnxZero,
{
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let min_rank: usize = res.rank().min(a.rank()).as_usize() + 1;

        for i in 0..min_rank {
            self.vec_znx_copy(res.data_mut(), i, a.data(), i);
        }

        for i in min_rank..(res.rank() + 1).into() {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }
}

pub trait GLWEShift<BE: Backend>
where
    Self: ModuleN
        + VecZnxRshAssign<BE>
        + VecZnxLshAddInto<BE>
        + VecZnxLshSub<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssign<BE>
        + VecZnxLsh<BE>,
{
    fn glwe_shift_tmp_bytes(&self) -> usize {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );
        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_rsh_assign(base2k, k, res.data_mut(), i, scratch);
        }
    }

    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();

        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_assign(base2k, k, res.data_mut(), i, scratch);
        }
    }

    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }

    fn glwe_lsh_add<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_add_into(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }

    fn glwe_lsh_sub<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res = &mut res.to_mut();
        let a = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_sub(base2k, k, res.data_mut(), i, a.data(), i, scratch);
        }
    }
}

pub trait GLWENormalize<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes()
    }

    fn glwe_maybe_cross_normalize_to_ref<'a, A>(
        &self,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return (glwe.to_ref(), scratch);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        (tmp_ref.to_ref(), scratch2)
    }

    fn glwe_maybe_cross_normalize_to_mut<'a, A>(
        &self,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return (glwe.to_mut(), scratch);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        (tmp_ref.to_mut(), scratch2)
    }

    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );

        let res_base2k = res.base2k().into();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize(res.data_mut(), res_base2k, 0, i, a.data(), a.base2k().into(), i, scratch);
        }
    }

    fn glwe_normalize_assign<R>(&self, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize_assign(res.base2k().into(), res.data_mut(), i, scratch);
        }
    }
}
