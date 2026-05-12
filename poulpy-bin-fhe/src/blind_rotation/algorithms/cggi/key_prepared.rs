use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena, SvpPPolOwned},
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

impl<BE: Backend> BlindRotationKeyPreparedFactory<CGGI, BE> for Module<BE>
where
    Self: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
    BE::OwnedBuf: HostDataMut + HostDataRef,
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
        other: &BlindRotationKey<BE::OwnedBuf, CGGI>,
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
            let mut buf: ScalarZnx<BE::OwnedBuf> = self.scalar_znx_alloc(1);
            (0..n << 1).for_each(|i| {
                let mut res: SvpPPolOwned<BE> = self.svp_ppol_alloc(1);
                set_xai_plus_y(self, i, 0, &mut res, &mut buf);
                x_pow_a.push(res);
            });
            res.x_pow_a = Some(x_pow_a);
        }
    }
}
