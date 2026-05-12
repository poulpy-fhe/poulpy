use poulpy_hal::{
    api::SvpPrepare,
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, SvpPPolOwned, SvpPPolToBackendMut, ZnxViewMut},
};

pub(crate) fn set_xai_plus_y<C, B: Backend>(
    module: &Module<B>,
    ai: usize,
    y: i64,
    res: &mut SvpPPolOwned<B>,
    buf: &mut ScalarZnx<C>,
) where
    C: HostDataMut,
    Module<B>: SvpPrepare<B>,
{
    let n: usize = res.n();

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);
        if ai < n {
            raw[ai] = 1;
        } else {
            raw[(ai - n) & (n - 1)] = -1;
        }
        raw[0] += y;
    }

    let mut res_backend = res.to_backend_mut();
    let buf_ref = buf.to_ref();
    let buf_backend = ScalarZnx::from_data(B::from_host_bytes(buf_ref.data), buf_ref.n(), buf_ref.cols());
    module.svp_prepare(
        &mut res_backend,
        0,
        &<ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendRef<B>>::to_backend_ref(&buf_backend),
        0,
    );

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);

        if ai < n {
            raw[ai] = 0;
        } else {
            raw[(ai - n) & (n - 1)] = 0;
        }
        raw[0] = 0;
    }
}
