//! Helpers for the tests of operations reading a view of a key's first digits.

use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::layouts::{Backend, HostDataRef, Module, ScratchArena, WriterTo};

use crate::{
    GLWENormalize,
    layouts::{
        Dnum, GGLWEInfos, GGLWEPreparedFactory, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEViewWithDigits, GGSWInfos,
        GGSWPreparedFactory, GGSWToBackendMut, GGSWToBackendRef, GGSWViewWithDigits, GLWEInfos,
    },
    test_suite::decryption::panic_message,
};

/// Normalizes each row of `view`, a key read at fewer digits, into `res`, a key
/// of that layout: the key the view reads as.
pub fn gglwe_narrowed_copy<BE, R, A>(module: &Module<BE>, res: &mut R, view: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    Module<BE>: GLWENormalize<BE>,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    A: GGLWEToBackendRef<BE>,
{
    let (dnum, rank_in) = (res.dnum().as_usize(), res.rank_in().as_usize());
    let view = view.to_backend_ref();
    let mut res = res.to_backend_mut();
    for row in 0..dnum {
        for col in 0..rank_in {
            module.glwe_normalize(&mut res.at_view_mut(row, col), &view.at_view(row, col), &mut scratch.borrow());
        }
    }
}

/// [`gglwe_narrowed_copy`] for a GGSW.
pub fn ggsw_narrowed_copy<BE, R, A>(module: &Module<BE>, res: &mut R, view: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    Module<BE>: GLWENormalize<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE>,
{
    let (dnum, cols) = (res.dnum().as_usize(), res.rank().as_usize() + 1);
    let view = view.to_backend_ref();
    let mut res = res.to_backend_mut();
    for row in 0..dnum {
        for col in 0..cols {
            module.glwe_normalize(&mut res.at_view_mut(row, col), &view.at_view(row, col), &mut scratch.borrow());
        }
    }
}

fn assert_refuses(what: &str, op: impl FnOnce(), expected: &str) {
    let err = catch_unwind(AssertUnwindSafe(op)).expect_err(what);
    let message = panic_message(&*err);
    assert!(message.contains(expected), "{what}: panicked with {message:?}");
}

/// Checks what a key refuses for a view of its first `digits` digits: a digit
/// count out of range, serialization of the view and its preparation.
pub fn assert_gglwe_view_refusals<BE, K>(module: &Module<BE>, key: &K, digits: usize, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    Module<BE>: GGLWEPreparedFactory<BE>,
    K: GGLWEToBackendRef<BE> + GGLWEInfos,
{
    assert!(key.view_with_digits(Dnum(0)).is_err(), "a view of no digit");
    assert!(
        key.view_with_digits(Dnum(key.dnum().0 + 1)).is_err(),
        "a view of more digits than the key"
    );
    let view = key.view_with_digits(Dnum(digits as u32)).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let err = GGLWEToBackendRef::<BE>::to_backend_ref(&view)
        .into_inner()
        .write_to(&mut bytes)
        .expect_err("serialized a view");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(bytes.is_empty(), "wrote a refused view");
    let mut prepared = module.gglwe_prepared_alloc_from_infos(&view);
    assert_refuses(
        "prepared a view",
        || module.gglwe_prepare(&mut prepared, &view, scratch),
        "gglwe_prepare: the source is a view of fewer digits",
    );
}

/// [`assert_gglwe_view_refusals`] for a GGSW.
pub fn assert_ggsw_view_refusals<BE, G>(module: &Module<BE>, ggsw: &G, digits: usize, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    Module<BE>: GGSWPreparedFactory<BE>,
    G: GGSWToBackendRef<BE> + GGSWInfos,
{
    assert!(ggsw.view_with_digits(Dnum(0)).is_err(), "a view of no digit");
    assert!(
        ggsw.view_with_digits(Dnum(ggsw.dnum().0 + 1)).is_err(),
        "a view of more digits than the GGSW"
    );
    let view = ggsw.view_with_digits(Dnum(digits as u32)).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let err = GGSWToBackendRef::<BE>::to_backend_ref(&view)
        .into_inner()
        .write_to(&mut bytes)
        .expect_err("serialized a view");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(bytes.is_empty(), "wrote a refused view");
    let mut prepared = module.ggsw_prepared_alloc_from_infos(&view);
    assert_refuses(
        "prepared a view",
        || module.ggsw_prepare(&mut prepared, &view, scratch),
        "ggsw_prepare: the source is a view of fewer digits",
    );
}
