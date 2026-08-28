//! A prepared GGLWE key paired with the bound it is being used through.
//!
//! Every GGLWE product takes one of these rather than a key and a bound side by
//! side: the pairing is checked once, here, so a key restored with metadata that
//! disagrees with its buffer, or a bound resolved from a different key of the
//! same shape, cannot reach a kernel.

use poulpy_hal::layouts::{Backend, DataView, VmpPMatBackendRef};

use crate::{
    error::{CoreError, Result},
    layouts::{GGLWEActiveUse, GGLWEInfos, GGLWELayout, LWEInfos, prepared::GGLWEPreparedBackendRef},
};

fn err(detail: String) -> CoreError {
    CoreError::GGLWEKeyUse {
        op: "GGLWEPreparedBound::new",
        detail,
    }
}

fn eq<T: PartialEq + std::fmt::Display>(have: T, want: T, what: &str) -> Result<()> {
    if have == want {
        Ok(())
    } else {
        Err(err(format!("prepared {what} is {have}, the bound resolved {want}")))
    }
}

/// A complete physical prepared key and the use resolved from it.
pub struct GGLWEPreparedBound<'a, B: Backend + 'a> {
    key: GGLWEPreparedBackendRef<'a, B>,
    use_: GGLWEActiveUse,
}

impl<'a, B: Backend + 'a> GGLWEPreparedBound<'a, B> {
    /// Pairs `key` with `use_`, or reports why they do not describe each other.
    ///
    /// Checks the stored geometry (degree, rows, both column counts, limb pitch)
    /// and the stored decomposition (`base2k`, `dsize`, `k_aux`), so two keys of
    /// the same shape but different radix, digit or guard cannot be swapped;
    /// then the selected rows and limb prefix, and that the backing covers the
    /// shape the key claims.
    pub fn new(key: GGLWEPreparedBackendRef<'a, B>, use_: GGLWEActiveUse) -> Result<Self> {
        let logical: &GGLWELayout = use_.logical_layout();
        let data = &key.data;

        eq(data.n(), logical.n().as_usize(), "degree")?;
        eq(data.rows(), use_.physical_rows(), "rows")?;
        eq(data.cols_in(), logical.rank_in().as_usize(), "input columns")?;
        eq(data.cols_out(), (logical.rank_out() + 1).as_usize(), "output columns")?;
        // A bound is always resolved from a complete key, never from a projection.
        eq(data.size(), use_.physical_size(), "limb pitch")?;

        let (physical_base2k, physical_dsize, physical_k_aux) = use_.physical_gadget();
        eq(key.base2k(), physical_base2k, "base2k")?;
        eq(key.dsize(), physical_dsize, "dsize")?;
        eq(key.k_aux(), physical_k_aux, "k_aux")?;

        if use_.logical_work_size() > data.size() {
            return Err(err(format!(
                "logical work size {} exceeds the stored pitch {}",
                use_.logical_work_size(),
                data.size()
            )));
        }

        // The last selected row, computed without wrapping.
        let last_row: Option<usize> = logical
            .dnum()
            .as_usize()
            .checked_sub(1)
            .and_then(|i| i.checked_mul(use_.physical_row_step().get()))
            .and_then(|o| o.checked_add(use_.first_physical_row()));
        match last_row {
            Some(last) if last < data.rows() => {}
            _ => {
                return Err(err(format!(
                    "selected rows {}..={last_row:?} step {} exceed the stored {}",
                    use_.first_physical_row(),
                    use_.physical_row_step(),
                    data.rows()
                )));
            }
        }

        let need: usize = B::bytes_of_vmp_pmat(data.n(), data.rows(), data.cols_in(), data.cols_out(), data.size());
        if B::len_bytes_ref(DataView::data(data)) < need {
            return Err(err("backing is shorter than its own shape requires".to_string()));
        }

        Ok(Self { key, use_ })
    }

    pub fn key(&self) -> &GGLWEPreparedBackendRef<'a, B> {
        &self.key
    }

    /// The complete physical matrix. Reads must stay inside [`Self::use_`].
    pub fn pmat(&self) -> &VmpPMatBackendRef<'a, B> {
        &self.key.data
    }

    pub fn use_(&self) -> &GGLWEActiveUse {
        &self.use_
    }
}
