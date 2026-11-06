use poulpy_hal::layouts::Module;

use crate::oep::CKKSPow2Impl;

pub(crate) fn ckks_double_tmp_bytes_derived<BE: CKKSPow2Impl>(module: &Module<BE>, res_size: usize) -> usize {
    BE::ckks_mul_pow2_tmp_bytes(module, res_size)
}
