mod cst;
mod vec;

pub use cst::{CKKSConstPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx};
pub use vec::{CKKSPlaintextConversion, CKKSPlaintextVecRnx, CKKSPlaintextVecZnx, alloc_pt_vec_znx, alloc_pt_znx};

/// Conventional alias for vector CKKS plaintexts in RNX form.
pub type CKKSPlaintextRnx<F> = CKKSPlaintextVecRnx<F>;
/// Conventional alias for vector CKKS plaintexts in ZNX form.
pub type CKKSPlaintextZnx<D> = CKKSPlaintextVecZnx<D>;
