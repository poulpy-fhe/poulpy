# poulpy-gbfv

`poulpy-gbfv` is the Poulpy crate implementing the GBFV (generalized BFV)
scheme, whose plaintext modulus is a ring element `t(X)` rather than an integer.

The crate currently exposes:

- `GBFVPlaintextModulus`: the element `t` in the smallest subring holding it,
  together with its inverse `numerator / characteristic` in lowest terms;
- `GBFVBatchEncoder`: the root of unity modulo `p^e` shared by the plaintext
  moduli over one prime `p`;
- `GBFVNttOps`: the negacyclic NTT over `Z/q` for a modulus of any size, on
  backend-resident operands, dispatched through the `GBFVNttImpl` backend
  contract, with the reference body selected by `impl_gbfv_ntt_reference_full!`.
