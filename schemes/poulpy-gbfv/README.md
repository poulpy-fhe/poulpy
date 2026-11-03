# poulpy-gbfv

`poulpy-gbfv` is the Poulpy crate implementing the GBFV (generalized BFV)
scheme, whose plaintext modulus is a ring element `t(X)` rather than an integer.

The crate currently exposes the plaintext modulus, `GBFVPlaintextModulus`: the
element `t` in the smallest subring holding it, together with its inverse
`numerator / characteristic` in lowest terms.
