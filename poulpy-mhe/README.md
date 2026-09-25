# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body (collective public key, share-to-encryption).
- `GGLWEPatCompressed`: seeded GGLWE bodies (switching and automorphism keys).
- `GGLWEPat`: unseeded full GGLWE (public-key based relinearization key).
- Unseeded GLWE transcripts (collective key switching) are core `GLWE`s.

A PAT carries a canonical flag; `write_to` requires it set. Allocate PATs
through `MHEModuleAlloc` on a `Module`.
