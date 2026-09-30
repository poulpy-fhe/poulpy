# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body.
- `GGLWEPatCompressed`: seeded GGLWE bodies.
- `GGLWEPat`: unseeded full GGLWE.
- Unseeded GLWE transcripts are core `GLWE`s.

Allocate PATs through `MHEModuleAlloc` on a `Module`.

## Randomness

A public mask seed is common to all parties contributing to one protocol
result. Use a fresh seed for every new result, separated by protocol, session
and key identity. A key set therefore needs
separate seeds derived from the common reference string.

Keep `source_xe` (errors)
private. Seed its streams
independently for each party and purpose, and consume fresh samples without
replaying a stream. Never initialize a private stream from a public mask seed.
