//! The API on `Module<BE>`: each call is a plain dispatch to the backend's `*Impl`, which validates its inputs, writes
//! `res` and, for a grid operation on `Ok`, stamps its batch record.

mod encoding;
mod inter;
mod intra;
mod matmul;
