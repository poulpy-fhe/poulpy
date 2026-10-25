//! The API on `Module<BE>`: each call is a plain dispatch to the backend's `*Impl`, which validates its inputs, writes
//! `res` and, on `Ok`, stamps its batch record.

mod inter;
mod intra;
mod matmul;
