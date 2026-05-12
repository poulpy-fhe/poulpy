Implementors must uphold all of the following for **every** call:

* **Memory domains**: Pointers produced by to_ref() / to_mut() must be valid
  in the target execution domain for Self (e.g., CPU host memory for CPU,
  device memory for a specific GPU). If host↔device transfers are required,
  perform them inside the implementation; do not assume the caller synchronized.

* **Alignment & layout**: All data must match the layout, stride, and element
  size expected by the kernel. size(), rows(), cols_in(), cols_out(),
  n(), etc... must be interpreted identically to the reference CPU implementation.

* **Scratch lifetime**: Any region carved from a `ScratchArena` must remain
  valid for the duration of the call; it may be reused by the caller
  afterwards. Do not retain pointers past return.

* **Synchronization**: The call must appear **logically synchronous** to the
  caller. If you enqueue asynchronous work (e.g., CUDA streams), you must
  ensure completion before returning or clearly document and implement a
  synchronization contract used by all backends consistently.

* **Aliasing & overlaps**: If res, a, b, etc... alias or overlap in ways
  that violate your kernel’s requirements, you must either handle safely or reject
  with a defined error path (e.g., debug assert). Never trigger UB.

* **Numerical contract**: For modular/integer arithmetic, results must be
  bit-exact to the specification. For floating-point, any permitted tolerance
  must be documented and consistent with the crate’s guarantees.
