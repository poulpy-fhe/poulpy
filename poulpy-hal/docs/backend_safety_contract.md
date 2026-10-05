Implementors must uphold all of the following for **every** call:

* **Memory domains**: Backend-native borrows must be valid in the target
  execution domain for `Self` (e.g., CPU host memory or a specific GPU).
  Host-view helpers such as `to_ref()` / `to_mut()` require host-accessible
  storage. If host/device transfers are required, perform them through the
  backend transfer hooks; do not assume the caller synchronized.

* **Alignment & layout**: All data must match the layout, stride, and element
  size expected by the kernel. `size()`, `rows()`, `cols_in()`, `cols_out()` and
  `n()` describe the logical shape defined by the HAL contracts; prepared and
  DFT storage may use a backend-specific representation. Declare
  `Backend::DFT_LIMBS_CONTIGUOUS` only when each DFT limb is one contiguous block
  containing equal-sized column blocks in column order, and a range of limbs
  is a valid independent DFT vector with the same encoding. Partial
  `with_limb_range_mut` / `with_size_mut` views and generic host `zero_at` require
  that capability; whole-buffer reborrows preserve every layout. Reference
  compositions that require narrowing must declare and enforce that requirement.

* **Scratch lifetime**: Any region carved from a `ScratchArena` must remain
  valid for the duration of the call; it may be reused by the caller
  afterwards.
  Do not retain pointers past return, except for deferred work under the synchronization contract below.

* **Synchronization**: The call must appear **logically synchronous** to the caller: every later call observes its complete effect.
  A backend may defer execution (e.g., CUDA streams) only under the following contract, which is the same for all backends.

  * **Order**: deferred work takes effect in submission order on every buffer it reads or writes.
    A caller can therefore reuse a scratch region or overwrite an operand as soon as the call returns.
  * **Transfers**: a hook that reads backend storage (`to_host_bytes`, `copy_to_host`, `copy_view_to_host`) first waits for the pending work on that storage.
    Results are observable without an explicit wait.
  * **Explicit wait**: `HalModuleImpl::synchronize`, reached through `ModuleSynchronize::synchronize`, returns only when every operation previously submitted through that module has completed.
    Timed regions must end with it.
  * **Host memory**: host data borrowed by a call (the slices of the transfer hooks, sampling sources) is not accessed after the call returns.
  * **Release**: storage is not released while pending work refers to it.
    Dropping an owned buffer, which includes a scratch arena, waits for that work or defers the release.
  * **Failures**: every condition a call reports by panic or error is detected before it returns.
    A device failure that surfaces later is reported by panic no later than the next transfer or explicit wait.

  A backend that completes each call before returning satisfies all of the above and keeps the default no-op `synchronize`.

* **Aliasing & overlaps**: If res, a, b, etc... alias or overlap in ways
  that violate your kernel’s requirements, you must either handle safely or reject
  with a defined error path (e.g., debug assert). Never trigger UB.

* **Numerical contract**: For modular/integer arithmetic, results must be
  bit-exact to the specification. For floating-point, any permitted tolerance
  must be documented and consistent with the crate’s guarantees.
