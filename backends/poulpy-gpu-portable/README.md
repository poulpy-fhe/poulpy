# Poulpy-GPU-Portable

**Poulpy-GPU-Portable** provides GPU backends for Poulpy on [CubeCL](https://github.com/tracel-ai/cubecl): the NTT4x30 arithmetic of the CPU backends, bit-identical to `NTT4x30Portable`, on CUDA (`NTT4x30Cuda`), HIP (`NTT4x30Hip`) and wgpu (`NTT4x30Wgpu`: Metal, Vulkan, WebGPU).

It implements the extension traits of [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal) and, with `enable-core` and `enable-ckks`, of [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core) and [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks).

| Feature | Backend type | Requirement |
|---|---|---|
| `enable-cuda` | `NTT4x30Cuda` | CUDA toolkit |
| `enable-hip` | `NTT4x30Hip` | ROCm/HIP toolkit |
| `enable-metal`, `enable-vulkan`, `enable-wgpu` | `NTT4x30Wgpu` | a Metal, Vulkan or WebGPU device |

CKKS encoding and decoding, the uniform fill of a coefficient column and the normalizations at radices above 62 bits are staged through the host, and everything else runs on the device, sampling included.
Scheme operations are the reference bodies of `poulpy-core` and `poulpy-ckks`; the backend implements their step hooks with fused launches: the gadget product adds, rotates and accumulates as it stores, or runs the first launch of its inverse transform, the products of several sources against one key share a launch that reads the key once, a square reads one prepared operand, the giant-step products of a linear transformation are written a batch per launch, and the terms of a constant combination are summed in one launch.
A CPU backend such as `NTT4x30Portable` can generate the keys and encrypt, and the GPU backend compiles the bootstrapping context, prepares the keys and bootstraps.

```bash
cargo test --release -p poulpy-gpu-portable --features enable-metal,enable-ckks
cargo bench -p poulpy-gpu-portable --bench bootstrapping --features enable-metal,enable-ckks
```

See `docs/guides/backends.md` for the runtimes, graph replay and the module's tuning.
