# ResNet-20 CIFAR-10 weights

Built into the crate with `include_bytes!`, so the binaries read no weight file at run time.

- `manifest.json`: tensor names, shapes, byte offsets and the SHA-256 of the payload.
- `weights.f32.bin`: one little-endian `float32` payload, tensors in manifest order, row-major.

A PyTorch ResNet-20 trained on CIFAR-10 for 240 epochs, batch norms as trained: 93.07% on the test set with the exact ReLU.
