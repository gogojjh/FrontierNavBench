# SRU-Odin inference dependency

This directory retains the **MIT-licensed SRU-Odin inference package** used by the
benchmark's CPU point-goal controller.

| Location | Purpose |
| --- | --- |
| [`SRU-Odin/Deployment/src/sru_nav_go2/`](SRU-Odin/Deployment/src/sru_nav_go2/) | ONNX model wrapper, constants and NumPy transform helpers |
| [`SRU-Odin/LICENSE`](SRU-Odin/LICENSE) | Original MIT notice |
| [`../assets/weights/sru/`](../assets/weights/sru/) | Pretrained ONNX model files |

Each retained source header records its upstream path and original SHA256.
The benchmark starts the controller in its own Python environment; setup is
explained in the [main README](../README.md#set-up-simulation).

Go2 simulation is maintained in
[`vnav_benchmark/sim/`](../vnav_benchmark/sim/). Model checksums and other asset
credits are described in the [asset guide](../assets/README.md).
