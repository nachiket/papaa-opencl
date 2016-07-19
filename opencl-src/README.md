# OpenCL Layers (2D convolution)

In this folder, we provide the set of OpenCL optimization labs for CPUs, GPUs, and FPGAs. Students will focus primarily on the convolution layer as it is the slowest/bottleneck phase of Deep Learning applications. We also provide an MNIST implementation that will be optimized by the students during the OpenCL bakeoff.

1. **cpu-labs/** contains three labs that explore the effect of vectorization, unrolling, and workgroup sizing (threading) on CPU performance.
2. **gpu-labs/** contains one lab that explores the effect of local memory usage on performance on NVIDIA K20 GPU.
3. **fpga-labs/** contains three labs that explore the effect of kernel datapath optimization, memory layout optimization, and compute unit parallelization on the Xilinx AlphaData card.

For curious gawkers, please look at **isolated-layers/** to inspect OpenCL implementations of a few individual layers relevant to MNIST/Lenet5 model.
