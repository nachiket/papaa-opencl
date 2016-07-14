## Description
This lab involves implementation of simple 2D convolution kernel commanly used in Convolutional Neural Networks(CNN) using OpenCL for Xilinx FPGAs. Starting with an un-optimized implementation, we will explore the simple **loop unrolling, loop pipelining and work item pipelining**.

- This implementation is fixed for image and filter size, because we will buffer the entire image into on-chip(local)memory and then perform computations. Hence, the max image resolution is limited by the FPGA resources.
- The local work size is equal to gloal work size in all dimensions because we copy the entire image in a single work group.
- Each work item will copy a single pixel from the global memory into the local memory and wait for all other work items to finish copying their respective pixels.
- The read process is pipelined across all work items and thus the SDAceel tool can optimize the HW to issue outstanding memory requests.
- The computation involves reading the pixels from 3x3 window and performing dot-product. Even the compute loop is unrolled and pipelined to perform parallel and pipelined computations within the work item. For example, reading pixels from the local BRAM can be overlapped with the MAC operation. Pipelining hint allows the tool to generate such hardware.

## Steps to run
_make all_ : to compile and run the CPU simulation and generate the HW usage and latency report.

You can also try the kernel __conv_2d which does not use the local memory. To run this kernel, just swap the kernel name __conv_2d with conv_2d and do _make all_
