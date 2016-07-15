## Description
This lab introduces OpenCL programming using a simple 2D convolution application kernel. We will run this lab on CPU as target device.
- We assume that you are familiar with the host program by now !
- The kernel code is simple 2D convolution using 3x3 filter. Each work item is going to compute 1 output pixel.
- We need to get the image width information to properly store the output pixel in the output buffer. We use _get_global_size(0)_ for that.
- The work item uses _get_global_id(dim)_ to know the co-ordinates of the output pixel that should be computed by it.
- Image pixels are read from global memory where as the filter coefficients are read form constant memory.
- Output pixel is written back to global memory.

# Steps to run
make all : to compile the host application.

make run : to run the application. Note that the kernel is compiled by the host application during the runtime.
