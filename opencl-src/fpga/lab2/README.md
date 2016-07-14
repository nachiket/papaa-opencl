## Description
We will extend the kernel optimizations that we did in Lab1 here. We will focus on optimizing memory structures for 2D convolution operation. All the optimizations that we did in Lab1 remain in this lab. On top of it we will instruct the tool to structure our local memory that we used to buffer the image and filter kernel(not to be confused with OpenCL kernel!)
- The inner _for_ loop in the compute portion of the kernel involves reading 3 pixels and 3 filter coefficients from the local memory. In FPGAs, the local memory is realized using on-chip BRAMs. Generally BRAMs provide 1 write and 1 read port and thus reading 3 consecutive pixels takes 3 cycles.
- One way to overcome this problem is to store the adjacent image colums staggered across 3 separate BRAMs. Thus the compute unit(mostly DSP slices in this case) can access all 3 pixels in a single clock cycle. The attribute _xcl_array_partition(cyclic,3,1)_ on local image buffer does exactly that.
- Even though we provided parallel access to the image pixels, the compute unit still cannot perform 1 inner loop iteration in a single cycle because the operation involves reading the filter coefficients from the local RAMs
- In the next step, we will instruct the tool to store all the filter coefficients in a set of registers by specifying _xcl_array_partition(complete,1)_ attribute on the filter buffer. Thus we have parallel access to the pixels as well as the filter coefficients and the unrolling and pipelining should take full advantage of this.
- All these optimizations improve performance costing extra FPGA resources.

## Steps to run
_make all_ : to compile and run the CPU simulation and generate the HW usage and latency report.


