## Description
This lab is on parallelizing the computation using multiple identical hardware units. In OpenCL terminology these are 
called Compute Units(CU). We created and optimized a single CU in previous two labs. In this lab we will instantiate 2 such CUs in the FPGA and divide the task among them. The SDAccel tool will internally connect all CUs to the control and memory infrastructure.
- The only difference to the kernel code in this lab wrt. previous lab is that a single CU is going to produce half of the output image. For example, in this case, the input image is 28x28 and the output image is going to be 26x26(neglecting the borders). Each CU is going to produce 13 rows of the final output image.
- The workgroup size in the _y_ dimension(row dimension) is reduced to half of that in the previous lab.
- The important point to note here is that we need FILER_SIZE-1 extra input image rows to produce half of the output. In this case each CU need to copy 15 rows of the input image into local memory in order to produce 13 rows of the output image.
- All the work items in the workgroup will copy their respective input pixels. The work items in the last 2 rows is responsilbe for copying those 2 extra rows.
- As in the previous lab, we partition the local memory to store the rows across 3 RAMs and we store the filter coefficients in the registers.
- Apart from these kernel modifications, we need to tell the SDAccel tool to instantiate 2 CUs in the final FPGA bitstream. This is done by introducing one more _create_compute_unit_ command in the TCL script. Note that the name for two compute units should be different. Both of them are part of the same binary.


## Steps to run
_make_all_ : to run build the kernel and host application and perform CPU simulation followed by the resource estimate.

Note that there are 2 CUs present in the resource estimate report.
