/*******************************************************************************
Vendor: Xilinx
Associated Filename: main.c
#Purpose: An example showing new kernels can be downloaded to FPGA while keeping
#         the data in device memory intact
#*******************************************************************************
Copyright (c) 2016, Xilinx, Inc.^M
All rights reserved.^M
^M
Redistribution and use in source and binary forms, with or without modification, ^M
are permitted provided that the following conditions are met:^M
^M
1. Redistributions of source code must retain the above copyright notice, ^M
this list of conditions and the following disclaimer.^M
^M
2. Redistributions in binary form must reproduce the above copyright notice, ^M
this list of conditions and the following disclaimer in the documentation ^M
and/or other materials provided with the distribution.^M
^M
3. Neither the name of the copyright holder nor the names of its contributors ^M
may be used to endorse or promote products derived from this software ^M
without specific prior written permission.^M
^M
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ^M
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, ^M
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. ^M
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, ^M
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, ^M
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) ^M
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, ^M
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, ^M
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <CL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

#define FILTER_SIZE    (3)
#define IMAGE_HEIGHT   (10)
#define IMAGE_WIDTH    (10)
#define NUM_WORKGROUPS_0 (IMAGE_WIDTH)
#define NUM_WORKGROUPS_1 (IMAGE_HEIGHT)
#define WORKGROUP_SIZE_0 (1)
#define WORKGROUP_SIZE_1 (1)
////////////////////////////////////////////////////////////////////////////////

int
load_file_to_memory(const char *filename, char **result)
{
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        {
            *result = NULL;
            return -1; // -1 means file opening fail
        }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f))
        {
            free(*result);
            return -2; // -2 means file reading fail
        }
    fclose(f);
    (*result)[size] = 0;
    return size;
}
void print_matrix(int *mat, int n_rows, int n_cols) {
    for(int r = 0; r < n_rows; r++) {
        for(int c = 0; c < n_cols; c++) {
            printf("%d, ", mat[r*n_cols+c]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    int test_fail = 0;

    int filter[FILTER_SIZE*FILTER_SIZE] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    int h_input[IMAGE_HEIGHT*IMAGE_WIDTH];      // input image buffer
    int hw_output[IMAGE_HEIGHT*IMAGE_WIDTH];    // host buffer for device output
    int sw_output[IMAGE_HEIGHT*IMAGE_WIDTH];    // host buffer for reference output

    size_t global[2];                   // global domain size for our calculation
    size_t local[2];                    // local domain size for our calculation

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem d_in_image;                  // device buffer for input image
    cl_mem d_in_filter;                 // device buffer for filter kernel
    cl_mem d_out_image;                 // device buffer for filtered image


    if (argc != 2) {
        printf("Usage: %s conv_2d.xclbin\n", argv[0]);
        return EXIT_FAILURE;
    }

    int row, col;
    // initialize the image buffer to some known pattern
    for(row = 0; row < IMAGE_HEIGHT; row++) {
        for(col = 0; col < IMAGE_WIDTH; col++) {
            h_input[row*IMAGE_WIDTH+col] = row;
        }
    }
    // Connect to first platform
    //
    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("INFO: CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("INFO: CL_PLATFORM_NAME %s\n",cl_platform_name);

    // Connect to a compute device
    //
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                         1, &device_id, NULL);
    if (err != CL_SUCCESS) {
            printf("Error: Failed to create a device group!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    int status;

    // Create Program Objects
    //

    // Load binary from disk
    unsigned char *kernelbinary;
    char *xclbin = argv[1];

    printf("INFO: loading xclbin %s\n", xclbin);
    int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin0: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    size_t n = n_i;

    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                        (const unsigned char **) &kernelbinary, &status, &err);

    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program0 from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "conv_2d", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the input and output arrays in device memory for our calculation
    //
    d_in_image = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    d_in_filter = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * FILTER_SIZE * FILTER_SIZE, NULL, NULL);
    d_out_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    if (!d_in_image || !d_in_filter || !d_out_image) {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Write the image from host buffer to device memory
    //
    err = clEnqueueWriteBuffer(commands, d_in_image, CL_TRUE, 0, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, h_input, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to image to device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    // Write filter kernel into device buffer
    //
    err = clEnqueueWriteBuffer(commands, d_in_filter, CL_TRUE, 0, sizeof(int) * FILTER_SIZE * FILTER_SIZE, filter, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to filter coeff into device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Set the arguments to our compute kernel
    //
    int filter_size = FILTER_SIZE;
    int bias = 1;
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_in_filter);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out_image);
    //err |= clSetKernelArg(kernel, 3, sizeof(int),    &filter_size);
    err |= clSetKernelArg(kernel, 3, sizeof(int),    &bias);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Launch computation kernel
    global[0] = NUM_WORKGROUPS_0 * WORKGROUP_SIZE_0;
    global[1] = NUM_WORKGROUPS_1 * WORKGROUP_SIZE_1;
    local[0] = WORKGROUP_SIZE_0;
    local[1] = WORKGROUP_SIZE_1;

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                 (size_t*)&global, (size_t*)&local, 0, NULL, NULL);
    if (err) {
            printf("Error: Failed to execute kernel! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Read back the results from the device to verify the output
    //
    cl_event readevent;
    err = clEnqueueReadBuffer( commands, d_out_image, CL_TRUE, 0, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, hw_output, 0, NULL, &readevent
);
    if (err != CL_SUCCESS) {
            printf("Error: Failed to read output array! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    clWaitForEvents(1, &readevent);

    // Generate reference output
    int kr, kc;
    int sum = 0;
    for(row = 0; row < IMAGE_HEIGHT-FILTER_SIZE+1; row++) {
        for(col = 0; col < IMAGE_WIDTH-FILTER_SIZE+1; col++) {
            sum = 0;
            for(kr = 0; kr < FILTER_SIZE; kr++) {
                for(kc = 0; kc < FILTER_SIZE; kc++ ) {
                    sum += (filter[kr*FILTER_SIZE + kc] * h_input[(row+kr)*IMAGE_WIDTH + col + kc]);
                }
            }
            sw_output[row*IMAGE_WIDTH + col] = sum + bias;
        }
    }
    // Check Results
    for(row = 0; row < IMAGE_HEIGHT-FILTER_SIZE+1; row++) {
        for(col = 0; col < IMAGE_WIDTH-FILTER_SIZE+1; col++) {
             if(sw_output[row*IMAGE_WIDTH+col] != hw_output[row*IMAGE_WIDTH+col]){
                 printf("Mismatch at : row = %d, col = %d, expected = %d, got = %d\n",
                     row, col, sw_output[row*IMAGE_WIDTH+col], hw_output[row*IMAGE_WIDTH+col]);
                 test_fail = 1;
             }
        }
    }
    printf("---------Input image-----------\n");
    print_matrix(h_input, IMAGE_HEIGHT, IMAGE_WIDTH);
    printf("---------Reference output------\n");
    print_matrix(sw_output, IMAGE_HEIGHT, IMAGE_WIDTH);
    printf("---------OCL Kernel output-----\n");
    print_matrix(hw_output, IMAGE_HEIGHT, IMAGE_WIDTH);
    //--------------------------------------------------------------------------
    // Shutdown and cleanup
    //--------------------------------------------------------------------------
    clReleaseMemObject(d_in_image);
    clReleaseMemObject(d_in_filter);
    clReleaseMemObject(d_out_image);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    if (test_fail) {
        printf("INFO: Test failed\n");
        return EXIT_FAILURE;
    } else {
        printf("INFO: Test passed\n");
    }
}
                                                    
