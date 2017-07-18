#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "pgm.h"

#define FILTER_SIZE  (3)
typedef float DTYPE;

int APPROX_EQ(DTYPE n1, DTYPE n2, float eps) {
	return abs(n1-n2) < eps ? 1: 0;
}

void print_output(DTYPE *out, int rows, int cols) {
	int r, c;
	for(r = 0; r < rows; r++) {
		for(c = 0; c < cols; c++) {
			printf("%f,", out[r*cols+c]);
		}
		printf("\n");
	}
}

int main(int argc, char** argv)
{
	cl_event event;
	int err, i = 0;                            // error code returned from api calls
	cl_ulong time_start, time_end;
	double total_time;

	pgm_t input_pgm, output_pgm;


	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                // compute kernel

	// OpenCL device memory for matrices
	cl_mem d_image, d_filter, d_output;

	// Simple laplacian kernel
	DTYPE lap_filter[FILTER_SIZE*FILTER_SIZE] = {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0};
	DTYPE bias = 0.01;

	if (argc != 2) {
		printf("Usage: %s <image_name.pgm>\n", argv[0]);
		exit(1);
	}

	// Read the input image
	readPGM(&input_pgm, argv[1]);

	printf("Host: Input image resolution:%dx%d\n", input_pgm.width, input_pgm.height);

	DTYPE  *h_image, *h_image_padded;
	DTYPE  *h_filter, *h_output, *ref_output;

	// Allocate host memory for images and outputs
	h_image    = (DTYPE*)malloc(sizeof(DTYPE)*input_pgm.width*input_pgm.height);
	ref_output = (DTYPE*)malloc(sizeof(DTYPE)*input_pgm.width*input_pgm.height);
	//setup padded input image
	const int PADDED_SIZE = sizeof(DTYPE)*(input_pgm.width+FILTER_SIZE-1)*(input_pgm.height+FILTER_SIZE-1);
	h_image_padded = (DTYPE*)malloc(PADDED_SIZE);
	memset((void*)h_image_padded, 0, PADDED_SIZE); //init padded image to 0s
	int offset = 0; //Used for padded image

	// Convert range from [0, 255] to [0.0, 1.0]
	for(i = 0; i < input_pgm.width * input_pgm.height; i++)
	{
		if(i%input_pgm.width == 0 && i>0){ //if end of image row
			offset += FILTER_SIZE-1; //bump padded image to next row
		}
		h_image[i] = (DTYPE) input_pgm.buf[i]/255.0;
		h_image_padded[i+offset] = h_image[i];
	}

	h_filter = (DTYPE*) lap_filter;
	h_output = (DTYPE*) malloc(sizeof(DTYPE)*input_pgm.width*input_pgm.height);


	// Platform and device query
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[5];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	for(i = 0;i < dev_cnt; i++)
	{
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
		if(err == CL_SUCCESS)
			break;
	}
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source file
	char *KernelSource;
	long lFileSize;
	lFileSize = LoadOpenCLKernel("conv_kernel.cl", &KernelSource);
	if( lFileSize < 0L ) {
		perror("File read failed");
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "conv_2d", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Allocate the device buffer for input image, kernel and transfer the data
	d_image  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PADDED_SIZE, h_image_padded, &err);

	// Create the input and output arrays in device memory for our calculation
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DTYPE)*FILTER_SIZE*FILTER_SIZE, h_filter, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DTYPE)*input_pgm.width*input_pgm.height, NULL, &err);

	if (!d_image || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
		
	size_t localWorkSize[2], globalWorkSize[2];
	int filter_size  = FILTER_SIZE;

	// Setup the kernel arguments
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_image);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &filter_size);
	err |= clSetKernelArg(kernel, 4, sizeof(DTYPE), &bias);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);	
		exit(1);
	}

	localWorkSize[0] = 4;
	localWorkSize[1] = 4;

	globalWorkSize[0] = input_pgm.width;
	globalWorkSize[1] = input_pgm.height;

	/*Enqueue task for parallel execution*/
	printf("Launching the kernel...\n");
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	// Wait for the commands to finish
	clWaitForEvents(1, &event);

	// Get the profiling info
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time  = (double)(time_end - time_start);

	// Retrieve result from device
	printf("Reading output buffer into host memory...\n");
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, sizeof(DTYPE)*input_pgm.width*input_pgm.height, h_output, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//-------------------------------------------------------------
	// Compare between host and device output
    // Generate reference output
    int kr, kc, row, col;
    DTYPE sum = 0;
    for(row = 0; row < input_pgm.height; row++) {
        for(col = 0; col < input_pgm.width; col++) {
            sum = 0;
            for(kr = 0; kr < FILTER_SIZE; kr++) {
                for(kc = 0; kc < FILTER_SIZE; kc++ ) {
                    sum += (lap_filter[kr*FILTER_SIZE + kc] * h_image_padded[(row+kr)*(input_pgm.width+FILTER_SIZE-1) + col + kc]);
                }
            }
            ref_output[row*input_pgm.width + col] = sum + bias;
        }
    }
    // Check Results
	int test_fail = 0;
    for(row = 0; row < input_pgm.height; row++) {
        for(col = 0; col < input_pgm.width; col++) {
             if(!APPROX_EQ(ref_output[row*input_pgm.width+col], h_output[row*input_pgm.width+col], 1e-14)) {
                 printf("Mismatch at : row = %d, col = %d, expected = %f, got = %f\n",
                     row, col, ref_output[row*input_pgm.width+col], h_output[row*input_pgm.width+col]);
                 test_fail = 1;
             }
        }
    }


	output_pgm.width = input_pgm.width;
	output_pgm.height = input_pgm.height;

	// Remove garbage pixels in the border. If not, this will effect the subsequent normalization.!
	for(row = 0; row < output_pgm.height; row++) {
		for(col = 0; col < output_pgm.width; col++) {
			if(row > output_pgm.height- FILTER_SIZE || col > output_pgm.width-FILTER_SIZE)
				h_output[row * output_pgm.width + col] = 0.0;
		}
	}

	normalizeF2PGM(&output_pgm, h_output);
	/* Output image */
	writePGM(&output_pgm, "ocl_output.pgm");

	if (test_fail) {
		printf("INFO: TEST FAILED !!!!\n");
	} else {
		printf("INFO: ****TEST PASSED****\n");
	}
	printf("Kernel runtime = %0.3f us\n", total_time / 1000.0);

	destroyPGM(&input_pgm);
	destroyPGM(&output_pgm);

	free(h_image);
	free(h_image_padded);
	free(h_output);
	free(ref_output);
	clReleaseMemObject(d_image);
	clReleaseMemObject(d_filter);
	clReleaseMemObject(d_output);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
