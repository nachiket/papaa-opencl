#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "pgm.h"

typedef float DTYPE;

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
	int err, j, i = 0;                            // error code returned from api calls
	pgm_t input_pgm,output_pgm;

	int ipgm_img_width;
	int ipgm_img_height;

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                // compute kernel

	// OpenCL device memory for matrices
	cl_mem d_image, d_filter, d_output;

	// Simple laplacian kernel
	DTYPE lap_filter[3*3] = {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0};
	DTYPE bias = 0.01;

	if (argc != 2) {
		printf("Usage: %s <image_name.pgm>\n", argv[0]);
		exit(1);
	}

	// Read the input image
	readPGM(&input_pgm, argv[1]);
	ipgm_img_width  = input_pgm.width;
	ipgm_img_height = input_pgm.height;

	printf("Host: Input image resolution:%dx%d\n", ipgm_img_width,ipgm_img_height);

	DTYPE  *h_image;
	DTYPE  *h_filter, *h_output;

	// Allocate host memory for matrices
	unsigned int size_image = ipgm_img_width*ipgm_img_height;
	unsigned int mem_size_image = sizeof(DTYPE) * size_image;
	h_image    = (DTYPE*)malloc(mem_size_image);

	// Convert range from [0, 255] to [0.0, 1.0]
	for(i=0;i<size_image;i++)
	{
		h_image[i] = (DTYPE) input_pgm.buf[i]/255.0;
	}

	unsigned int size_filter = 3*3;
	unsigned int mem_size_filter = sizeof(DTYPE) * size_filter;
	h_filter = (DTYPE*) lap_filter;

	unsigned int size_output = ipgm_img_width * ipgm_img_height;
	unsigned int mem_size_output = sizeof(DTYPE) * size_output;
	h_output = (DTYPE*) malloc(mem_size_output);


	// Platform and device query
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[5];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	for(i=0;i<dev_cnt;i++)
	{
#ifdef DEVICE_GPU
		printf("Target is GPU\n");
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#else
		printf("Target is CPU\n");
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
#endif
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
	lFileSize = LoadOpenCLKernel("simple.cl", &KernelSource);
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

	kernel = clCreateKernel(program, "convolve", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Allocate the device buffer for input image, kernel and transfer the data
	d_image  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_image, h_image, &err);

	// Create the input and output arrays in device memory for our calculation
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_filter, h_filter, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size_output, NULL, &err);

	if (!d_image || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
		
	size_t localWorkSize[2], globalWorkSize[2];
	int filter_size  = 3;

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

	globalWorkSize[0] = ipgm_img_width;
	globalWorkSize[1] = ipgm_img_height;

	/*Enqueue task for parallel execution*/
	printf("Launching the kernel...\n");
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	// Wait for the commands to finish
	clFinish(commands);

	// Retrieve result from device
	printf("Reading output buffer into host memory...\n");
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_output, h_output, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	output_pgm.width = ipgm_img_width;
	output_pgm.height = ipgm_img_height;

	// Remove garbage pixels in the border. If not, this will effect the subsequent normalization.!
	for(i = 0; i < output_pgm.height; i++) {
		for(j = 0; j < output_pgm.width; j++) {
			if(i > output_pgm.height-filter_size || j > output_pgm.width-filter_size)
				h_output[i*output_pgm.width+j] = 0.0;
		}
	}

	normalizeF2PGM(&output_pgm, h_output);
	/* Output image */
	writePGM(&output_pgm, "ocl_output.pgm");


	destroyPGM(&input_pgm);
	destroyPGM(&output_pgm);

	free(h_image);
	free(h_output);
	clReleaseMemObject(d_image);
	clReleaseMemObject(d_filter);
	clReleaseMemObject(d_output);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
