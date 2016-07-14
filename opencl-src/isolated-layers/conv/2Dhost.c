#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "../../../caffe-ref/scripts/gen/lenet5_model.h"
#include "pgm.h"

typedef float DTYPE;

#define NUM_WORK_GROUPS 4

extern long LoadOpenCLKernel(char const* path, char **buf);
int main(int argc, char** argv)
{
	cl_event event,event1,event2;
	int j =0,stride=2;
	int err, i =0, index =0;                            // error code returned from api calls
	pgm_t input_pgm,output_pgm;

	int ipgm_img_width,opgm_img_width;
	int ipgm_img_height,opgm_img_height;

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel[3];                // compute kernel

	// OpenCL device memory for matrices
	cl_mem d_image, d_filter, d_output, d_bias;

	if (argc != 2) {
		printf("Expecting 2 arguments.\n");
		exit(1);
	}

	readPGM(&input_pgm,argv[1]);
	ipgm_img_width  = input_pgm.width;
	ipgm_img_height = input_pgm.height;
	opgm_img_width  = input_pgm.width;//-CONV1_FILTER_WIDTH+1;
	opgm_img_height = input_pgm.height;//-CONV1_FILTER_HEIGHT+1;

	printf("cl:main input image resolution:%dx%d\n", ipgm_img_width,ipgm_img_height);
	printf("cl:main output image resolution:%dx%d\n", opgm_img_width,opgm_img_height);

	DTYPE  *h_image;
	DTYPE  *h_filter, *h_bias, *h_output;

	// Allocate host memory for matrices
	unsigned int size_image = ipgm_img_width*ipgm_img_height;
	unsigned int mem_size_image = sizeof(DTYPE) * size_image;
	h_image    = (DTYPE*)malloc(mem_size_image);
	for(i=0;i<size_image;i++)
	{
		h_image[i] = (DTYPE) input_pgm.buf[i]/255;
	}

	unsigned int size_filter = CONV1_FILTER_WIDTH*CONV1_FILTER_HEIGHT;
	unsigned int mem_size_filter = sizeof(DTYPE) * size_filter;
	h_filter = (DTYPE*) conv1_weights;

	unsigned int size_output = opgm_img_width * opgm_img_height;
	unsigned int mem_size_output = sizeof(DTYPE) * size_output;
	h_output = (DTYPE*) malloc(mem_size_output);

	unsigned int size_bias = 1; //1 bias value for 1 output map 
	unsigned int mem_size_bias = sizeof(DTYPE) * size_bias;
	h_bias = (DTYPE*) conv1_bias;

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[5];

	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	for(i=0;i<dev_cnt;i++)
	{
#ifdef DEVICE_GPU
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#else
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
#endif
		if(err == CL_SUCCESS)
			break;
	}
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_PLATFORM)
			printf("CL_INVALID_PLATFORM\n");
		if(err == CL_INVALID_DEVICE_TYPE)
			printf("CL_INVALID_DEVICE_TYPE\n");
		if(err == CL_INVALID_VALUE)
			printf("CL_INVALID_VALUE\n");
		if(err == CL_DEVICE_NOT_FOUND)
			printf("CL_DEVICE_NOT_FOUND\n");
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
	lFileSize = LoadOpenCLKernel("kernels.cl", &KernelSource);
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

	kernel[0] = clCreateKernel(program, "conv_2d", &err);
	if (!kernel[0] || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	d_image  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR /*| CL_MEM_USE_MSMC_TI*/, mem_size_image, h_image, &err);

	cl_ulong time_start, time_end;
	double total_time;

	// Create the input and output arrays in device memory for our calculation
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR /*| CL_MEM_USE_MSMC_TI*/, mem_size_filter, h_filter, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY /*| CL_MEM_USE_MSMC_TI*/, mem_size_output, NULL, &err);
	d_bias   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias, h_bias, &err);

	if (!d_image || !d_filter || !d_output || !d_bias)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
		
	// Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];
	int filter_width  = CONV1_FILTER_WIDTH;
	int filter_height = CONV1_FILTER_HEIGHT;

	localWorkSize[0] = opgm_img_width;
	localWorkSize[1] = opgm_img_height/NUM_WORK_GROUPS;

	globalWorkSize[0] = opgm_img_width;
	globalWorkSize[1] = opgm_img_height;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_image);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&filter_width);
	err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&filter_height);
	err |= clSetKernelArg(kernel[0], 5, sizeof(cl_mem), (void*)&d_bias);
	err |= clSetKernelArg(kernel[0], 6, sizeof(float)*localWorkSize[0]*(localWorkSize[1]+filter_height-1), (void*)NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);	
		exit(1);
	}

	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[0], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_WORK_ITEM_SIZE)
			printf("CL_INVALID_WORK_ITEM_SIZE \n");
		if(err == CL_INVALID_WORK_GROUP_SIZE)
			printf("CL_INVALID_WORK_GROUP_SIZE \n");
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
	clWaitForEvents(1,&event);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time  = (double)(time_end - time_start);

	// Retrieve result from device
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_output, h_output, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	clReleaseMemObject(d_filter);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_bias);

	char fileoutputname[15];
	output_pgm.width = opgm_img_width;
	output_pgm.height = opgm_img_height;
	normalizeF2PGM(&output_pgm, h_output);
	sprintf(fileoutputname, "output2d.pgm");	
	/* Output image */
	writePGM(&output_pgm,fileoutputname);

	printf("cl:main timing %0.3f us\n", total_time / 1000.0);

	destroyPGM(&input_pgm);
	destroyPGM(&output_pgm);

	free(h_image);
	free(h_output);
	clReleaseMemObject(d_image);

	clReleaseProgram(program);
	clReleaseKernel(kernel[0]);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
