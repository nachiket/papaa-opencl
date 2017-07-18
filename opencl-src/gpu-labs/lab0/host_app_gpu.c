#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "pgm.h"

#define NUM_WORK_GROUPS (4)
#define FILTER_SIZE   (3)
typedef float DTYPE;


extern long LoadOpenCLKernel(char const* path, char **buf);

int APPROX_EQ(DTYPE n1, DTYPE n2, float eps) {
	return abs(n1-n2) < eps ? 1: 0;
}

int main(int argc, char** argv)
{
	cl_event event,event1,event2;
	int err, i;                            // error code returned from api calls
	pgm_t input_pgm,output_pgm;
	DTYPE lap_filter[FILTER_SIZE*FILTER_SIZE] = {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0};
	DTYPE bias = 0.01;
	DTYPE *ref_output;


	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                // compute kernel

	// OpenCL device memory for matrices
	cl_mem d_image, d_filter, d_output;

	if (argc != 2) {
		printf("Usage: %s <image_name.pgm>\n", argv[0]);
		exit(1);
	}

	readPGM(&input_pgm,argv[1]);

	printf("Input image resolution:%dx%d\n", input_pgm.width,input_pgm.height);

	DTYPE  *h_image, *h_image_padded;
	DTYPE  *h_filter, *h_bias, *h_output;

	// Allocate host memory for matrices
	unsigned int size_image = input_pgm.width*input_pgm.height;
	unsigned int mem_size_image = sizeof(DTYPE) * size_image;
	h_image    = (DTYPE*)malloc(mem_size_image);
	ref_output = (DTYPE*)malloc(mem_size_image);
	
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

	unsigned int size_filter = FILTER_SIZE*FILTER_SIZE;
	unsigned int mem_size_filter = sizeof(DTYPE) * size_filter;
	h_filter = (DTYPE*) lap_filter;

	unsigned int size_output = input_pgm.width * input_pgm.height;
	unsigned int mem_size_output = sizeof(DTYPE) * size_output;
	h_output = (DTYPE*) malloc(mem_size_output);

	unsigned int size_bias = 1; //1 bias value for 1 output map 
	unsigned int mem_size_bias = sizeof(DTYPE) * size_bias;

	h_bias = (DTYPE*) &bias;

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[5];

	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	for(i=0;i<dev_cnt;i++)
	{
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		if(err == CL_SUCCESS) {
			printf("GPU selected for acceleration!\n");
			break;
		}
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

	kernel = clCreateKernel(program, "conv_local", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	d_image  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PADDED_SIZE, h_image_padded, &err);

	cl_ulong time_start, time_end;
	double total_time;

	// Create the input and output arrays in device memory for our calculation
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_filter, h_filter, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size_output, NULL, &err);

	if (!d_image || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
		
	// Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];
	int filter_width  = FILTER_SIZE;
	int filter_height = FILTER_SIZE;

	localWorkSize[0] = input_pgm.width;
	localWorkSize[1] = input_pgm.height/NUM_WORK_GROUPS;

	globalWorkSize[0] = input_pgm.width;
	globalWorkSize[1] = input_pgm.height;

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_image);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&filter_width);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&filter_height);
	err |= clSetKernelArg(kernel, 5, sizeof(float), (void*)&bias);
	err |= clSetKernelArg(kernel, 6, sizeof(float)*localWorkSize[0]*(localWorkSize[1]+filter_height-1), (void*)NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);	
		exit(1);
	}

	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
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
	clFinish(commands);

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
    for(row = 0; row < input_pgm.height - FILTER_SIZE + 1; row++) {
        for(col = 0; col < input_pgm.width - FILTER_SIZE + 1; col++) {
             if(!APPROX_EQ(ref_output[row*input_pgm.width+col], h_output[row*input_pgm.width+col], 1e-14)) {
                 printf("Mismatch at : row = %d, col = %d, expected = %f, got = %f\n",
                     row, col, ref_output[row*input_pgm.width+col], h_output[row*input_pgm.width+col]);
                 test_fail = 1;
             }
        }
    }
	if (test_fail) {
		printf("INFO: TEST FAILED !!!!\n");
	} else {
		printf("INFO: ****TEST PASSED****\n");
	}

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	clReleaseMemObject(d_filter);
	clReleaseMemObject(d_output);

	char fileoutputname[15];
	output_pgm.width = input_pgm.width;
	output_pgm.height = input_pgm.height;
	normalizeF2PGM(&output_pgm, h_output);
	sprintf(fileoutputname, "output2d.pgm");	
	/* Output image */
	writePGM(&output_pgm,fileoutputname);

	printf("cl:main timing %0.3f us\n", total_time / 1000.0);

	destroyPGM(&input_pgm);
	destroyPGM(&output_pgm);
	free(ref_output);
	free(h_image);
	free(h_output);
	clReleaseMemObject(d_image);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
