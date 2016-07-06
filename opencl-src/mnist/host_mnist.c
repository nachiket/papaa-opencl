#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "papi.h"
#include "lenet5_model.h"
#include "pgm.h"

#define POOL_SIZE 2
#define STRIDE    2

typedef float DTYPE;

unsigned int ipgm_img_width, ipgm_img_height;
unsigned int size_image;
unsigned int mem_size_image;
unsigned int size_filter1;
unsigned int mem_size_filter1;
unsigned int size_conv1;
unsigned int mem_size_conv1;
unsigned int size_pool1;
unsigned int mem_size_pool1;
unsigned int size_filter2;
unsigned int mem_size_filter2;
unsigned int size_conv2;
unsigned int mem_size_conv2;
unsigned int size_pool2;
unsigned int mem_size_pool2;

unsigned int nMisPredict = 0;
unsigned int noTestImgs = 0;

typedef struct conv {
unsigned int iW;
unsigned int iH;
unsigned int oW;
unsigned int oH;
unsigned int nFilW;
unsigned int nFilH;
unsigned int nInputs;
unsigned int nOutputs;
DTYPE * pFilter;
DTYPE * pBias;
DTYPE * pInput;
DTYPE * pOutput;
}conv_layer;

typedef struct pool {
unsigned int nPoolSize;
unsigned int nStride;
unsigned int iW;
unsigned int iH;
unsigned int oW;
unsigned int oH;
unsigned int nMaps;
DTYPE * pInput;
DTYPE * pOutput;
}pool_layer;

typedef struct ip {
unsigned int nInputs;
unsigned int nOutputs;
DTYPE* pInput;
DTYPE* pOutput;
DTYPE* pWeights;
DTYPE* pBias;
}ip_layer;

typedef struct lenet5 {
conv_layer conv1;
pool_layer pool1;
conv_layer conv2;
pool_layer pool2;
ip_layer   ip1;
ip_layer   ip2;
}lenet5;

cl_context context;                 // compute context
cl_device_id device_id;             // compute device id 
cl_mem d_image, d_filter1, d_conv1, d_bias1, d_pool1, d_filter2, d_conv2, d_bias2, d_pool2, d_weights1, d_output1, d_cbias1, d_cbias2, d_weights2, d_output;                       //OpenCL device memory for matrices
cl_kernel kernel[5];                // compute kernel
cl_program program;                 // compute program
cl_command_queue commands;          // compute command queue

extern long LoadOpenCLKernel(char const* path, char **buf);

int initApp(lenet5* plenet5, unsigned int  width, unsigned int height)
{
	plenet5->conv1.iW 		= width;
	plenet5->conv1.iH       = height;
	plenet5->conv1.nInputs  = CONV1_NO_INPUTS;
	plenet5->conv1.nOutputs = CONV1_NO_OUTPUTS;
	plenet5->conv1.nFilW	= CONV1_FILTER_WIDTH;
	plenet5->conv1.nFilH	= CONV1_FILTER_HEIGHT;
	plenet5->conv1.pFilter  = (DTYPE*) conv1_weights;
	plenet5->conv1.pBias 	= (DTYPE*) conv1_bias;
	plenet5->conv1.oW		= width-CONV1_FILTER_WIDTH+1;
	plenet5->conv1.oH		= height-CONV1_FILTER_HEIGHT+1;

	plenet5->pool1.nPoolSize = POOL_SIZE;
	plenet5->pool1.nStride   = STRIDE;
	plenet5->pool1.iW 		 = plenet5->conv1.oW;
	plenet5->pool1.iH 		 = plenet5->conv1.oH;
	plenet5->pool1.oW		 = ((plenet5->pool1.iW-plenet5->pool1.nPoolSize)/plenet5->pool1.nStride) +1;
	plenet5->pool1.oH		 = ((plenet5->pool1.iH-plenet5->pool1.nPoolSize)/plenet5->pool1.nStride) +1;
	plenet5->pool1.nMaps     = plenet5->conv1.nOutputs;

	plenet5->conv2.iW 		= plenet5->pool1.oW;
	plenet5->conv2.iH       = plenet5->pool1.oH;
	plenet5->conv2.nInputs  = CONV2_NO_INPUTS;
	plenet5->conv2.nOutputs = CONV2_NO_OUTPUTS;
	plenet5->conv2.nFilW	= CONV2_FILTER_WIDTH;
	plenet5->conv2.nFilH	= CONV2_FILTER_HEIGHT;
	plenet5->conv2.pFilter  = (DTYPE*) conv2_weights;
	plenet5->conv2.pBias 	= (DTYPE*) conv2_bias;
	plenet5->conv2.oW		= plenet5->conv2.iW-CONV2_FILTER_WIDTH+1;
	plenet5->conv2.oH		= plenet5->conv2.iH-CONV2_FILTER_HEIGHT+1;

	plenet5->pool2.nPoolSize = POOL_SIZE;
	plenet5->pool2.nStride   = STRIDE;
	plenet5->pool2.iW 		 = plenet5->conv2.oW;
	plenet5->pool2.iH 		 = plenet5->conv2.oH;
	plenet5->pool2.oW		 = ((plenet5->pool2.iW-plenet5->pool2.nPoolSize)/plenet5->pool2.nStride) +1;
	plenet5->pool2.oH		 = ((plenet5->pool2.iH-plenet5->pool2.nPoolSize)/plenet5->pool2.nStride) +1;
	plenet5->pool2.nMaps     = plenet5->conv2.nOutputs;

	plenet5->ip1.nInputs     = IP1_NO_INPUTS;
	plenet5->ip1.nOutputs 	 = IP1_NO_OUTPUTS;
	plenet5->ip1.pWeights	 = (DTYPE*) ip1_weights;
	plenet5->ip1.pBias 		 = (DTYPE*) ip1_bias;

	plenet5->ip2.nInputs     = IP2_NO_INPUTS;
	plenet5->ip2.nOutputs 	 = IP2_NO_OUTPUTS;
	plenet5->ip2.pWeights	 = (DTYPE*) ip2_weights;
	plenet5->ip2.pBias 		 = (DTYPE*) ip2_bias;

}

int alloc_host_memory(lenet5* plenet5)
{
	size_image = plenet5->conv1.iW * plenet5->conv1.iH;
	mem_size_image = sizeof(DTYPE) * size_image;
	plenet5->conv1.pInput  = (DTYPE*)malloc(mem_size_image * plenet5->conv1.nInputs);
	
	size_filter1 = plenet5->conv1.nFilW*plenet5->conv1.nFilH;
	mem_size_filter1 = sizeof(DTYPE) * size_filter1;

	size_conv1 = plenet5->conv1.oW * plenet5->conv1.oH;
	mem_size_conv1 = sizeof(DTYPE) * size_conv1;
	plenet5->conv1.pOutput = (DTYPE*) malloc(mem_size_conv1 * plenet5->conv1.nOutputs);
	
	size_pool1 = plenet5->pool1.oW * plenet5->pool1.oH;
	mem_size_pool1 = sizeof(DTYPE) * size_pool1;
	plenet5->pool1.pOutput = (DTYPE*) malloc(mem_size_pool1 * plenet5->pool1.nMaps);
	
	size_filter2 = plenet5->conv2.nFilW * plenet5->conv2.nFilH;
	mem_size_filter2 = sizeof(DTYPE) * size_filter2;
   
	size_conv2 = plenet5->conv2.oW * plenet5->conv2.oH;
	mem_size_conv2 = sizeof(DTYPE) * size_conv2;
	plenet5->conv2.pOutput = (DTYPE*) malloc(mem_size_conv2 * plenet5->conv2.nOutputs);

	size_pool2 = plenet5->pool2.oW * plenet5->pool2.oH;
	mem_size_pool2 = sizeof(DTYPE) * size_pool2;
	plenet5->pool2.pOutput = (DTYPE*) malloc(mem_size_pool2 * plenet5->pool2.nMaps);

	plenet5->ip1.pOutput = (DTYPE*) malloc(sizeof(DTYPE)* plenet5->ip1.nOutputs);

	plenet5->ip2.pOutput = (DTYPE*) malloc(sizeof(DTYPE)* plenet5->ip2.nOutputs);
	
	if(!plenet5->conv1.pInput || !plenet5->conv1.pOutput || !plenet5->pool1.pOutput || !plenet5->conv2.pOutput || !plenet5->pool2.pOutput || !plenet5->ip1.pOutput|| !plenet5->ip2.pOutput )
	{
	    printf("Error: Failed to allocate host memory \n");
	    exit(1);
	}
	
	return 0;
}

int free_host_mem (lenet5 *plenet5)
{
	free(plenet5->conv1.pInput);
	free(plenet5->conv1.pOutput);
	free(plenet5->pool1.pOutput);
	free(plenet5->conv2.pOutput);
	free(plenet5->pool2.pOutput);
	free(plenet5->ip1.pOutput);
	free(plenet5->ip2.pOutput);

	return 0;
}

int InitDevice()
{
	int i,err;
	cl_uint dev_cnt = 0;
	cl_platform_id platform_ids[5];

	clGetPlatformIDs(0, 0, &dev_cnt);
	
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	for(i=0;i<dev_cnt;i++)
	{
#ifdef DEVICE_GPU
	   err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#elif DEVICE_ACC
	   err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
#else
	   err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
#endif
	   if(err == CL_SUCCESS)
	   {
			char name[100],vendor[100],version[100];
			clGetPlatformInfo(platform_ids[i],CL_PLATFORM_NAME,sizeof(name),&name[0],NULL);
			clGetPlatformInfo(platform_ids[i],CL_PLATFORM_VENDOR,sizeof(vendor),&vendor[0],NULL);
			clGetPlatformInfo(platform_ids[i],CL_PLATFORM_VERSION,sizeof(version),&version[0],NULL);
			//printf("Using Platform %s from vendor %s \n Opencl Version Implemented is %s \n",name,vendor,version);
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

	return 0;
}

int BuildDeviceKernel ()
{
	int err; 
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

	kernel[0] = clCreateKernel(program, "filter3D", &err);
	if (!kernel[0] || err != CL_SUCCESS)
	{
	printf("Error: Failed to create compute kernel!\n");
	exit(1);
	}

	kernel[1] = clCreateKernel(program, "maxpool3D", &err);
	if (!kernel[1] || err != CL_SUCCESS)
	{
	printf("Error: Failed to create compute kernel!\n");
	exit(1);
	}

	kernel[2] = clCreateKernel(program, "iplayer", &err);
	if (!kernel[2] || err != CL_SUCCESS)
	{
	printf("Error: Failed to create compute kernel!\n");
	exit(1);
	}

	kernel[3] = clCreateKernel(program, "relu_layer", &err);
	if (!kernel[3] || err != CL_SUCCESS)
	{
	printf("Error: Failed to create compute kernel!\n");
	exit(1);
	}

	kernel[4] = clCreateKernel(program, "softmax", &err);
	if (!kernel[4] || err != CL_SUCCESS)
	{
	printf("Error: Failed to create compute kernel!\n");
	exit(1);
	}
	return 0;
}

int alloc_dev_memory(lenet5 * plenet5)
{
	int err;
	d_image    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_image*plenet5->conv1.nInputs, plenet5->conv1.pInput, &err);
	d_filter1  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_filter1*plenet5->conv1.nInputs*plenet5->conv1.nOutputs, plenet5->conv1.pFilter, &err);
	d_conv1    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_conv1*plenet5->conv1.nOutputs, NULL, &err);
	d_bias1    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(DTYPE)*plenet5->conv1.nOutputs, plenet5->conv1.pBias, &err);
	d_pool1    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_pool1*plenet5->pool1.nMaps, NULL, &err);
	d_filter2  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_filter2*plenet5->conv2.nInputs*plenet5->conv2.nOutputs, plenet5->conv2.pFilter, &err);
	d_conv2    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_conv2*plenet5->conv2.nOutputs, NULL, &err);
	d_bias2    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(DTYPE)*plenet5->conv2.nOutputs,plenet5->conv2.pBias, &err);
	d_pool2    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_pool2*plenet5->pool2.nMaps, NULL, &err);
	d_weights1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(DTYPE)*plenet5->ip1.nInputs*plenet5->ip1.nOutputs, plenet5->ip1.pWeights, &err);
	d_output1  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DTYPE)*plenet5->ip1.nOutputs, NULL, &err);
	d_cbias1   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(DTYPE)*plenet5->ip1.nOutputs, plenet5->ip1.pBias, &err);
	d_weights2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(DTYPE)*plenet5->ip2.nInputs*plenet5->ip2.nOutputs, plenet5->ip2.pWeights, &err);
	d_output   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DTYPE)*plenet5->ip2.nOutputs, NULL, &err);
	d_cbias2   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(DTYPE)*plenet5->ip2.nOutputs, plenet5->ip2.pBias, &err);

	if (!d_image || !d_filter1 || !d_conv1 || !d_bias1 || !d_pool1 || !d_filter2 || !d_conv2 || !d_bias2 || !d_pool2 || !d_weights1 || !d_output1 || !d_cbias1 || !d_weights2 || !d_output|| !d_cbias2 )
	{
	  printf("Error: Failed to allocate device memory!\n");
	  exit(1);
	}
	return 0;
}

int FreeDeviceData()
{
	clReleaseMemObject(d_cbias1);
	clReleaseMemObject(d_cbias2);
	clReleaseMemObject(d_weights1);
	clReleaseMemObject(d_pool2);
	clReleaseMemObject(d_conv2);
	clReleaseMemObject(d_pool1);
	clReleaseMemObject(d_conv1);
	clReleaseMemObject(d_image);
	clReleaseMemObject(d_filter1);
	clReleaseMemObject(d_bias1);
	clReleaseMemObject(d_output1);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_bias2);
	clReleaseMemObject(d_weights2);

	return 0;
}

int FreeDeviceProg()
{
	clReleaseProgram(program);
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);
	clReleaseKernel(kernel[3]);
	clReleaseKernel(kernel[4]);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}

int lenet5App(lenet5 *plenet5, int label)
{

	cl_event event[9];
	int err=0,i=0;
	int idx=-1;
	float result = -1.0;
	register long long int ptimer1=0;
	register long long int ptimer2=0;

	//Launch OpenCL kernel
	size_t global[3];

	ptimer1 = PAPI_get_virt_usec();

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_image);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter1);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_conv1);
	err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&plenet5->conv1.nFilW);
	err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&plenet5->conv1.nFilH);
	err |= clSetKernelArg(kernel[0], 5, sizeof(int), (void *)&plenet5->conv1.nInputs);
	err |= clSetKernelArg(kernel[0], 6, sizeof(cl_mem), (void*)&d_bias1);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);	
		exit(1);
	}


	global[0] = plenet5->conv1.oW;
	global[1] = plenet5->conv1.oH;
	global[2] = plenet5->conv1.nOutputs;

	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[0], 3, NULL, global, NULL, 0, NULL, &event[0]);
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_WORK_ITEM_SIZE)
			printf("CL_INVALID_WORK_ITEM_SIZE \n");
		if(err == CL_INVALID_WORK_GROUP_SIZE)
			printf("CL_INVALID_WORK_GROUP_SIZE \n");
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&d_conv1);
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&d_pool1);
	err |= clSetKernelArg(kernel[1], 2, sizeof(int), (void *)&plenet5->conv1.oW);
	err |= clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&plenet5->conv1.oH);
	err |= clSetKernelArg(kernel[1], 4, sizeof(int), (void *)&plenet5->pool1.nPoolSize);
	err |= clSetKernelArg(kernel[1], 5, sizeof(int), (void *)&plenet5->pool1.nStride);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	global[0] = plenet5->pool1.oW;
	global[1] = plenet5->pool1.oH;
	global[2] = plenet5->pool1.nMaps;


	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[1], 3, NULL, global, NULL, 1, &event[0], &event[1]);
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_WORK_ITEM_SIZE)
			printf("CL_INVALID_WORK_ITEM_SIZE \n");
		if(err == CL_INVALID_WORK_GROUP_SIZE)
			printf("CL_INVALID_WORK_GROUP_SIZE \n");
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_pool1);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter2);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_conv2);
	err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&plenet5->conv2.nFilW);
	err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&plenet5->conv2.nFilH);
	err |= clSetKernelArg(kernel[0], 5, sizeof(int), (void *)&plenet5->conv2.nInputs);
	err |= clSetKernelArg(kernel[0], 6, sizeof(cl_mem), (void*)&d_bias2);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	global[0] = plenet5->conv2.oW;
	global[1] = plenet5->conv2.oH;
	global[2] = plenet5->conv2.nOutputs;

	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[0], 3, NULL, global, NULL, 1, &event[1], &event[2]);
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_WORK_ITEM_SIZE)
			printf("CL_INVALID_WORK_ITEM_SIZE \n");
		if(err == CL_INVALID_WORK_GROUP_SIZE)
			printf("CL_INVALID_WORK_GROUP_SIZE \n");
		printf("Error: Failed to execute kernel! %d \n", err);
		exit(1);
	}

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&d_conv2);
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&d_pool2);
	err |= clSetKernelArg(kernel[1], 2, sizeof(int), (void *)&plenet5->conv2.oW);
	err |= clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&plenet5->conv2.oH);
	err |= clSetKernelArg(kernel[1], 4, sizeof(int), (void *)&plenet5->pool2.nPoolSize);
	err |= clSetKernelArg(kernel[1], 5, sizeof(int), (void *)&plenet5->pool2.nStride);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	global[0] = plenet5->pool2.oW;
	global[1] = plenet5->pool2.oH;
	global[2] = plenet5->pool2.nMaps;


	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[1], 3, NULL, global, NULL, 1, &event[2], &event[3]);
	if (err != CL_SUCCESS)
	{
		if(err == CL_INVALID_WORK_ITEM_SIZE)
			printf("CL_INVALID_WORK_ITEM_SIZE \n");
		if(err == CL_INVALID_WORK_GROUP_SIZE)
			printf("CL_INVALID_WORK_GROUP_SIZE \n");
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_pool2);
	err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_weights1);
	err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *)&d_output1);
	err |= clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&plenet5->ip1.nInputs);
	err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void*)&d_cbias1);


	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	size_t ip_global = plenet5->ip1.nOutputs;

	/*Enqueue task for parallel execution*/
	err = clEnqueueNDRangeKernel(commands, kernel[2], 1, NULL, &ip_global, NULL, 1, &event[3], &event[4]);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d \n", err);
		exit(1);
	}

	err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), (void*)&d_output1);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err = clEnqueueNDRangeKernel(commands,kernel[3],1,NULL,&ip_global,NULL,1,&event[4], &event[5]);
	if(err!= CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel %d \n", err);
		exit(1);
	}

	err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_output1);
	err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_weights2);
	err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&plenet5->ip2.nInputs);
	err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void*)&d_cbias2);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d \n", err);
		exit(1);
	}

	ip_global = plenet5->ip2.nOutputs;

	err = clEnqueueNDRangeKernel(commands, kernel[2], 1, NULL, &ip_global, NULL, 1,&event[5], &event[6]);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d \n", err);
		exit(1);
	}

	err = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), (void*)&d_output);
	if(err!= CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel %d \n", err);
		exit(1);
	}

	size_t smax_local = plenet5->ip2.nOutputs;
	size_t smax_global= plenet5->ip2.nOutputs;

	err = clEnqueueNDRangeKernel(commands, kernel[4], 1, NULL, &smax_global, &smax_local, 1, &event[6], &event[7]);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d \n", err);
		exit(1);
	}
	if(label==0xFFFF)
	{
		ptimer2 = PAPI_get_virt_usec();
		printf("cl:main timing:PAPI clEnqueueNDRangeKernel %llu us\n",(ptimer2-ptimer1));
	}
	clFinish(commands);
	if(label == 0xFFFF)
	{
		cl_ulong time_start, time_end;
		double total_time;
		clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event[7], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf("cl:main timing:opencl clEnqueueNDRangeKernel %0.3f us\n", total_time / 1000.0);
	}

	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, sizeof(DTYPE)*plenet5->ip2.nOutputs, plenet5->ip2.pOutput, 1, &event[7], &event[8]);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
	clWaitForEvents(1,&event[8]);
	printf("%d Output Probabilities are \n",noTestImgs);
	for(i=0;i<plenet5->ip2.nOutputs;i++)
	{
		printf("%e ,",plenet5->ip2.pOutput[i]);
		if(plenet5->ip2.pOutput[i]>result)
		{
	   		result = plenet5->ip2.pOutput[i];
	   		idx = i;
		}
	}
	printf("\n");
	
	if(label == 0xFFFF)
	{
    		printf("The digit in the image is %d \n",idx);
	}
	else
	{
		if(idx != label)
			nMisPredict++;		
	}
	return 0;
}

void print_help(char **argv)
{
        printf("Usage : %s\n"
                "-m sample -i <image path>\n"
                "\tOR\t\n"
                "-m test -f <image list file> -d <image dir> -n <no images to test>\n",argv[0]);
}


int main(int argc, char **argv)
{
	int i=0,j =0;
	pgm_t input_pgm;
	lenet5 olenet5;
       
        char * mode = NULL;
        char * imgName = NULL;
        char * imgListFile = NULL;
        char * imgDir = NULL;
        if(argc == 1)
	{
                print_help(argv);
                return -1;
        }

        // parse arguments and decide the application mode.
        for(i = 1; i < argc; i++)
	{
                if(!strcmp(argv[i], "-m")) {
                        mode = argv[++i];
                } else if (!strcmp(argv[i], "-i")){
                        imgName = argv[++i];
                } else if(!strcmp(argv[i], "-f")) {
                        imgListFile = argv[++i];
                } else if(!strcmp(argv[i], "-d")) {
                        imgDir = argv[++i];
                } else if(!strcmp(argv[i], "-n")) {
                        noTestImgs = atoi(argv[++i]);
                }
        }

	if(!strcmp(mode, "sample"))
	{

		readPGM(&input_pgm,imgName);
		ipgm_img_width  = input_pgm.width;
		ipgm_img_height = input_pgm.height;
		printf("cl:main program:img_width %d\n", ipgm_img_width);
		printf("cl:main program:img_height %d\n", ipgm_img_height);

		// initialize the lenet5 app
		initApp(&olenet5,ipgm_img_width, ipgm_img_height);

		//Allocate host memory for matrices
		if(alloc_host_memory(&olenet5))
		{
		   printf("Error : unable to allocate host memory!!! \n");
		   exit(1);
		}

		//read input image to host buffer
		for(j=0;j<olenet5.conv1.nInputs;j++)
		{
		   for(i=0;i<size_image;i++)
		   {
				olenet5.conv1.pInput[(i+(j*size_image))] = (DTYPE) input_pgm.buf[i]/255;
		   }
		}

		if(InitDevice())
		{
			printf("Error: Device cannot be initialized \n");
			exit(1);
		}

		if(BuildDeviceKernel())
		{
			printf("Error: Kernel Compilation error \n");
			exit(1);
		}

		// Create the input and output arrays in device memory for our calculation
		if(alloc_dev_memory(&olenet5))
		{
			printf("Error: Failed to allocate device memory \n");
			exit(1);
		}
		
		if(lenet5App(&olenet5,0xFFFF))
		{
			printf("Error: Failed to execute APP \n");
			exit(1);
		}
		
		destroyPGM(&input_pgm);
		
		if (FreeDeviceData())
		{
			printf("Error: Device Data memory cannot be freed \n!!");
			exit(1);
		}

		if(FreeDeviceProg())
		{
			printf("Error: Device Prog memory cannot be freed \n!!");
			exit(1);
		}
	}
	else if(!strcmp(mode, "test"))
	{
        	printf("********MNIST Test Mode*********\n ");
		FILE* fp;
		char* line = NULL;
		size_t len =0;
		ssize_t read;
		int num = 0, label;

		// initialize the lenet5 app
		initApp(&olenet5,28,28);

		if(InitDevice())
		{
			printf("Error: Device cannot be initialized \n");
			exit(1);
		}
		if(BuildDeviceKernel())
		{
			printf("Error: Kernel Compilation error \n");
			exit(1);
		}

		fp = fopen(imgListFile, "r");
		if (fp == NULL)
		{
			printf("Error in Opening image list file \n");
			exit(1);
		}

		while ((read = getline(&line, &len, fp)) != -1)
		{
			char* tok, filename[100];
    			for (tok = strtok(line, ",");tok && *tok;tok = strtok(NULL, ",\n"))
			{
				if(num==0)
				imgName = tok;
				if(num==1)
				label = atoi(tok);
				num++;
			}
			num = 0;
			noTestImgs++;
			strcpy(filename,imgDir);
			strcat(filename,"/");
			strcat(filename,imgName);
			
			readPGM(&input_pgm,filename);
			//ipgm_img_width  = input_pgm.width;
			//ipgm_img_height = input_pgm.height;


			//Allocate host memory for matrices
			if(alloc_host_memory(&olenet5))
			{
			   printf("Error : unable to allocate host memory!!! \n");
			   exit(1);
			}
			
			unsigned char max=0,min=0;
			for(i=0;i<size_image;i++)
			{
				if(max<input_pgm.buf[i])
					max = input_pgm.buf[i];
				if(min>input_pgm.buf[i])
					min = input_pgm.buf[i];
			}	
			//read input image to host buffer
			for(j=0;j<olenet5.conv1.nInputs;j++)
			{
			   for(i=0;i<size_image;i++)
			   {
					olenet5.conv1.pInput[(i+(j*size_image))] = (DTYPE) (input_pgm.buf[i]-min)/(max-min);
			   }
			}

			// Create the input and output arrays in device memory for our calculation
			if(alloc_dev_memory(&olenet5))
			{
				printf("Error: Failed to allocate device memory \n");
				exit(1);
			}
			
			if(lenet5App(&olenet5,label))
			{
				printf("Error: Failed to execute APP \n");
				exit(1);
			}
			
			if (FreeDeviceData())
			{
				printf("Error: Device Data memory cannot be freed \n!!");
				exit(1);
			}
			
			if(free_host_mem (&olenet5))
			{
				printf("Host memory cannot be freed \n !!");
				exit(1);
			}
		}
		if(FreeDeviceProg())
		{
			printf("Error: Device Prog memory cannot be freed \n!!");
			exit(1);
		}
		printf("Total Number of Images tested %d\n",noTestImgs);
		printf("Percentage Error: %f \n Number of Mispredicted images %d \n",(float)(nMisPredict/noTestImgs)*100,nMisPredict);
		destroyPGM(&input_pgm);
		fclose(fp);
		if(line)
			free(line);
	}
	else
	{
                printf("Invalid application mode \n");
                return -1;
        }
	return 0;
}
