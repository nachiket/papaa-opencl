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

int main(int argc, char **argv) {

        if(argc < 2) {
                printf("Please specify the image path \n");
                exit(1);
        }

	cl_event event[8];
	int err, i=0,j =0, stride=STRIDE,poolsize=POOL_SIZE;
	register long long int ptimer1=0;
	register long long int ptimer2=0;
        pgm_t input_pgm;

        int ipgm_img_width,conv1_width,l1_width,conv2_width,l2_width;
	int ipgm_img_height,conv1_height,l1_height,conv2_height,l2_height;

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel[5];                // compute kernel
	
	//OpenCL device memory for matrices
	cl_mem d_image, d_filter1, d_conv1, d_bias1, d_pool1, d_filter2, d_conv2, d_bias2, d_pool2;
	DTYPE *h_image, *h_filter1, *h_conv1, *h_bias1, *h_pool1, *h_filter2, *h_conv2, *h_bias2, *h_pool2;
        cl_mem d_weights1, d_output1, d_cbias1, d_cbias2, d_weights2, d_output;
        DTYPE  *h_weights1, *h_cbias1, *h_output1, *h_cbias2, *h_weights2, *h_output;

	unsigned int size_bias = 1; //1 bias value for 1 output map 
	unsigned int mem_size_bias = sizeof(DTYPE) * size_bias;
        unsigned int size_weights = 1;
        unsigned int mem_size_weights = sizeof(DTYPE) * size_weights;
        unsigned int size_output = 1;
        unsigned int mem_size_output = sizeof(DTYPE) * size_output;

	readPGM(&input_pgm,argv[1]);
	ipgm_img_width  = input_pgm.width;
	ipgm_img_height = input_pgm.height;
	printf("cl:main program:img_width %d\n", ipgm_img_width);
	printf("cl:main program:img_height %d\n", ipgm_img_height);

 	//Allocate host memory for matrices
	unsigned int size_image = ipgm_img_width*ipgm_img_height;
	unsigned int mem_size_image = sizeof(DTYPE) * size_image;
        h_image  = (DTYPE*)malloc(mem_size_image * CONV1_NO_INPUTS);
	for(j=0;j<CONV1_NO_INPUTS;j++)
	{
	   for(i=0;i<size_image;i++)
	   {
	   	h_image[(i+(j*size_image))] = (DTYPE) input_pgm.buf[i]/255;
	   }
	}
	
	unsigned int size_filter1 = CONV1_FILTER_WIDTH*CONV1_FILTER_HEIGHT;
	unsigned int mem_size_filter1 = sizeof(DTYPE) * size_filter1;
	h_filter1 = (DTYPE*) conv1_weights;
	
	conv1_width  = ipgm_img_width-CONV1_FILTER_WIDTH+1; 
	conv1_height = ipgm_img_height-CONV1_FILTER_HEIGHT+1;

	unsigned int size_conv1 = conv1_width * conv1_height;
	unsigned int mem_size_conv1 = sizeof(DTYPE) * size_conv1;
	h_conv1 = (DTYPE*) malloc(mem_size_conv1*CONV1_NO_OUTPUTS);
	 
	h_bias1 = (DTYPE*) conv1_bias;
	
	l1_width  = ((conv1_width-poolsize)/stride) +1; 
	l1_height = ((conv1_height-poolsize)/stride) +1;

        unsigned int size_pool1 = l1_width * l1_height;
        unsigned int mem_size_pool1 = sizeof(DTYPE) * size_pool1;
        h_pool1 = (DTYPE*) malloc(mem_size_pool1*CONV1_NO_OUTPUTS);
	
	unsigned int size_filter2 = CONV2_FILTER_WIDTH*CONV2_FILTER_HEIGHT;
	unsigned int mem_size_filter2 = sizeof(DTYPE) * size_filter2;
	h_filter2 = (DTYPE*) conv2_weights;
	
	conv2_width = l1_width-CONV2_FILTER_WIDTH+1;
	conv2_height = l1_height-CONV2_FILTER_HEIGHT+1;
   
	unsigned int size_conv2 = conv2_width * conv2_height ;
	unsigned int mem_size_conv2 = sizeof(DTYPE) * size_conv2;
	h_conv2 = (DTYPE*) malloc(mem_size_conv2*CONV2_NO_OUTPUTS);
	 
	h_bias2 = (DTYPE*) conv2_bias;
	
	l2_width  = ((conv2_width-poolsize)/stride) +1; 
	l2_height = ((conv2_height-poolsize)/stride) +1;

        unsigned int size_pool2 = l2_width*l2_height;
        unsigned int mem_size_pool2 = sizeof(DTYPE) * size_pool2;
        h_pool2 = (DTYPE*) malloc(mem_size_pool2*CONV2_NO_OUTPUTS);

        h_weights1 = (DTYPE*) ip1_weights;

        h_cbias1 = (DTYPE*) ip1_bias;

        h_output1 = (DTYPE*) malloc(sizeof(DTYPE)*IP1_NO_OUTPUTS);

        h_cbias2 = (DTYPE*) ip2_bias;

        h_weights2 = (DTYPE*) ip2_weights;

        h_output = (DTYPE*) malloc(sizeof(DTYPE)*IP2_NO_OUTPUTS);
	
	if(!h_image || !h_conv1 || !h_pool1 || !h_conv2 || !h_pool2 || !h_output1 || !h_output || !h_filter1 || !h_filter2 || !h_bias1 || !h_bias2 || !h_weights1 || !h_cbias1 || !h_output1 || !h_weights2 || !h_cbias2 || !h_output )
	{
	    printf("Error: Failed to allocate host memory \n");
	    exit(1);
	}

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
	   {
		char name[20],vendor[20],version[20];
		clGetPlatformInfo(platforms_ids[i],CL_PLATFORM_NAME,sizeof(name);&name[0],NULL);
		clGetPlatformInfo(platforms_ids[i],CL_PLATFORM_VENDOR,sizeof(vendor);&vendor[0],NULL);
		clGetPlatformInfo(platforms_ids[i],CL_PLATFORM_VERSION,sizeof(version);&version[0],NULL);
		printf("Using Platform %s from vendor %s \n Opencl Version Implemented is %s \n",name,vendor,version);
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

 	   // Create the input and output arrays in device memory for our calculation
       	   d_image    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_image*CONV1_NO_INPUTS, h_image, &err);
       	   d_filter1  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_filter1*CONV1_NO_INPUTS*CONV1_NO_OUTPUTS, h_filter1, &err);
       	   d_conv1    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_conv1*CONV1_NO_OUTPUTS, NULL, &err);
       	   d_bias1    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias*CONV1_NO_OUTPUTS, h_bias1, &err);
	   d_pool1    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_pool1*CONV1_NO_OUTPUTS, NULL, &err);
       	   d_filter2  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_filter2*CONV2_NO_INPUTS*CONV2_NO_OUTPUTS, h_filter2, &err);
       	   d_conv2    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_conv2*CONV2_NO_OUTPUTS, NULL, &err);
       	   d_bias2    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias*CONV2_NO_OUTPUTS, h_bias2, &err);
           d_pool2    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_pool2*CONV2_NO_OUTPUTS, NULL, &err);
       	   d_weights1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_weights*IP1_NO_INPUTS*IP1_NO_OUTPUTS, h_weights1, &err);
           d_output1  = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_output*IP1_NO_OUTPUTS, NULL, &err);
           d_cbias1   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias*IP1_NO_OUTPUTS, h_cbias1, &err);
           d_weights2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_weights*IP2_NO_INPUTS*IP2_NO_OUTPUTS, h_weights2, &err);
           d_output   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size_output*IP2_NO_OUTPUTS, NULL, &err);
           d_cbias2   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias*IP2_NO_OUTPUTS, h_cbias2, &err);
 
	   if (!d_image || !d_filter1 || !d_conv1 || !d_bias1 || !d_pool1 || !d_filter2 || !d_conv2 || !d_bias2 || !d_pool2 || !d_weights1 || !d_output1 || !d_cbias1 || !d_weights2 || !d_output|| !d_cbias2 )
       	   {
       	      printf("Error: Failed to allocate device memory!\n");
       	      exit(1);
       	   }    

       	   //Launch OpenCL kernel
       	   size_t local[3],global[3];
       	   local[0] = 1;
       	   local[1] = 1;
       	   local[2] = 1;

       	   int filter_width1  = CONV1_FILTER_WIDTH;
       	   int filter_height1 = CONV1_FILTER_HEIGHT;
       	   int in_maps1       = CONV1_NO_INPUTS;
       
       	   err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_image);
       	   err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter1);
       	   err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_conv1);
       	   err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&filter_width1);
       	   err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&filter_height1);
       	   err |= clSetKernelArg(kernel[0], 5, sizeof(int), (void *)&in_maps1);
       	   err |= clSetKernelArg(kernel[0], 6, sizeof(cl_mem), (void*)&d_bias1);
       	
       	   if (err != CL_SUCCESS)
	   {
       	        printf("Error: Failed to set kernel arguments! %d\n", err);	
       	        exit(1);
           }

       
       	   global[0] = conv1_width;
       	   global[1] = conv1_height;
       	   global[2] = CONV1_NO_OUTPUTS;
       	   
	  /*Enqueue task for parallel execution*/
       	   err = clEnqueueNDRangeKernel(commands, kernel[0], 3, NULL, global, local, 0, NULL, &event[0]);
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
           err |= clSetKernelArg(kernel[1], 2, sizeof(int), (void *)&conv1_width);
           err |= clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&conv1_height);
           err |= clSetKernelArg(kernel[1], 4, sizeof(int), (void *)&poolsize);
           err |= clSetKernelArg(kernel[1], 5, sizeof(int), (void *)&stride);

           if (err != CL_SUCCESS) {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
                   }

           global[0] = l1_width;
           global[1] = l1_height;
           global[2] = CONV1_NO_OUTPUTS;


           /*Enqueue task for parallel execution*/
           err = clEnqueueNDRangeKernel(commands, kernel[1], 3, NULL, global, local, 1, &event[0], &event[1]);
           if (err != CL_SUCCESS)
           {
                if(err == CL_INVALID_WORK_ITEM_SIZE)
                printf("CL_INVALID_WORK_ITEM_SIZE \n");
                if(err == CL_INVALID_WORK_GROUP_SIZE)
                        printf("CL_INVALID_WORK_GROUP_SIZE \n");
                printf("Error: Failed to execute kernel! %d\n", err);
                exit(1);
	   }

           int filter_width2  = CONV2_FILTER_WIDTH;
           int filter_height2 = CONV2_FILTER_HEIGHT;
           int in_maps2       = CONV2_NO_INPUTS;

           err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_pool1);
           err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter2);
           err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_conv2);
           err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&filter_width2);
           err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&filter_height2);
           err |= clSetKernelArg(kernel[0], 5, sizeof(int), (void *)&in_maps2);
           err |= clSetKernelArg(kernel[0], 6, sizeof(cl_mem), (void*)&d_bias2);

           if (err != CL_SUCCESS) {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
                   }

           global[0] = conv2_width;
           global[1] = conv2_height;
           global[2] = CONV2_NO_OUTPUTS;

           /*Enqueue task for parallel execution*/
           err = clEnqueueNDRangeKernel(commands, kernel[0], 3, NULL, global, local, 1, &event[1], &event[2]);
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
           err |= clSetKernelArg(kernel[1], 2, sizeof(int), (void *)&conv2_width);
           err |= clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&conv2_height);
           err |= clSetKernelArg(kernel[1], 4, sizeof(int), (void *)&poolsize);
           err |= clSetKernelArg(kernel[1], 5, sizeof(int), (void *)&stride);

           if (err != CL_SUCCESS) {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
                   }

           global[0] = l2_width;
           global[1] = l2_height;
           global[2] = CONV2_NO_OUTPUTS;


           /*Enqueue task for parallel execution*/
           err = clEnqueueNDRangeKernel(commands, kernel[1], 3, NULL, global, local, 1, &event[2], &event[3]);
           if (err != CL_SUCCESS)
           {
                if(err == CL_INVALID_WORK_ITEM_SIZE)
                printf("CL_INVALID_WORK_ITEM_SIZE \n");
                if(err == CL_INVALID_WORK_GROUP_SIZE)
                        printf("CL_INVALID_WORK_GROUP_SIZE \n");
                printf("Error: Failed to execute kernel! %d\n", err);
                exit(1);
           }

           int inputs1 = IP1_NO_INPUTS;

           err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_pool2);
           err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_weights1);
           err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *)&d_output1);
           err |= clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&inputs1);
           err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void*)&d_cbias1);


           if (err != CL_SUCCESS) {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
                   }

          size_t ip_local  = 2;
          size_t ip_global = IP1_NO_OUTPUTS;

           ptimer1 = PAPI_get_virt_usec();
           /*Enqueue task for parallel execution*/
           err = clEnqueueNDRangeKernel(commands, kernel[2], 1, NULL, &ip_global, &ip_local, 1, &event[3], &event[4]);
           if (err != CL_SUCCESS)
           {
                printf("Error: Failed to execute kernel! %d \n", err);
                exit(1);
           }

           err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), (void*)&d_output1);
           if (err != CL_SUCCESS) {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
	   }

           err = clEnqueueNDRangeKernel(commands,kernel[3],1,NULL,&ip_global,&ip_local,1,&event[4], &event[5]);
           if(err!= CL_SUCCESS)
           {
                printf("Error: Failed to execute kernel %d \n", err);
                exit(1);
           }

           int inputs2 = IP2_NO_INPUTS;

           err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_output1);
           err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_weights2);
           err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *)&d_output);
           err |= clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&inputs2);
           err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void*)&d_cbias2);
           if (err != CL_SUCCESS)
           {
                printf("Error: Failed to execute kernel! %d \n", err);
                exit(1);
           }

           ip_global = IP2_NO_OUTPUTS;

           err = clEnqueueNDRangeKernel(commands, kernel[2], 1, NULL, &ip_global, &ip_local, 1,&event[5], &event[6]);
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

           size_t smax_local=IP2_NO_OUTPUTS;
           size_t smax_global=IP2_NO_OUTPUTS;

           err = clEnqueueNDRangeKernel(commands, kernel[4], 1, NULL, &smax_global, &smax_local, 1, &event[6], &event[7]);
           if (err != CL_SUCCESS)
           {
                printf("Error: Failed to execute kernel! %d \n", err);
                exit(1);
           }

          // ptimer2 = PAPI_get_virt_usec();
          // printf("cl:main timing:PAPI clEnqueueNDRangeKernel %llu us\n",(ptimer2-ptimer1));
           clFinish(commands);
           cl_ulong time_start, time_end;
           double total_time;
           clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
           clGetEventProfilingInfo(event[7], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
           total_time = time_end - time_start;
           printf("cl:main timing:opencl clEnqueueNDRangeKernel %0.3f us\n", total_time / 1000.0);

           err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, sizeof(DTYPE)*IP2_NO_OUTPUTS, h_output, 0, NULL, NULL);
           if (err != CL_SUCCESS)
           {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
           }

           int idx=-1;
           float result = -1.0;
           printf("Output Probabilities \n");
           for(i=0;i<IP2_NO_OUTPUTS;i++)
           {
                printf("%f,",h_output[i]);
                if(h_output[i]>result)
                {
                   result = h_output[i];
                   idx = i;
                }
           }
           printf("\n");


           printf("The digit in the image is %d \n",idx);
#if 0

           clFinish(commands);

           err = clEnqueueReadBuffer(commands, d_pool2, CL_TRUE, 0, mem_size_pool2*CONV2_NO_OUTPUTS, h_pool2, 0, NULL, NULL);
           /*Retrieve result from device*/
           if (err != CL_SUCCESS)
           {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
           }

	   pgm_t output_pgm;
           char fileoutputname[15];
           output_pgm.width  = l2_width;
           output_pgm.height = l2_height;
	   printf("output width %d \n", output_pgm.width);
	   printf("output height %d \n", output_pgm.height);

           for(i=0;i<CONV2_NO_OUTPUTS;i++)
           {
              normalizeF2PGM(&output_pgm,h_pool2+(i*output_pgm.width*output_pgm.height));
              sprintf(fileoutputname, "output3d%d.pgm",i);
              /* Output image */
              writePGM(&output_pgm,fileoutputname);
           }

           destroyPGM(&output_pgm);
#endif
           destroyPGM(&input_pgm);
	   
	   free(h_image);
	   free(h_conv1);
	   free(h_conv2);
	   free(h_pool1);
	   free(h_pool2);
	   free(h_output1);
	   free(h_output);

	
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
