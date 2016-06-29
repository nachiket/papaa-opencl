#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "papi.h"
#include "../../../caffe-ref/scripts/gen/lenet5_model.h"
#include "pgm.h"

typedef float DTYPE;

extern long LoadOpenCLKernel(char const* path, char **buf);

int main()
{
	   cl_event event,event1,event2,event3;
	   int j =0;
	   register long long ptimer1=0;
	   register long long ptimer2=0;
	   int err,i =0,index =0;                            // error code returned from api calls
           pgm_t input_pgm;

	   cl_device_id device_id;             // compute device id 
	   cl_context context;                 // compute context
	   cl_command_queue commands;          // compute command queue
	   cl_program program;                 // compute program
	   cl_kernel kernel[4];                // compute kernel
	   cl_uint max_compute_units;
	   cl_ulong max_const_size;
	   size_t max_wg_size, max_wg_dim[3];
	
	    // OpenCL device memory for matrices
	   cl_mem d_input, d_weights1, d_output1, d_bias1, d_bias2, d_weights2, d_output2;
	   DTYPE  *h_input, *h_weights1, *h_bias1, *h_output1, *h_bias2, *h_weights2, *h_output2;

 	  //Allocate host memory for matrices
	   unsigned int size_input = IP1_NO_INPUTS;
	   unsigned int mem_size_input = sizeof(DTYPE) * size_input;
           h_input  = (DTYPE*)malloc(mem_size_input);
	   
	   char filenameinput[50];
	   for(j=0;j<CONV2_NO_OUTPUTS;j++)
	   {
	       sprintf(filenameinput, "../pool2/output3d%d.pgm",j);	
	       readPGM(&input_pgm,filenameinput);
	       for(i=0;i<input_pgm.width*input_pgm.height;i++)
	       {
	        	h_input[(j*input_pgm.width*input_pgm.height)+i] = (DTYPE) input_pgm.buf[i]/255;
	       }
	   }
	   
	   unsigned int size_weights1 = IP1_NO_INPUTS*IP1_NO_OUTPUTS;
	   unsigned int mem_size_weights1 = sizeof(DTYPE) * size_weights1;
	   h_weights1 = (DTYPE*) ip1_weights;
	   
	   unsigned int size_output1 = IP1_NO_OUTPUTS;
	   unsigned int mem_size_output1 = sizeof(DTYPE) * size_output1;
	   h_output1 = (DTYPE*) malloc(mem_size_output1);
	 
	   unsigned int size_bias1 = IP1_NO_OUTPUTS; 
	   unsigned int mem_size_bias1 = sizeof(DTYPE) * size_bias1;
	   h_bias1 = (DTYPE*) ip1_bias;

	   unsigned int size_weights2 = IP2_NO_INPUTS*IP2_NO_OUTPUTS;
	   unsigned int mem_size_weights2 = sizeof(DTYPE) * size_weights2;
	   h_weights2 = (DTYPE*) ip2_weights;
	   
	   unsigned int size_output2 = IP2_NO_OUTPUTS;
	   unsigned int mem_size_output2 = sizeof(DTYPE) * size_output2;
	   h_output2 = (DTYPE*) malloc(mem_size_output2);
	 
	   unsigned int size_bias2 = IP2_NO_OUTPUTS; 
	   unsigned int mem_size_bias2 = sizeof(DTYPE) * size_bias2;
	   h_bias2 = (DTYPE*) ip2_bias;

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
  	   
	   err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(max_compute_units),&max_compute_units,NULL);
	   if(err != CL_SUCCESS)
	   {
	     printf("Error: Failed to get device info! \n");
	     return EXIT_FAILURE;
	   }
	   //else
	    // printf("Max Compute Units :%d \n",max_compute_units);

	   err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_wg_size),&max_wg_size,NULL);
	   if(err != CL_SUCCESS)
	   {
	     printf("Error: Failed to get device info! \n");
	     return EXIT_FAILURE;
	   }
	   //else
	     //printf("Max WG size :%d \n",(int)max_wg_size);

	   err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(max_const_size),&max_const_size,NULL);
	   if(err != CL_SUCCESS)
	   {
	     printf("Error: Failed to get device info! \n");
	     return EXIT_FAILURE;
	   }
	   //else
	     //printf("Max Const buff size :%lld \n",(long)max_const_size);

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
	   //err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	   if (err != CL_SUCCESS)
	   {
	       size_t len;
	       char buffer[2048];
	       printf("Error: Failed to build program executable!\n");
	       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	       printf("%s\n", buffer);
	       exit(1);
	   }
	
	   kernel[0] = clCreateKernel(program, "iplayer", &err);
	   if (!kernel[0] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel1!\n");
	       exit(1);
	   }

	   kernel[1] = clCreateKernel(program, "relu_layer", &err);
	   if (!kernel[1] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel2!%d\n",err);
	       exit(1);
	   }

	   kernel[2] = clCreateKernel(program, "iplayer", &err);
	   if (!kernel[2] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel3!\n");
	       exit(1);
	   }

	   kernel[3] = clCreateKernel(program, "softmax", &err);
	   if (!kernel[3] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel4%d!\n",err);
	       exit(1);
	   }
 	   // Create the input and output arrays in device memory for our calculation
       	   d_input    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_input, h_input, &err);
       	   d_weights1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size_weights1, h_weights1, &err);
       	   d_output1  = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_output1, NULL, &err);
       	   d_bias1    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias1, h_bias1, &err); 
       	   d_weights2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_weights2, h_weights2, &err);
       	   d_output2  = clCreateBuffer(context, CL_MEM_READ_WRITE , mem_size_output2, NULL, &err);
       	   d_bias2    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , mem_size_bias2, h_bias2, &err);

       	   if (!d_input || !d_weights1 || !d_output1 || !d_bias1 || !d_weights2 || !d_output2 || !d_bias2)
       	   {
       	      printf("Error: Failed to allocate device memory! %d \n",err);
       	      exit(1);
       	   }    
       	  	//Launch OpenCL kernel
       	   size_t localWorkSize, globalWorkSize, global,local,g,l;
       	   int inputs1 = IP1_NO_INPUTS;
	   int inputs2 = IP2_NO_INPUTS;

       	   err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_input);
       	   err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_weights1);
       	   err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_output1);
       	   err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&inputs1);
       	   err |= clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void*)&d_bias1);
	
       	   err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&d_output1);

       	   err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_output1);
       	   err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_weights2);
       	   err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *)&d_output2);
       	   err |= clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&inputs2);
       	   err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void*)&d_bias2);
	
       	   err |= clSetKernelArg(kernel[3], 0, sizeof(cl_mem), (void*)&d_output2);

       	   if (err != CL_SUCCESS) {
       	        printf("Error: Failed to set kernel arguments! %d\n", err);	
       	        exit(1);
          	   }

       	   localWorkSize = 4; 
       	   globalWorkSize = IP1_NO_OUTPUTS;
       	
       	   ptimer1 = PAPI_get_virt_usec();
       	   /*Enqueue task for parallel execution*/
       	   err = clEnqueueNDRangeKernel(commands, kernel[0], 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
       	   if (err != CL_SUCCESS)
       	   {
		printf("Error: Failed to execute kernel! %d \n", err);
	        exit(1);
	   }
	
	   err = clEnqueueNDRangeKernel(commands,kernel[1],1,NULL,&globalWorkSize,&localWorkSize,1,&event, &event1);
	   if(err!= CL_SUCCESS)
	   {
		printf("Error: Failed to execute kernel %d \n", err);
		exit(1);
	   }

       	   local  = 2; 
       	   global = IP2_NO_OUTPUTS;
       	
       	   err = clEnqueueNDRangeKernel(commands, kernel[2], 1, NULL, &global, &local, 1,&event1, &event2);
       	   if (err != CL_SUCCESS)
       	   {
		printf("Error: Failed to execute kernel! %d \n", err);
	        exit(1);
	   }

	   l=IP2_NO_OUTPUTS;
	   g=IP2_NO_OUTPUTS;

       	   err = clEnqueueNDRangeKernel(commands, kernel[3], 1, NULL, &g, &l, 1, &event2, &event3);
       	   if (err != CL_SUCCESS)
       	   {
		printf("Error: Failed to execute kernel! %d \n", err);
	        exit(1);
	   }

	   ptimer2 = PAPI_get_virt_usec();
	   printf("cl:main timing:PAPI clEnqueueNDRangeKernel %llu us\n",(ptimer2-ptimer1));
	   clFinish(commands);
	   cl_ulong time_start, time_end;
           double total_time;
	   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	   clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	   total_time = time_end - time_start;
           printf("cl:main timing:opencl clEnqueueNDRangeKernel %0.3f us\n", total_time / 1000.0);

	   /*Retrieve result from device*/
           err = clEnqueueReadBuffer(commands, d_output2, CL_TRUE, 0, mem_size_output2, h_output2, 0, NULL, NULL);
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
		printf("%f,",h_output2[i]);
		if(h_output2[i]>result)
		{
		   result = h_output2[i];
		   idx = i;
		}
	   }
	   printf("\n");
	   
           
	   printf("The digit in the image is %d \n",idx);
	   destroyPGM(&input_pgm);
	   
	   free(h_input);
	   free(h_output1);
	   free(h_output2);

	   clReleaseMemObject(d_input);
           clReleaseMemObject(d_weights1);
	   clReleaseMemObject(d_output1);
           clReleaseMemObject(d_bias1);
           clReleaseMemObject(d_weights2);
	   clReleaseMemObject(d_output2);
           clReleaseMemObject(d_bias2);

	   clReleaseProgram(program);
	   clReleaseKernel(kernel[0]);
	   clReleaseKernel(kernel[1]);
	   clReleaseKernel(kernel[2]);
	   clReleaseCommandQueue(commands);
	   clReleaseContext(context);
	   
	   printf("clmain Program:  Complete\n");
	   return 0;
}

void printFilter(DTYPE* data, unsigned int width, unsigned int height, unsigned int mapnum)
{
    int m,k;
    DTYPE* temp = data+(mapnum*width*height);
    for(m=0;m<height;m++)
    {
       for(k=0;k<width;k++)
       {
           printf("%1.2f,",temp[(width*m)+k]);
       }
       printf("\n");
    }
    printf("\n\n");
}
