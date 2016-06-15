#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "papi.h"
#include "mnist.h"
#include "pgm.h"

#define DTYPE float

int main()
{
	   cl_event event,event1,event2;
	   int j =0,stride=2;
	   register long long ptimer1=0;
	   register long long ptimer2=0;
	   int err, i =0, index =0;                            // error code returned from api calls
           pgm_t input_pgm,output_pgm;

	   int ipgm_img_width = IMAGE_WIDTH;
	   int ipgm_img_height = IMAGE_HEIGHT;
	 
	   cl_device_id device_id;             // compute device id 
	   cl_context context;                 // compute context
	   cl_command_queue commands;          // compute command queue
	   cl_program program;                 // compute program
	   cl_kernel kernel[3];                // compute kernel
	
	    // OpenCL device memory for matrices
	   cl_mem d_image, d_filter, d_output, d_D;

	   //readPGM(input_pgm,"lena.pgm");
	
	   printf("cl:main program:img_width %d\n", ipgm_img_width);
	   printf("cl:main program:img_height %d\n", ipgm_img_height);
 	
 	  //Allocate host memory for matrices
	   unsigned int size_image = ipgm_img_width*ipgm_img_height;
	   unsigned int mem_size_image = sizeof(DTYPE) * size_image;
	   DTYPE* h_image = (DTYPE*) malloc(mem_size_image);
	 
	   unsigned int size_filter = L1_KERNEL_SIZE*L1_KERNEL_SIZE;
	   unsigned int mem_size_filter = sizeof(DTYPE) * size_filter;
	   DTYPE* h_filter = (DTYPE*) malloc(mem_size_filter);
	   
	   unsigned int size_output = ipgm_img_width * ipgm_img_height;
	   unsigned int mem_size_output = sizeof(DTYPE) * size_output;
	   DTYPE* h_output = (DTYPE*) malloc(mem_size_output);
	 
	   unsigned int size_D = (ipgm_img_width/stride) * (ipgm_img_height/stride);
	   unsigned int mem_size_D = sizeof(DTYPE) * size_D;
	   DTYPE* h_D = (DTYPE*) malloc(mem_size_D);

	   cl_uint dev_cnt = 0;
	   clGetPlatformIDs(0, 0, &dev_cnt);
		
	   cl_platform_id platform_ids[5];
	
	   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	   for(i=0;i<dev_cnt;i++)
	   {
	    err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
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
	   lFileSize = LoadOpenCLKernel("layers.cl", &KernelSource);
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
	
	   kernel[0] = clCreateKernel(program, "filter2D", &err);
	   if (!kernel[0] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel!\n");
	       exit(1);
	   }
#if 0
	   kernel[1] = clCreateKernel(program, "relu", &err);
	   if (!kernel[1] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel!\n");
	       exit(1);
	   }
	   kernel[2] = clCreateKernel(program, "maxpool", &err);
	   if (!kernel[2] || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel!\n");
	       exit(1);
	   }
#endif
	   // Create the input and output arrays in device memory for our calculation
	   d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY /*| CL_MEM_USE_MSMC_TI*/, mem_size_output, NULL, &err);
	   d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR /*| CL_MEM_USE_MSMC_TI*/, mem_size_image, h_image, &err);
	   d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR /*| CL_MEM_USE_MSMC_TI*/, mem_size_filter, h_filter, &err);
#if 0
	   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_MSMC_TI | CL_MEM_ALLOC_HOST_PTR, mem_size_A, NULL, &err);
	   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_MSMC_TI | CL_MEM_ALLOC_HOST_PTR, mem_size_A, NULL, &err);
	   clEnqueueWriteBuffer(commands, d_A, CL_TRUE, 0, mem_size_A, h_A, 0, NULL, NULL);
	   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_MSMC_TI | CL_MEM_ALLOC_HOST_PTR, mem_size_B, NULL, &err);
	   clEnqueueWriteBuffer(commands, d_B, CL_TRUE, 0, mem_size_B, h_B, 0, NULL, NULL);
#endif

	   if (!d_image || !d_filter || !d_output)
	   {
	       printf("Error: Failed to allocate device memory!\n");
	       exit(1);
	   }    
   	 
	   //Launch OpenCL kernel
	   size_t localWorkSize[2], globalWorkSize[2];
//	   size_t local,global;
//	   size_t localWork[2],globalWork[2];
	 
	   int wA = ipgm_img_width;
	   int wB = L1_KERNEL_SIZE;
	   err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&d_image);
	   err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&d_filter);
	   err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&d_output);
	   err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&wA);
	   err |= clSetKernelArg(kernel[0], 4, sizeof(int), (void *)&wB);

//	   err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&d_C);

//	   err |= clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&d_C);
//	   err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&d_D);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to set kernel arguments! %d\n", err);
       	       exit(1);
   	  }
 
	   localWorkSize[0] = ipgm_img_width/4;
	   localWorkSize[1] = ipgm_img_height/4;
	   globalWorkSize[0] = ipgm_img_width;
	   globalWorkSize[1] = ipgm_img_height;
	 
	   ptimer1 = PAPI_get_virt_usec();
	   /*Enqueue task for parallel execution*/
	   err = clEnqueueNDRangeKernel(commands, kernel[0], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
#if 0	   
	   clFlush(commands);

           local = ipgm_img_width*ipgm_img_height/8;
           global = ipgm_img_width*ipgm_img_height;

	   err = clEnqueueNDRangeKernel(commands, kernel[1], 1, NULL, &global,&local, 1, &event, &event1);	
	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
	   
	   clFlush(commands);
	   localWork[0] = ipgm_img_width/stride/4;
	   localWork[1] = ipgm_img_height/stride/4;
	   globalWork[0] = ipgm_img_width/stride;
	   globalWork[1] = ipgm_img_height/stride; 
	   err = clEnqueueNDRangeKernel(commands, kernel[2], 2, NULL, globalWork, localWork, 1, &event1, &event2);
	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
#endif
	   ptimer2 = PAPI_get_virt_usec();
	   printf("cl:main timing:PAPI clEnqueueNDRangeKernel %llu us\n",(ptimer2-ptimer1));	
	// clWaitForEvents(1, &event2);
	   clFinish(commands);
	   cl_ulong time_start, time_end;
           double total_time;
	   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	   total_time = time_end - time_start;
	   printf("cl:main timing:opencl clEnqueueNDRangeKernel %0.3f us\n", total_time / 1000.0);
	
	   /*Retrieve result from device*/
	   err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, mem_size_output, h_output, 0, NULL, NULL);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to read output array! %d\n", err);
	       exit(1);
	   }
	 
	   /* Copy to buffer - not needed */

	   printf("cl:main program:completed\n"); 
 
	   output_pgm.width = ipgm_img_width;
	   output_pgm.height = ipgm_img_height;
	   normalizeF2PGM(&output_pgm, h_output);
	
	   /* Output image */
	   writePGM(&output_pgm, "output.pgm");
		
	   destroyPGM(&input_pgm);
	   destroyPGM(&output_pgm);
	   
	   free(h_image);
	   free(h_filter);
	   free(h_output);
	
	   clReleaseMemObject(d_image);
	   clReleaseMemObject(d_filter);
	   clReleaseMemObject(d_output);
	
	   clReleaseProgram(program);
	   clReleaseKernel(kernel[0]);
	   clReleaseKernel(kernel[1]);
	   clReleaseKernel(kernel[2]);
	   clReleaseCommandQueue(commands);
	   clReleaseContext(context);
	
	   return 0;
}
