#include <stdio.h>
#include <iostream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"

cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> devices;
cl_device_id target_device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;

ConvLayer conv;

bool init_opencl();

int main(int argc, char **argv) {
	DataShape input_shape = {256, 256, 3};
	cl_int status;
	size_t global_ws[3];
	size_t local_ws[3];
	init_opencl();
	conv.bot_shape = &input_shape; conv.K = 11; conv.pad = 0;
	conv.W = NULL;	conv.b = NULL;	conv.stride = 4; conv.top_shape.z = 96;
	conv.top_shape.x = (conv.bot_shape->x - conv.K + 1 + 2*conv.pad + conv.stride-1)/conv.stride;
	conv.top_shape.y = (conv.bot_shape->y - conv.K + 1 + 2*conv.pad + conv.stride-1)/conv.stride;

	conv.W = (WTYPE *)malloc(conv.K*conv.K*conv.bot_shape->z*conv.top_shape.z*sizeof(WTYPE));
	conv.b = (WTYPE *)malloc(conv.top_shape.z*sizeof(WTYPE));

	conv.h_input.reset((conv.bot_shape->x+2*conv.pad) * (conv.bot_shape->y+2*conv.pad) * conv.bot_shape->z);
	conv.h_output.reset(conv.top_shape.x * conv.top_shape.y * conv.top_shape.z);

	conv.d_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
                (conv.bot_shape->x+2*conv.pad) * (conv.bot_shape->y+2*conv.pad) * conv.bot_shape->z * sizeof(DTYPE), NULL, &status);
	conv.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		conv.top_shape.x * conv.top_shape.y  * conv.top_shape.z * sizeof(DTYPE), NULL, &status);
	
	conv.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv.K * conv.K * conv.bot_shape->z * conv.top_shape.z * sizeof(WTYPE), conv.W, &status);
	conv.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv.top_shape.z * sizeof(WTYPE), conv.b, &status);
	
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
		
	/*status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.K);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.stride);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.bot_shape->z);
	checkError(status, "Failed to set argument %d", argi - 1);	
	int in_h = conv.bot_shape->y + 2*conv.pad;
	int in_w = conv.bot_shape->x + 2*conv.pad;
	status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.top_shape.z);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &in_h);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &in_w);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.top_shape.y);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(int), &conv.top_shape.x);
	checkError(status, "Failed to set argument %d", argi - 1);	*/
    global_ws[0] = 1;//conv.top_shape.x;
    global_ws[1] = 1;//conv.top_shape.y;
    global_ws[2] = 1;//conv.top_shape.z;

	local_ws[0] = 1;//global_ws[0];
	local_ws[1] = 1;//global_ws[1];
	local_ws[2] = 1;
	cl_event event;
	std::cout << "Starting execution" << std::endl;
	const double start_time = getCurrentTimestamp();
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_ws, local_ws, 0, NULL, &event);
	checkError(status, "Failed to launch conv kernel");
	clFinish(queue);
	cl_ulong start, end;
	status  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	status |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	checkError(status, "Error: could not get profile information");
	clReleaseEvent(event);

	std::cout << "Kernel time:" << (end-start) << std::endl;
	const double end_time = getCurrentTimestamp();
	const double total_time = (end_time - start_time)*1000;
	std::cout << "Conv Layer Runtime(ms) = " << total_time << std::endl;
	
}
bool init_opencl() {
	cl_int status;
	
	printf("Initializing OpenCL\n");
	
	if(!setCwdToExeDir()) {
	  return false;
	}
	
	// Get the OpenCL platform.
	platform = findPlatform("Altera");
	if(platform == NULL) {
	  printf("ERROR: Unable to find Altera OpenCL platform.\n");
	  return false;
	}
	
	// Query the available OpenCL device.
	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Found %d devices in the board. Using only one device for this app\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
	  printf("  %s\n", getDeviceName(devices[i]).c_str());
	}
	target_device = devices[0];	
	// Create the context.
	context = clCreateContext(NULL, num_devices, &target_device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");
	
	std::string binary_file = getBoardBinaryFile("conv_kernel", target_device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &target_device, num_devices);
	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	
	
	// Command queue.
	queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	
	// Kernel.
	kernel = clCreateKernel(program, "conv_3d_relu", &status);
	checkError(status, "Failed to create kernel");
	std::cout << "OpenCL init done" << std::endl;
	return true;
}

void cleanup() {
	std::cout << "Releasing all OpenCL objects" << std::endl;
	clReleaseKernel(kernel);

	if(queue) {
		clReleaseCommandQueue(queue);
	}
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
	clReleaseMemObject(conv.d_input);
	clReleaseMemObject(conv.d_output);
	clReleaseMemObject(conv.d_W);
	clReleaseMemObject(conv.d_b);
	free(conv.W);
	free(conv.b);
}
