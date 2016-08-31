#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"
#include "data_utils.h"

#define FC_WG_SIZE	256

cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> devices;
cl_device_id target_device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;

FcLayer fc;
aocl_utils::scoped_aligned_ptr<DTYPE> ref_output;
aocl_utils::scoped_aligned_ptr<DTYPE> h_input;
cl_mem d_input;

bool init_opencl();
void compute_reference();
void compare();

int main(int argc, char **argv) {
	DataShape input_shape = {9216, 1, 1};
	cl_int status;
	size_t global_ws;
	size_t local_ws;

	init_opencl();
	fc.bot_shape = &input_shape; fc.W = NULL;	fc.b = NULL; fc.top_shape.z = 1;
	fc.top_shape.x = 4096; fc.top_shape.y = 1;
	fc.W = (WTYPE *)malloc(fc.bot_shape->x * fc.top_shape.x * sizeof(WTYPE));
	fc.b = (WTYPE *)malloc(fc.top_shape.x * sizeof(WTYPE));

	h_input.reset(fc.bot_shape->x);
	fc.h_output.reset(fc.top_shape.x);
	ref_output.reset(fc.top_shape.x);
	CHECK_NEAR(1, 2);
	// random input init
	rand_init(h_input, fc.bot_shape->x, 0);
	rand_init(fc.W, fc.top_shape.x * fc.bot_shape->x, 123);
	rand_init(fc.b, fc.top_shape.x, 321);

	d_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA | CL_MEM_COPY_HOST_PTR,
                fc.bot_shape->x * sizeof(DTYPE), h_input, &status);
	fc.d_input = &d_input;
	fc.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		fc.top_shape.x * sizeof(DTYPE), NULL, &status);
	
	fc.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.bot_shape->x * fc.top_shape.x * sizeof(WTYPE), fc.W, &status);
	fc.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.top_shape.x * sizeof(WTYPE), fc.b, &status);
	
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), fc.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(int), &fc.bot_shape->x);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);
	const unsigned char act = 1;
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned char), &act);
	checkError(status, "Failed to set argument %d", argi - 1);

    global_ws = fc.top_shape.x;
	local_ws = FC_WG_SIZE;

	cl_event event;
	std::cout << "Starting execution" << std::endl;
	const double start_time = getCurrentTimestamp();
	// launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &event);
	checkError(status, "Failed to launch conv kernel");
	clFinish(queue);

	// Get kernel profiling info
	cl_ulong start, end;
	status  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	status |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	checkError(status, "Error: could not get profile information");
	clReleaseEvent(event);

	std::cout << "Kernel time:" << (end-start) << std::endl;
	const double end_time = getCurrentTimestamp();
	const double total_time = (end_time - start_time)*1000;
	std::cout << "Conv Layer Runtime(ms) = " << total_time << std::endl;

	// read output from device buffer	
	status = clEnqueueReadBuffer(queue, fc.d_output, CL_TRUE, 0,
		fc.top_shape.x  * sizeof(DTYPE), fc.h_output, 0, NULL, NULL);
	checkError(status, "Failed to read data from the device");
	clFinish(queue);

	std::cout << "Computing reference output" << std::endl;
	// compute reference output and compare
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(conv.h_output, conv.top_shape.z, conv.top_shape.y, conv.top_shape.x, 1);
	compute_reference();
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(ref_output, conv.top_shape.z, conv.top_shape.y, conv.top_shape.x, 1);
	std::cout << "Comparing" << std::endl;
	compare();

	cleanup();
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
	
	std::string binary_file = getBoardBinaryFile("fc_kernel", target_device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &target_device, num_devices);
	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	
	
	// Command queue.
	queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	
	// Kernel.
	kernel = clCreateKernel(program, "fc_layer_relu", &status);
	checkError(status, "Failed to create kernel");
	std::cout << "OpenCL init done" << std::endl;
	return true;
}

void compute_reference() {
	DTYPE zero = 0.0f;
	for(int out = 0; out < fc.top_shape.x; out++) {
		DTYPE sum = 0.0f;
		for(int in = 0; in < fc.bot_shape->x; in++) {
			sum += h_input[in] * fc.W[out * fc.bot_shape->x + in];
		}
		sum += fc.b[out];
		ref_output[out] = std::max(zero, sum);
	}
}

void compare() {
	float mse = 0.0f;
	for(unsigned out = 0; out < fc.top_shape.x; out++) {
		CHECK_NEAR(ref_output[out], fc.h_output[out]);
		float diff = ref_output[out] - fc.h_output[out];
		mse += diff * diff;
	}
	mse /= fc.top_shape.x;
	std::cout << "MSE = " << mse << std::endl;
		
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
	clReleaseMemObject(fc.d_output);
	clReleaseMemObject(d_input);
	clReleaseMemObject(fc.d_W);
	clReleaseMemObject(fc.d_b);
	free(fc.W);
	free(fc.b);
}
