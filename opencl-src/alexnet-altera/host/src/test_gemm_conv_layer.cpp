#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"
#include "data_utils.h"

#define BLOCK_SIZE 64

cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> devices;
cl_device_id target_device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;

ConvLayer conv;
aocl_utils::scoped_aligned_ptr<DTYPE> ref_output;
aocl_utils::scoped_aligned_ptr<DTYPE> data_mat;
WTYPE * weight_mat;


bool init_opencl();
void compute_reference();
void compare();

int main(int argc, char **argv) {
	DataShape input_shape = {227, 227, 3};
	cl_int status;
	size_t global_ws[2];
	size_t local_ws[2];
	aocl_utils::scoped_aligned_ptr<DTYPE> out_mat;

	init_opencl();
	conv.bot_shape = &input_shape; conv.K = 11; conv.pad = 0;
	conv.W = NULL;	conv.b = NULL;	conv.stride = 4; conv.top_shape.z = 96;
	conv.top_shape.x = (conv.bot_shape->x - conv.K + 1 + 2 * conv.pad + conv.stride-1) / conv.stride;
	conv.top_shape.y = (conv.bot_shape->y - conv.K + 1 + 2 * conv.pad + conv.stride-1) / conv.stride;

	int dmat_cols = conv.K * conv.K * conv.bot_shape->z;
	int dmat_rows = conv.top_shape.x * conv.top_shape.y;
	int &wmat_rows = dmat_cols;
	int wmat_cols = conv.top_shape.z;

	// make dimensions multiple of BLOCK_SIZE
	int extra_dmat_rows = BLOCK_SIZE - dmat_rows % BLOCK_SIZE;
	int extra_dmat_cols = BLOCK_SIZE - dmat_cols % BLOCK_SIZE;
	int extra_wmat_cols = BLOCK_SIZE - wmat_cols % BLOCK_SIZE;
	int M = dmat_rows + extra_dmat_rows;
	int N = dmat_cols + extra_dmat_cols;
	int P = wmat_cols + extra_wmat_cols;

	conv.W = (WTYPE *)malloc(conv.K * conv.K * conv.bot_shape->z * conv.top_shape.z * sizeof(WTYPE));
	conv.b = (WTYPE *)malloc(P * sizeof(WTYPE));

	conv.h_input.reset((conv.bot_shape->x + 2 * conv.pad) * (conv.bot_shape->y + 2 * conv.pad) * conv.bot_shape->z);
	conv.h_output.reset(conv.top_shape.x * conv.top_shape.y * conv.top_shape.z);
	ref_output.reset(conv.top_shape.x * conv.top_shape.y * conv.top_shape.z);

	// random input init
	rand_init(conv.h_input, (conv.bot_shape->x + 2 * conv.pad) * (conv.bot_shape->y + 2 * conv.pad) * conv.bot_shape->z, 0);
	rand_init(conv.W, conv.K * conv.K * conv.bot_shape->z * conv.top_shape.z, 123);
	rand_init(conv.b, conv.top_shape.z, 321);


	std::cout << "Input shape = " << "(" << conv.bot_shape->z << "," << conv.bot_shape->y <<","<< conv.bot_shape->x << ")" << std::endl;
	std::cout << "Output shape = " << "(" << conv.top_shape.z << "," << conv.top_shape.y <<","<< conv.top_shape.x << ")" << std::endl;
	/*showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(conv.h_input, conv.bot_shape->z,
		conv.bot_shape->y + 2*conv.pad, conv.bot_shape->x + 2*conv.pad, 2);*/

	//showMat<WTYPE * >(conv.W, conv.bot_shape->z, conv.K, conv.K, 4);


	std::cout << "Data mat dim = " << M << "x" << N << std::endl;
	std::cout << "Weight mat dim = " << N << "x" << P << std::endl;

	weight_mat = (WTYPE *)malloc((wmat_rows + extra_dmat_cols) * (wmat_cols + extra_wmat_cols) * sizeof(WTYPE));
	data_mat.reset((dmat_rows + extra_dmat_rows) * (dmat_cols + extra_dmat_cols));
	out_mat.reset(M * P);

	// image to matrix conversion
	img2col(conv.h_input, data_mat, conv.bot_shape->z, conv.bot_shape->y + 2*conv.pad,
		conv.bot_shape->x + 2 * conv.pad, conv.K, conv.stride, 0, extra_dmat_rows, extra_dmat_cols);
	weight2col(conv.W, weight_mat, conv.bot_shape->z, conv.K, conv.top_shape.z, extra_dmat_cols, extra_wmat_cols);
	
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(data_mat, 1, M, N, 1);
	//showMat<WTYPE* >(weight_mat, 1, N, P, 1);

	conv.d_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA | CL_MEM_COPY_HOST_PTR,
                 M * N * sizeof(DTYPE), data_mat, &status);
	conv.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_2_ALTERA, 
		M * P * sizeof(DTYPE), NULL, &status);
	
	conv.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		N * P * sizeof(WTYPE), weight_mat, &status);
	conv.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		P * sizeof(WTYPE), conv.b, &status);
	
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(int), &N);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel, argi++, sizeof(int), &P);
	checkError(status, "Failed to set argument %d", argi - 1);

    global_ws[0] = P;
    global_ws[1] = M;

	local_ws[0] = BLOCK_SIZE;
	local_ws[1] = BLOCK_SIZE;

	cl_event event;
	std::cout << "Starting execution" << std::endl;
	const double start_time = getCurrentTimestamp();
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_ws, local_ws, 0, NULL, &event);
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
	
	status = clEnqueueReadBuffer(queue, conv.d_output, CL_TRUE, 0,
		M * P * sizeof(DTYPE), out_mat, 0, NULL, NULL);
	checkError(status, "Failed to read data from the device");
	clFinish(queue);
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(out_mat, 1, M, P, 1);

	col2img(out_mat, conv.h_output, conv.top_shape.z, conv.top_shape.y, conv.top_shape.x, extra_wmat_cols);
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(conv.h_output, conv.top_shape.z, conv.top_shape.y, conv.top_shape.x, 2);

	// compute reference output and compare
	std::cout << "Computing reference output" << std::endl;
	compute_reference();
	//showMat<aocl_utils::scoped_aligned_ptr<DTYPE>& >(ref_output, conv.top_shape.z, conv.top_shape.y, conv.top_shape.x, 2);

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
	
	std::string binary_file = getBoardBinaryFile("matrix_mult", target_device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &target_device, num_devices);
	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	
	
	// Command queue.
	queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	
	// Kernel.
	kernel = clCreateKernel(program, "matrixMult", &status);
	checkError(status, "Failed to create kernel");
	std::cout << "OpenCL init done" << std::endl;
	return true;
}

void compute_reference() {
	int H = conv.bot_shape->y  + 2*conv.pad;
	int W = conv.bot_shape->x  + 2*conv.pad;
	DTYPE zero = 0.0f;
	for(unsigned omap = 0; omap < conv.top_shape.z; omap++) {
		for(unsigned row = 0; row < conv.top_shape.y; row++) {
			int hstart = row * conv.stride;
			for(unsigned col = 0; col < conv.top_shape.x; col++) {
				int wstart = col * conv.stride;
				DTYPE sum = 0.0f;
				for(unsigned imap = 0; imap < conv.bot_shape->z; imap++) {
					for(unsigned kr = 0; kr < conv.K; kr++) {
						for(unsigned kc = 0; kc < conv.K; kc++) {
							sum += 	conv.W[omap * conv.bot_shape->z * conv.K * conv.K + imap * conv.K * conv.K + kr * conv.K + kc] *
								conv.h_input[imap * W * H + (hstart + kr) * W + wstart + kc];
						}
					}
				}
				sum += conv.b[omap];
				ref_output[(omap * conv.top_shape.y + row) * conv.top_shape.x + col] = std::max(zero, sum); 
				//ref_output[(omap * conv.top_shape.y + row) * conv.top_shape.x + col] = sum; 

			}	
		}
	}
}

void compare() {
	int N = conv.top_shape.z;
	int H = conv.top_shape.y;
	int W = conv.top_shape.x;
	float mse = 0.0f;
	for(unsigned omap = 0; omap < N; omap++) {
		for(unsigned row = 0; row < H; row++) {
			for(unsigned col = 0; col < W; col++) {
				CHECK_NEAR(ref_output[(omap * H + row)*W + col], conv.h_output[(omap * H + row)*W + col]);
				float diff = ref_output[(omap * H + row)*W + col] - conv.h_output[(omap * H + row)*W + col];
				mse += diff * diff;
			}
		}
	}
	mse /= (N*H*W);
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
	clReleaseMemObject(conv.d_input);
	clReleaseMemObject(conv.d_output);
	clReleaseMemObject(conv.d_W);
	clReleaseMemObject(conv.d_b);
	free(conv.W);
	free(conv.b);
	free(weight_mat);
}
