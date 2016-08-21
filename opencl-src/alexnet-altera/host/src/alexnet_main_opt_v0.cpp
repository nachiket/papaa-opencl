#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"
#include "device_utils.h"
#include "data_utils.h"
#include "cnpy.h"

#define SSTR( x ) static_cast< std::ostringstream & >( \
	( std::ostringstream() << std::dec << x ) ).str()

using namespace aocl_utils;
using namespace std;
using namespace cv;

cl_platform_id platform = NULL;
unsigned num_devices = 0;
unsigned num_kernels = 5;
scoped_array<cl_device_id> devices;
cl_device_id target_device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
scoped_array<cl_kernel> kernel;

scoped_aligned_ptr<DTYPE> h_input_img;
ConvLayer conv1;
PoolLayer pool1;
BatchNormLayer norm1;
SliceLayer slice1;
ConvLayer conv2_1;
ConvLayer conv2_2;
ConcatLayer concat1;
PoolLayer pool2;
BatchNormLayer norm2;

ConvLayer conv3;
SliceLayer slice2;
ConvLayer conv4_1;
ConvLayer conv4_2;
ConcatLayer concat2;

SliceLayer slice3;
ConvLayer conv5_1;
ConvLayer conv5_2;
ConcatLayer concat3;
PoolLayer pool5;

FcLayer fc6;
FcLayer fc7;
ActLayer relu7;
FcLayer fc8;
ActLayer smax;

// Full model container
cnpy::npz_t full_model;
// Buffer to read output of grouped conv layers and concatenate. The size of these
// will be max requirement across whole network.
cl_mem d_concat_buff;
scoped_aligned_ptr<DTYPE> h_concat_buff, h_concat_buff2;

void allocateHostBuffer();
void allocateDeviceBuffer();
bool init_opencl();
void initModel(std::string);
void initNetParams(DataShape &input_shape);
unsigned int runApplication();

void printHelp(char *app) {
	cout << "------------------------------------" << endl;
	cout << "Usage:" << endl;
	cout << app;
	cout << " -model <model file>";
	cout << " -m <mode>";
	cout << " [-i] <image path>";
	cout << " [-l] <image list>";
	cout << " [-d] <test image dir>" << endl;
	cout << "------------------------------------" << endl;
}

int main(int argc, char **argv) {
	cout << "ImageNet object classification using AlexNet" << endl;
	std::string model_path;
	Options options(argc, argv);
	if(options.has("model")) {
		model_path = options.get<std::string>("model");
	} else {
		printHelp(argv[0]);
		exit(1);
	}
	DataShape input_shape = {227, 227, 3};
	initNetParams(input_shape);
	initModel(model_path);
	init_opencl();
	allocateHostBuffer();
	allocateDeviceBuffer();
	//TODO: Init image
	runApplication();
	cleanup();
	return 0;
}

void clubNormParams(BatchNormLayer &norm, cnpy::NpyArray &beta,
	cnpy::NpyArray &gamma, cnpy::NpyArray &mean, cnpy::NpyArray &inv_std) {
	
	assert(norm.top_shape.z == beta.shape[0]);
	assert(norm.top_shape.z == gamma.shape[0]);
	assert(norm.top_shape.z == mean.shape[0]);
	assert(norm.top_shape.z == inv_std.shape[0]);
	WTYPE * p_beta = (WTYPE *)beta.data;
	WTYPE * p_gamma = (WTYPE *)gamma.data;
	WTYPE * p_mean = (WTYPE *)mean.data;
	WTYPE * p_inv_std = (WTYPE *)inv_std.data;
	for(int map = 0; map < norm.top_shape.z; map++) {
		norm.scale[map] = p_gamma[map] * p_inv_std[map];
		norm.offset[map] = -p_mean[map] * p_inv_std[map] * p_gamma[map] + p_beta[map];
	}
}
// Read numpy arrays from npz file and assign the model param pointers
void initModel(std::string model_path) {

	std::string param_prefix = "arr_";
	std::string param_name;
	// Read npz model file into global container.
	full_model = cnpy::npz_load(model_path);
	// Allocate buffers for normalization layer parameters.
	// Since we are considering clubbed paramters, they are directly not
	// present in the full_model. Hence they need to be created
	norm1.scale = new WTYPE [norm1.top_shape.z];
	norm1.offset = new WTYPE [norm1.top_shape.z];
	norm2.scale = new WTYPE [norm2.top_shape.z];
	norm2.offset = new WTYPE [norm2.top_shape.z];

	cnpy::npz_t::iterator pstart = full_model.begin();
	for(; pstart != full_model.end(); pstart++) {
		std::cout << pstart->first << endl;
		cnpy::NpyArray param = pstart->second;
		for(unsigned int s = 0; s < param.shape.size(); s++) {
			std::cout << param.shape[s] << ",";
		}
		std::cout << std::endl;
	}
	int param_idx = 0;
	conv1.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv1.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	// Extract batch norm layer 1 params and club them
	cnpy::NpyArray &norm1_beta = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm1_gamma = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm1_mean = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm1_inv_std = full_model[param_prefix + SSTR(param_idx++)];
	clubNormParams(norm1, norm1_beta, norm1_gamma, norm1_mean, norm1_inv_std);

	conv2_1.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv2_1.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv2_2.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv2_2.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	// Extract batch norm layer 2 params and club them
	cnpy::NpyArray &norm2_beta = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm2_gamma = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm2_mean = full_model[param_prefix + SSTR(param_idx++)];
	cnpy::NpyArray &norm2_inv_std = full_model[param_prefix + SSTR(param_idx++)];
	clubNormParams(norm2, norm2_beta, norm2_gamma, norm2_mean, norm2_inv_std);

	conv3.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv3.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;

	conv4_1.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv4_1.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv4_2.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv4_2.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;

	conv5_1.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv5_1.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv5_2.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	conv5_2.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;

	fc6.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	fc6.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	fc7.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	fc7.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	fc8.W = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
	fc8.b = (WTYPE *)full_model[param_prefix + SSTR(param_idx++)].data;
}
unsigned int runApplication() {
	cl_int status;
	size_t global_work_size[3];
	size_t local_work_size[3];
	scoped_array<cl_event> kernel_event(17);
	std::cout << "------Starting execution----" << std::endl;
	const double start_time = getCurrentTimestamp();

	// zero pad the input image and transfer to device memory allocated for conv1 input.
	zeropadAndTx(h_input_img, conv1.h_input, conv1.bot_shape->z,
		conv1.bot_shape->y, conv1.bot_shape->x, conv1.pad, conv1.pad, conv1.d_input, queue, true);
	// conv 1 -> relu1 -> pool1 -> norm1 layer execution
	setKernelArgs(conv1, kernel[0], conv1.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[0]);
	checkError(status, "Failed to launch conv1 kernel");
	setKernelArgs(pool1, kernel[1], global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[1], 3, NULL, global_work_size, NULL, 1, &kernel_event[0], &kernel_event[1]);
	checkError(status, "Failed to launch pool1 kernel");
	setKernelArgs(norm1, kernel[3], global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[3], 3, NULL, global_work_size, NULL, 1, &kernel_event[1], &kernel_event[2]);
	checkError(status, "Failed to launch norm1 kernel");

	// read the norm1 output and zero pad appropriately and then split maps to feed into 2 conv layers(group = 2)
	status = clEnqueueReadBuffer(queue, norm1.d_output, CL_TRUE, 0,
		norm1.top_shape.x * norm1.top_shape.y * norm1.top_shape.z * sizeof(DTYPE), norm1.h_output, 1, &kernel_event[2], NULL);
	checkError(status, "Failed to read data from the device");
	zeropadAndTx(norm1.h_output, h_concat_buff, norm1.bot_shape->z,
		norm1.bot_shape->y, norm1.bot_shape->x, conv2_1.pad, conv2_1.pad, conv2_1.d_input, queue, false);
	status = clEnqueueWriteBuffer(queue, conv2_1.d_input, CL_FALSE, 0,
		conv2_1.bot_shape->z * (conv2_1.bot_shape->x+2*conv2_1.pad) * (conv2_1.bot_shape->y+2*conv2_1.pad) * sizeof(DTYPE),
		h_concat_buff, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, conv2_2.d_input, CL_FALSE, 0,
		conv2_2.bot_shape->z * (conv2_2.bot_shape->x+2*conv2_2.pad) * (conv2_2.bot_shape->y+2*conv2_2.pad) * sizeof(DTYPE),
		// split the maps into 2 half using pointer offset
		(DTYPE *)h_concat_buff + conv2_1.bot_shape->z * (conv2_1.bot_shape->x+2*conv2_1.pad) * (conv2_1.bot_shape->y+2*conv2_1.pad),
		0, NULL, NULL);
	checkError(status, "Failed to transfer data to the device\n");
	clFinish(queue);
	// conv2 layer, group size = 2
	setKernelArgs(conv2_1, kernel[0], conv2_1.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[3]);
	checkError(status, "Failed to launch conv2_2 kernel");
	setKernelArgs(conv2_2, kernel[0], conv2_2.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[4]);
	checkError(status, "Failed to launch conv2_2 kernel");
	// read the output of split conv layers
	status = clEnqueueReadBuffer(queue, conv2_1.d_output, CL_TRUE, 0,
		conv2_1.top_shape.x * conv2_1.top_shape.y * conv2_1.top_shape.z * sizeof(DTYPE), h_concat_buff, 1, &kernel_event[3], NULL);
	status = clEnqueueReadBuffer(queue, conv2_2.d_output, CL_TRUE, 0,
		conv2_2.top_shape.x * conv2_2.top_shape.y * conv2_2.top_shape.z * sizeof(DTYPE),
		(DTYPE *)h_concat_buff + conv2_1.top_shape.x * conv2_1.top_shape.y * conv2_1.top_shape.z, 1, &kernel_event[4], NULL);
	status = clEnqueueWriteBuffer(queue, d_concat_buff, CL_FALSE, 0,
		pool2.bot_shape->x * pool2.bot_shape->y * pool2.bot_shape->z * sizeof(DTYPE),
		h_concat_buff, 0, NULL, NULL);
	clFinish(queue);
	//
	// pool2 -> norm2
	setKernelArgs(pool2, kernel[1], global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[1], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[5]);
	checkError(status, "Failed to launch pool2 kernel");
	setKernelArgs(norm2, kernel[3], global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[3], 3, NULL, global_work_size, NULL, 1, &kernel_event[5], &kernel_event[6]);
	checkError(status, "Failed to launch norm1 kernel");
	// read norm2 output and zero pad according to conv3 layer paramters
	status = clEnqueueReadBuffer(queue, norm2.d_output, CL_TRUE, 0,
		norm2.top_shape.x * norm2.top_shape.y * norm2.top_shape.z * sizeof(DTYPE), h_concat_buff, 1, &kernel_event[6], NULL);
	zeropadAndTx(h_concat_buff, conv3.h_input, conv3.bot_shape->z,
		conv3.bot_shape->y, conv3.bot_shape->x, conv3.pad, conv3.pad, conv3.d_input, queue, true);
	clFinish(queue);
	// conv3 -> relu3
	setKernelArgs(conv3, kernel[0], conv3.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[7]);
	checkError(status, "Failed to launch conv3 kernel");
	// read the conv3 output and zero pad appropriately and then split maps to feed into 2 conv layers(group = 2)
	status = clEnqueueReadBuffer(queue, conv3.d_output, CL_TRUE, 0,
		conv3.top_shape.x * conv3.top_shape.y * conv3.top_shape.z * sizeof(DTYPE), conv3.h_output, 1, &kernel_event[7], NULL);
	checkError(status, "Failed to read data from the device");
	zeropadAndTx(conv3.h_output, h_concat_buff, conv3.bot_shape->z,
		conv3.bot_shape->y, conv3.bot_shape->x, conv4_1.pad, conv4_1.pad, conv4_1.d_input, queue, false);
	status = clEnqueueWriteBuffer(queue, conv4_1.d_input, CL_FALSE, 0,
		conv4_1.bot_shape->z * (conv4_1.bot_shape->x+2*conv4_1.pad) * (conv4_1.bot_shape->y+2*conv4_1.pad) * sizeof(DTYPE),
		h_concat_buff, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, conv4_2.d_input, CL_FALSE, 0,
		conv4_2.bot_shape->z * (conv4_2.bot_shape->x+2*conv4_2.pad) * (conv4_2.bot_shape->y+2*conv4_2.pad) * sizeof(DTYPE),
		// split the maps into 2 half using pointer offset
		(DTYPE *)h_concat_buff + conv4_1.bot_shape->z * (conv4_1.bot_shape->x+2*conv4_1.pad) * (conv4_1.bot_shape->y+2*conv4_1.pad),
		0, NULL, NULL);
	checkError(status, "Failed to transfer data to the device\n");
	clFinish(queue);
	// conv4 layer, group size = 2
	setKernelArgs(conv4_1, kernel[0], conv4_1.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[8]);
	checkError(status, "Failed to launch conv4_1 kernel");
	setKernelArgs(conv4_2, kernel[0], conv4_2.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[9]);
	checkError(status, "Failed to launch conv4_2 kernel");
	// read the output of split conv layers
	status = clEnqueueReadBuffer(queue, conv4_1.d_output, CL_TRUE, 0,
		conv4_1.top_shape.x * conv4_1.top_shape.y * conv4_1.top_shape.z * sizeof(DTYPE), h_concat_buff, 1, &kernel_event[8], NULL);
	status = clEnqueueReadBuffer(queue, conv4_2.d_output, CL_TRUE, 0,
		conv4_2.top_shape.x * conv4_2.top_shape.y * conv4_2.top_shape.z * sizeof(DTYPE),
		(DTYPE *)h_concat_buff + conv4_1.top_shape.x * conv4_1.top_shape.y * conv4_1.top_shape.z, 1, &kernel_event[9], NULL);
	zeropadAndTx(h_concat_buff, h_concat_buff2, concat2.top_shape.z,
		concat2.top_shape.y, concat2.top_shape.x, conv5_1.pad, conv5_1.pad, conv5_1.d_input, queue, false);
	status = clEnqueueWriteBuffer(queue, conv5_1.d_input, CL_FALSE, 0,
		conv5_1.bot_shape->z * (conv5_1.bot_shape->x+2*conv5_1.pad) * (conv5_1.bot_shape->y+2*conv5_1.pad) * sizeof(DTYPE),
		h_concat_buff2, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, conv5_2.d_input, CL_FALSE, 0,
		conv5_2.bot_shape->z * (conv5_2.bot_shape->x+2*conv5_2.pad) * (conv5_2.bot_shape->y+2*conv5_2.pad) * sizeof(DTYPE),
		// split the maps into 2 half using pointer offset
		(DTYPE *)h_concat_buff2 + conv5_1.bot_shape->z * (conv5_1.bot_shape->x+2*conv5_1.pad) * (conv5_1.bot_shape->y+2*conv5_1.pad),
		0, NULL, NULL);
	checkError(status, "Failed to transfer data to the device\n");
	clFinish(queue);
	// conv5 layer, group size = 2
	setKernelArgs(conv5_1, kernel[0], conv5_1.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[10]);
	checkError(status, "Failed to launch conv5_1 kernel");
	setKernelArgs(conv5_2, kernel[0], conv5_2.d_input, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[11]);
	checkError(status, "Failed to launch conv5_2 kernel");
	// read the output of split conv layers
	status = clEnqueueReadBuffer(queue, conv5_1.d_output, CL_TRUE, 0,
		conv5_1.top_shape.x * conv5_1.top_shape.y * conv5_1.top_shape.z * sizeof(DTYPE), h_concat_buff, 1, &kernel_event[10], NULL);
	status = clEnqueueReadBuffer(queue, conv5_2.d_output, CL_TRUE, 0,
		conv5_2.top_shape.x * conv5_2.top_shape.y * conv5_2.top_shape.z * sizeof(DTYPE),
		(DTYPE *)h_concat_buff + conv5_1.top_shape.x * conv5_1.top_shape.y * conv5_1.top_shape.z, 1, &kernel_event[11], NULL);
	status = clEnqueueWriteBuffer(queue, d_concat_buff, CL_FALSE, 0,
		concat3.top_shape.x * concat3.top_shape.y * concat3.top_shape.z * sizeof(DTYPE),
		h_concat_buff, 0, NULL, NULL);
	clFinish(queue);
	//  pool5
	setKernelArgs(pool5, kernel[1], global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[1], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[12]);
	checkError(status, "Failed to launch pool5 kernel");
	// fc6
	unsigned char act_en = 1;
	setKernelArgs(fc6, kernel[2], &act_en, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[2], 3, NULL, global_work_size, NULL, 1, &kernel_event[12], &kernel_event[13]);
	checkError(status, "Failed to launch fc6 kernel");
	setKernelArgs(fc7, kernel[2], &act_en, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[2], 3, NULL, global_work_size, NULL, 1, &kernel_event[13], &kernel_event[14]);
	checkError(status, "Failed to launch fc7 kernel");
	act_en = 0;
	setKernelArgs(fc8, kernel[2], &act_en, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[2], 3, NULL, global_work_size, NULL, 1, &kernel_event[14], &kernel_event[15]);
	checkError(status, "Failed to launch fc8 kernel");

	setKernelArgs(smax, kernel[4], global_work_size);
	local_work_size[0] = global_work_size[0];
	local_work_size[1] = global_work_size[1];
	local_work_size[2] = global_work_size[2];
	status = clEnqueueNDRangeKernel(queue, kernel[4], 3, NULL, global_work_size, local_work_size, 1, &kernel_event[15], &kernel_event[16]);
	checkError(status, "Failed to launch smax kernel");
	clFinish(queue);

	const double end_time = getCurrentTimestamp();
	const double total_time = end_time - start_time;
	std::cout << "Runtime = " << total_time << std::endl;
	// Read the final results
	cout << "Reading the output from the device" << endl;
	status = clEnqueueReadBuffer(queue, *smax.d_output, CL_TRUE, 0,
		smax.top_shape.x * smax.top_shape.y * smax.top_shape.z * sizeof(DTYPE), *smax.h_output, 0, NULL, NULL);
	checkError(status, "Failed to read data from the device");
	cout << "Done---------------" << endl;
	return 0;
}
void initNetParams(DataShape &input_shape) {
	cout << "CNN model initialization\n";
	conv1.bot_shape = &input_shape; conv1.K = 11; conv1.pad = 0;
	conv1.W = NULL;	conv1.b = NULL;	conv1.stride = 4; conv1.top_shape.z = 96;
	conv1.top_shape.x = (conv1.bot_shape->x - conv1.K + 1 + 2*conv1.pad + conv1.stride-1)/conv1.stride;
	conv1.top_shape.y = (conv1.bot_shape->y - conv1.K + 1 + 2*conv1.pad + conv1.stride-1)/conv1.stride;

	std::cout << "conv1:" << conv1.top_shape.z << "," << conv1.top_shape.y << "," << conv1.top_shape.x << endl;

	pool1.bot_shape = &conv1.top_shape; pool1.type = MAX; pool1.stride = 2; pool1.winSize = 3; pool1.pad = 0;
	// See Lasagne Pool layer output calculation
	pool1.top_shape.x = (pool1.bot_shape->x + 2*pool1.pad - pool1.winSize + 1 + pool1.stride - 1)/pool1.stride;
	pool1.top_shape.y = (pool1.bot_shape->y + 2*pool1.pad - pool1.winSize + 1 + pool1.stride - 1)/pool1.stride;
	pool1.top_shape.z = pool1.bot_shape->z;
	std::cout << "pool1:" << pool1.top_shape.z << "," << pool1.top_shape.y << "," << pool1.top_shape.x << endl;
	norm1.bot_shape = &pool1.top_shape;	norm1.scale = NULL;	norm1.offset = NULL;
	norm1.top_shape.x = norm1.bot_shape->x; norm1.top_shape.y = norm1.bot_shape->y; norm1.top_shape.z = norm1.bot_shape->z;

	slice1.bot_shape = &norm1.top_shape;
	slice1.top_shape_0.x = slice1.bot_shape->x; slice1.top_shape_0.y = slice1.bot_shape->y; slice1.top_shape_0.z = slice1.bot_shape->z/2;
	slice1.top_shape_1.x = slice1.bot_shape->x; slice1.top_shape_1.y = slice1.bot_shape->y; slice1.top_shape_1.z = slice1.bot_shape->z/2;
	conv2_1.bot_shape = &slice1.top_shape_0; conv2_1.K = 5; conv2_1.pad = 2;
	conv2_1.W = NULL;	conv2_1.b = NULL;	conv2_1.stride = 1; conv2_1.top_shape.z = 128;
	conv2_1.top_shape.x = (conv2_1.bot_shape->x - conv2_1.K + 1 + 2*conv2_1.pad + conv2_1.stride-1)/conv2_1.stride;
	conv2_1.top_shape.y = (conv2_1.bot_shape->y - conv2_1.K + 1 + 2*conv2_1.pad + conv2_1.stride-1)/conv2_1.stride;

	conv2_2.bot_shape = &slice1.top_shape_1; conv2_2.K = 5; conv2_2.pad = 2;
	conv2_2.W = NULL;	conv2_2.b = NULL; conv2_2.stride = 1; conv2_2.top_shape.z = 128;
	conv2_2.top_shape.x = (conv2_2.bot_shape->x - conv2_2.K + 1 + 2*conv2_2.pad + conv2_2.stride-1)/conv2_2.stride;
	conv2_2.top_shape.y = (conv2_2.bot_shape->y - conv2_2.K + 1 + 2*conv2_2.pad + conv2_2.stride-1)/conv2_2.stride;

	concat1.bot_shape_0 = &conv2_1.top_shape; concat1.bot_shape_1 = &conv2_2.top_shape;
	concat1.top_shape.x = concat1.bot_shape_0->x; concat1.top_shape.y = concat1.bot_shape_0->y; concat1.top_shape.z = concat1.bot_shape_0->z + concat1.bot_shape_1->z;

	std::cout << "conv2:" << concat1.top_shape.z << "," << concat1.top_shape.y << "," << concat1.top_shape.x << endl;

	pool2.bot_shape = &concat1.top_shape; pool2.type = MAX; pool2.stride = 2; pool2.winSize = 3; pool2.pad = 0;
	pool2.top_shape.x = (pool2.bot_shape->x + 2*pool2.pad - pool2.winSize + 1 + pool2.stride - 1)/pool2.stride;
	pool2.top_shape.y = (pool2.bot_shape->y + 2*pool2.pad - pool2.winSize + 1 + pool2.stride - 1)/pool2.stride;
	pool2.top_shape.z = pool2.bot_shape->z;

	norm2.bot_shape = &pool2.top_shape;	norm2.scale = NULL;	norm2.offset = NULL;
	norm2.top_shape.x = norm2.bot_shape->x; norm2.top_shape.y = norm2.bot_shape->y; norm2.top_shape.z = norm2.bot_shape->z;

	conv3.bot_shape = &norm2.top_shape; conv3.K = 3; conv3.pad = 1;
	conv3.W = NULL;	conv3.b = NULL;	conv3.stride = 1; conv3.top_shape.z = 384;
	conv3.top_shape.x = (conv3.bot_shape->x - conv3.K + 1 + 2*conv3.pad + conv3.stride-1)/conv3.stride;
	conv3.top_shape.y = (conv3.bot_shape->y - conv3.K + 1 + 2*conv3.pad + conv3.stride-1)/conv3.stride;
	std::cout << "conv3:" << conv3.top_shape.z << "," << conv3.top_shape.y << "," << conv3.top_shape.x << std::endl;
	slice2.bot_shape = &conv3.top_shape;
	slice2.top_shape_0.x = slice2.bot_shape->x; slice2.top_shape_0.y = slice2.bot_shape->y; slice2.top_shape_0.z = slice2.bot_shape->z/2;
	slice2.top_shape_1.x = slice2.bot_shape->x; slice2.top_shape_1.y = slice2.bot_shape->y; slice2.top_shape_1.z = slice2.bot_shape->z/2;

	conv4_1.bot_shape = &slice2.top_shape_0; conv4_1.K = 3;	conv4_1.pad = 1;
	conv4_1.W = NULL;	conv4_1.b = NULL;	conv4_1.stride = 1; conv4_1.top_shape.z = 192;
	conv4_1.top_shape.x = (conv4_1.bot_shape->x - conv4_1.K + 1 + 2*conv4_1.pad + conv4_1.stride-1)/conv4_1.stride;
	conv4_1.top_shape.y = (conv4_1.bot_shape->y - conv4_1.K + 1 + 2*conv4_1.pad + conv4_1.stride-1)/conv4_1.stride;

	conv4_2.bot_shape = &slice2.top_shape_1; conv4_2.K = 3; conv4_2.pad = 1;
	conv4_2.W = NULL;	conv4_2.b = NULL; conv4_2.stride = 1; conv4_2.top_shape.z = 192;
	conv4_2.top_shape.x = (conv4_2.bot_shape->x - conv4_2.K + 1 + 2*conv4_2.pad + conv4_2.stride-1)/conv4_2.stride;
	conv4_2.top_shape.y = (conv4_2.bot_shape->y - conv4_2.K + 1 + 2*conv4_2.pad + conv4_2.stride-1)/conv4_2.stride;

	concat2.bot_shape_0 = &conv4_1.top_shape; concat2.bot_shape_1 = &conv4_2.top_shape;
	concat2.top_shape.x = concat2.bot_shape_0->x; concat2.top_shape.y = concat2.bot_shape_0->y; concat2.top_shape.z = concat2.bot_shape_0->z + concat2.bot_shape_1->z;


	slice3.bot_shape = &concat2.top_shape;
	slice3.top_shape_0.x = slice3.bot_shape->x; slice3.top_shape_0.y = slice3.bot_shape->y; slice3.top_shape_0.z = slice3.bot_shape->z/2;
	slice3.top_shape_1.x = slice3.bot_shape->x; slice3.top_shape_1.y = slice3.bot_shape->y; slice3.top_shape_1.z = slice3.bot_shape->z/2;

	conv5_1.bot_shape = &slice3.top_shape_0; conv5_1.K = 3;	conv5_1.pad = 1;
	conv5_1.W = NULL;	conv5_1.b = NULL;	conv5_1.stride = 1; conv5_1.top_shape.z = 128;
	conv5_1.top_shape.x = (conv5_1.bot_shape->x - conv5_1.K + 1 + 2*conv5_1.pad + conv5_1.stride-1)/conv5_1.stride;
	conv5_1.top_shape.y = (conv5_1.bot_shape->y - conv5_1.K + 1 + 2*conv5_1.pad + conv5_1.stride-1)/conv5_1.stride;

	conv5_2.bot_shape = &slice3.top_shape_1; conv5_2.K = 3; conv5_2.pad = 1;
	conv5_2.W = NULL;	conv5_2.b = NULL; conv5_2.stride = 1; conv5_2.top_shape.z = 128;
	conv5_2.top_shape.x = (conv5_2.bot_shape->x - conv5_2.K + 1 + 2*conv5_2.pad + conv5_2.stride-1)/conv5_2.stride;
	conv5_2.top_shape.y = (conv5_2.bot_shape->y - conv5_2.K + 1 + 2*conv5_2.pad + conv5_2.stride-1)/conv5_2.stride;

	concat3.bot_shape_0 = &conv5_1.top_shape; concat3.bot_shape_1 = &conv5_2.top_shape;
	concat3.top_shape.x = concat3.bot_shape_0->x; concat3.top_shape.y = concat3.bot_shape_0->y; concat3.top_shape.z = concat3.bot_shape_0->z + concat3.bot_shape_1->z;

	std::cout << "conv5:" << concat3.top_shape.z << "," << concat3.top_shape.y << "," << concat3.top_shape.x << endl;

	pool5.bot_shape = &concat3.top_shape; pool5.type = MAX; pool5.stride = 2; pool5.winSize = 3; pool5.pad = 0;
	pool5.top_shape.x = (pool5.bot_shape->x + 2*pool5.pad - pool5.winSize + 1 + pool5.stride - 1)/pool5.stride;
	pool5.top_shape.y = (pool5.bot_shape->y + 2*pool5.pad - pool5.winSize + 1 + pool5.stride - 1)/pool5.stride;
	pool5.top_shape.z = pool5.bot_shape->z;

	std::cout << "pool5:" << pool5.top_shape.z << "," << pool5.top_shape.y << "," << pool5.top_shape.x << endl;
	fc6.bot_shape = &pool5.top_shape; fc6.W = NULL;	fc6.b = NULL; fc6.no_units = 4096;
	fc6.top_shape.z = 1; fc6.top_shape.y = 1; fc6.top_shape.x = fc6.no_units;
	std::cout << "fc6:" << fc6.top_shape.z << "," << fc6.top_shape.y << "," << fc6.top_shape.x << endl;

	fc7.bot_shape = &fc6.top_shape; fc7.W = NULL;	fc7.b = NULL; fc7.no_units = 4096;
	fc7.top_shape.z = 1; fc7.top_shape.y = 1; fc7.top_shape.x = fc7.no_units;
	std::cout << "fc7:" << fc7.top_shape.z << "," << fc7.top_shape.y << "," << fc7.top_shape.x << endl;

	fc8.bot_shape = &fc7.top_shape; fc8.W = NULL;	fc8.b = NULL; fc8.no_units = 1000;
	fc8.top_shape.z = 1; fc8.top_shape.y = 1; fc8.top_shape.x = fc8.no_units;
	std::cout << "fc8:" << fc8.top_shape.z << "," << fc8.top_shape.y << "," << fc8.top_shape.x << endl;

	smax.bot_shape = &fc8.top_shape; smax.type = SOFTMAX; smax.top_shape.x = smax.bot_shape->x;
	smax.top_shape.y = smax.bot_shape->y; smax.top_shape.z = smax.bot_shape->z;
}

// Allocate host buffers for intermediate map storage.
// Some of them are not going to be used as the layers directly communicate via device memory.
// Still allocating to keep it uniform
void allocateHostBuffer() {
	cout << "Allocating host memory for inputs and outputs\n";
	h_input_img.reset(conv1.bot_shape->x * conv1.bot_shape->y * conv1.bot_shape->z);
	// FIXME: this buffer must be as large as the max buffer size required for concatenation. Check if the norm1 output
	// requires largest buffer
	h_concat_buff.reset(concat1.top_shape.x * concat1.top_shape.y * concat1.top_shape.z);
	h_concat_buff2.reset(concat1.top_shape.x * concat1.top_shape.y * concat1.top_shape.z);
	allocConvHostBuff(conv1);
	// ActLayer performs in-place ops. No need of output buffer.
	allocPoolHostBuff(pool1);
	allocNormHostBuff(norm1, pool1.h_output);

	allocConvHostBuff(conv2_1);
	allocConvHostBuff(conv2_2);
	allocPoolHostBuff(pool2);
	allocNormHostBuff(norm2, pool2.h_output);

	allocConvHostBuff(conv3);

	allocConvHostBuff(conv4_1);
	allocConvHostBuff(conv4_2);

	allocConvHostBuff(conv5_1);
	allocConvHostBuff(conv5_2);
	allocPoolHostBuff(pool5);

	allocFcHostBuff(fc6, pool5.h_output);
	allocFcHostBuff(fc7, fc6.h_output);
	allocFcHostBuff(fc8, fc7.h_output);

	smax.h_input = &fc8.h_output;
	smax.h_output = smax.h_input;

}


void allocateDeviceBuffer() {

	cl_int status;
	cout << "Allocating device memory for intermediate data and model params." << endl;
	// FIXME: this buffer must be as large as the max buffer size required for concatenation. Check if the norm1 output
	// requires largest buffer
	d_concat_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
		concat1.top_shape.x *concat1.top_shape.y * concat1.top_shape.z * sizeof(DTYPE), NULL, &status);
	allocConvDevBuff(context, conv1);
	// ActLayer performs in-place ops. No need of output buffer.
	allocPoolDevBuff(context, pool1, conv1.d_output);
	allocBatchNormDevBuff(context, norm1, pool1.d_output, norm1.top_shape.z);

	allocConvDevBuff(context, conv2_1);
	allocConvDevBuff(context, conv2_2);
	allocPoolDevBuff(context, pool2, d_concat_buff);
	allocBatchNormDevBuff(context, norm2, pool2.d_output, norm1.top_shape.z);

	allocConvDevBuff(context, conv3);

	allocConvDevBuff(context, conv4_1);
	allocConvDevBuff(context, conv4_2);

	allocConvDevBuff(context, conv5_1);
	allocConvDevBuff(context, conv5_2);
	allocPoolDevBuff(context, pool5, d_concat_buff);

	allocFcDevBuff(context, fc6, pool5.d_output);
	allocFcDevBuff(context, fc7, fc6.d_output);
	allocFcDevBuff(context, fc8, fc7.d_output);

	smax.d_input = &fc8.d_output;
	smax.d_output = smax.d_input;
}

//------------------------------------------------------
// Taken from Altera design examples
// Initializes the OpenCL objects.
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
	
	std::string binary_file = getBoardBinaryFile("cnn_kernels_opt_v0", target_device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &target_device, num_devices);
	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");
	
	kernel.reset(num_kernels);
	
	// Command queue.
	queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	
	// Kernel.
	kernel[0] = clCreateKernel(program, "conv_3d_relu", &status);
	checkError(status, "Failed to create kernel");
	kernel[1] = clCreateKernel(program, "maxpool_3d", &status);
	checkError(status, "Failed to create kernel");
	kernel[2] = clCreateKernel(program, "fc_layer_relu", &status);
	checkError(status, "Failed to create kernel");
	kernel[3] = clCreateKernel(program, "batch_norm_layer", &status);
	checkError(status, "Failed to create kernel");
	kernel[4] = clCreateKernel(program, "softmax", &status);
	checkError(status, "Failed to create kernel");
	cout << "OpenCL init done" << endl;
	return true;
}

void cleanup() {
	cout << "Releasing all OpenCL objects" << endl;
	for(unsigned i = 0; i < num_kernels; ++i) {
		if(kernel && kernel[i]) {
			clReleaseKernel(kernel[i]);
		}
	}

	clReleaseMemObject(d_concat_buff);
	freeConvDevBuff(conv1);
	freePoolDevBuff(pool1);
	freeBatchNormDevBuff(norm1);
	freeConvDevBuff(conv2_1);
	freeConvDevBuff(conv2_2);
	freePoolDevBuff(pool2);
	freeBatchNormDevBuff(norm2);
	freeConvDevBuff(conv3);
	freeConvDevBuff(conv4_1);
	freeConvDevBuff(conv4_2);
	freeConvDevBuff(conv5_1);
	freeConvDevBuff(conv5_2);
	freePoolDevBuff(pool5);
	freeFcDevBuff(fc6);
	freeFcDevBuff(fc7);
	freeFcDevBuff(fc8);
	if(queue) {
		clReleaseCommandQueue(queue);
	}
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}

	full_model.destruct();
	delete[] norm1.scale;
	delete[] norm1.offset;
	delete[] norm2.scale;
	delete[] norm2.offset;
}
