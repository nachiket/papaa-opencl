#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"
#include "device_utils.h"

using namespace aocl_utils;
using namespace std;
using namespace cv;

cl_platform_id platform = NULL;
unsigned num_devices = 0;
unsigned num_kernels = 6;
scoped_array<cl_device_id> devices;
cl_device_id target_device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
scoped_array<cl_kernel> kernel;

scoped_aligned_ptr<DTYPE> h_input_img;
ConvLayer conv1;
ActLayer relu1;
PoolLayer pool1;
BatchNormLayer norm1;

ConvLayer conv2_1;
ConvLayer conv2_2;
ActLayer relu2;
PoolLayer pool2;
BatchNormLayer norm2;

ConvLayer conv3;
ActLayer relu3;
ConvLayer conv4_1;
ConvLayer conv4_2;
ActLayer relu4;

ConvLayer conv5_1;
ConvLayer conv5_2;
ActLayer relu5;
PoolLayer pool5;

FcLayer fc6;
ActLayer relu6;
FcLayer fc7;
ActLayer relu7;
FcLayer fc8;
ActLayer smax;

// Buffer to read output of grouped conv layers and concatenate. The size of these
// will be max requirement across whole network.
cl_mem d_concat_buff;
scoped_aligned_ptr<DTYPE> h_concat_buff;

void allocateHostBuffer();
void allocateDeviceBuffer();

void printHelp(char *app) {
	cout << "------------------------------------" << endl;
	cout << "Usage:" << endl;
	cout << app;
	cout << " -m <mode>";
	cout << " [-i] <image path>";
	cout << " [-l] <image list>";
	cout << " [-d] <test image dir>" << endl;
	cout << "------------------------------------" << endl;
}

int main(int argc, char **argv) {
	cout << "ImageNet object classification using AlexNet" << endl;

	printHelp(argv[0]);
	return 0;
}

void allocateHostBuffer() {
	cout << "Allocating host memory for inputs and outputs\n";
	h_input_img.reset(conv1.bot_shape->x * conv1.bot_shape->y * conv1.bot_shape->z);
	//TODO: Allocate h_concat_buff
	allocateConvHosticeBuff(conv1);
	// ActLayer performs in-place ops. No need of output buffer.
	relu1.h_input = &conv1.h_output;
	relu1.h_output = relu1.h_input;
	allocPoolHostBuff(pool1);
	allocBatchNormHostBuff(norm1, pool1.h_output);

	allocateConvHostBuff(conv2_1);
	allocateConvHostBuff(conv2_2);
	relu2.h_input = &h_concat_buff;
	relu2.h_output = relu2.h_input;
	allocPoolHostBuff(pool2);
	allocBatchNormHostBuff(norm2, pool2.h_output);

	allocateConvHostBuff(conv3);
	relu3.h_input = &conv3.h_output;
	relu3.h_output = relu3.h_input;

	allocateConvHostBuff(conv4_1);
	allocateConvHostBuff(conv4_2);
	relu4.h_input = &h_concat_buff;
	relu4.h_output = relu4.h_input;

	allocateConvHostBuff(conv5_1);
	allocateConvHostBuff(conv5_2);
	relu5.h_input = &h_concat_buff;
	relu5.h_output = relu5.h_input;
	allocPoolHostBuff(pool5);

	allocFcHostBuff(fc6, pool5.h_output);
	relu6.h_input = &fc6.h_output;
	relu6.h_output = relu6.h_input;
	allocFcHostBuff(fc7, *relu6.h_output);
	relu7.h_input = &fc7.h_output;
	relu7.h_output = relu7.h_input;
	allocFcDevBuff(fc8, *relu7.h_output);

	smax.h_input = &fc8.h_output;
	smax.h_output = smax.h_input;
}


void allocateDeviceBuffer() {

	cl_int status;
	cout << "Allocating device memory for intermediate data and model params." << endl;
	// TODO: Allocate d_concat_buff
	allocateConvDeviceBuff(conv1);
	// ActLayer performs in-place ops. No need of output buffer.
	relu1.d_input = &conv1.d_output;
	relu1.d_output = relu1.d_input;
	allocPoolDevBuff(pool1, *relu1.d_output);
	allocBatchNormDevBuff(norm1, pool1.d_output, norm1.top_shape.z);

	allocateConvDevBuff(conv2_1);
	allocateConvDevBuff(conv2_2);
	relu2.d_input = &d_concat_buff;
	relu2.d_output = relu2.d_input;
	allocPoolDevBuff(pool2, *relu2.d_output);
	allocBatchNormDevBuff(norm2, pool2.d_output, norm1.top_shape.z);

	allocateConvDevBuff(conv3);
	relu3.d_input = &conv3.d_output;
	relu3.d_output = relu3.d_input;

	allocateConvDevBuff(conv4_1);
	allocateConvDevBuff(conv4_2);
	relu4.d_input = &d_concat_buff;
	relu4.d_output = relu4.d_input;

	allocateConvDevBuff(conv5_1);
	allocateConvDevBuff(conv5_2);
	relu5.d_input = &d_concat_buff;
	relu5.d_output = relu5.d_input;
	allocPoolDevBuff(pool5, *relu5.d_output);

	allocFcDevBuff(fc6, pool5.d_output);
	relu6.d_input = &fc6.d_output;
	relu6.d_output = relu6.d_input;
	allocFcDevBuff(fc7, *relu6.d_output);
	relu7.d_input = &fc7.d_output;
	relu7.d_output = relu7.d_input;
	allocFcDevBuff(fc8, *relu7.d_output);

	smax.d_input = &fc8.d_output;
	smax.d_output = smax.d_input;
}
void cleanup() {
	cout << "Releasing all OpenCL objects" << endl;
}
