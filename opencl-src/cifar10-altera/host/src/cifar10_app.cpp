#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cifar10_model.h"
#include "cnn_structs.h"

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

scoped_array<DTYPE> h_input_img;
ConvLayer conv1;
PoolLayer pool1;
ActLayer relu1;
ConvLayer conv2;
ActLayer relu2;
PoolLayer pool2;
ConvLayer conv3;
ActLayer relu3;
PoolLayer pool3;
FcLayer fc1;
FcLayer fc2;
ActLayer smax;

void initModel(DataShape &inputShape);
void allocateHostBuffer();
void initInputImage(const Mat &img, int pad);
void createDeviceBuffer();
bool init_opencl();
void runApplication();
void cleanup();
void printNetShapes();

int main(int argc, char **argv) {
	Mat input_img;
	DataShape input_shape;

	cout << "CIFAR-10 Classification using Altera FPGA and OpenCL\n";
	std::string img_path = "";
	//FIXME: Need to input the image mean and subtract
	Options options(argc, argv);

	if(options.has("img")) {
		img_path = options.get<std::string>("img");
	}
	cout << img_path << endl;
	input_img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	if( ! input_img.data) {
		cout << "Failed to read the image. Specify correct image path" << endl;
		return -1;
	}
	input_shape.x = input_img.cols;
	input_shape.y = input_img.rows;
	input_shape.z = 3;
	cout << "Image resolution :" << input_img.cols << "x" << input_img.rows << "x" << input_img.channels() << endl;
	uint8_t *p_img = (uint8_t *)input_img.data;
	uint8_t r,g,b;
	unsigned int no_ch = input_img.channels();
	for(int row = 0; row < input_img.rows; row++) {
		for(int col = 0; col < input_img.cols; col++) {
			b = p_img[row*input_img.cols*no_ch + col*no_ch + 0];
			g = p_img[row*input_img.cols*no_ch + col*no_ch + 1];
			r = p_img[row*input_img.cols*no_ch + col*no_ch + 2];
			//printf("%d,", r);
		}
		//printf("\n");
	}
	// CNN model parameter init
	initModel(input_shape);
	printNetShapes();
	// Host buffer allocation
	allocateHostBuffer();
	// OpenCL context init
	init_opencl();
	// Allocate all necessary buffers on the device global memory.
	createDeviceBuffer();
	// This will normalize the image and transfer to the device memory
#if 0
	initInputImage(input_pgm);
	// Main application run
	runApplication();
	// Release OpenCL objects
#endif
	cleanup();
	return 0;
}
void setConvKernelArgs(const ConvLayer &conv, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), conv.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
		
	status = clSetKernelArg(kernel[0], argi++, sizeof(unsigned int), &conv.K);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(unsigned int), &conv.K);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(unsigned int), &conv.bot_shape->z);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);

    global_ws[0] = conv.top_shape.x;
    global_ws[1] = conv.top_shape.y;
    global_ws[2] = conv.top_shape.z;
}

void setPoolKernelArgs(const PoolLayer &pool, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), pool.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &pool.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[1], argi++, sizeof(unsigned int), &pool.bot_shape->x);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(unsigned int), &pool.bot_shape->y);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(unsigned int), &pool.winSize);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(unsigned int), &pool.stride);
	checkError(status, "Failed to set argument %d", argi - 1);

    global_ws[0] = pool.top_shape.x;
    global_ws[1] = pool.top_shape.y;
    global_ws[2] = pool.top_shape.z;
}

void setFcKernelArgs(const FcLayer &fc, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;

	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), fc.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &fc.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &fc.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);

	unsigned int no_inputs = fc.bot_shape->x * fc.bot_shape->y * fc.bot_shape->z;
	
	status = clSetKernelArg(kernel[2], argi++, sizeof(unsigned int), &no_inputs);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &fc.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);	

    global_ws[0] = fc.top_shape.x;
    global_ws[1] = fc.top_shape.y;
    global_ws[2] = fc.top_shape.z;
}

void setActKernelArgs(const ActLayer &act, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	switch(act.type) {
		case RELU:
			status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), act.d_input);
			break;
		case SOFTMAX:
			status = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), act.d_input);
			break;
	}
	checkError(status, "Failed to set argument %d", argi - 1);	
    global_ws[0] = act.top_shape.x;
    global_ws[1] = act.top_shape.y;
    global_ws[2] = act.top_shape.z;
}

#if 0
void runApplication() {
	cl_int status;
	size_t global_work_size[3];
	size_t local_work_size[3];
	scoped_array<cl_event> kernel_event(8);
	const double start_time = getCurrentTimestamp();

	setConvKernelArgs(conv1, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 0, NULL, &kernel_event[0]);
	checkError(status, "Failed to launch conv1 kernel");

	setPoolKernelArgs(pool1, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[1], 3, NULL, global_work_size, NULL, 1, &kernel_event[0], &kernel_event[1]);
	checkError(status, "Failed to launch pool1 kernel");

	setConvKernelArgs(conv2, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[0], 3, NULL, global_work_size, NULL, 1, &kernel_event[1], &kernel_event[2]);
	checkError(status, "Failed to launch conv2 kernel");

	setPoolKernelArgs(pool2, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[1], 3, NULL, global_work_size, NULL, 1, &kernel_event[2], &kernel_event[3]);
	checkError(status, "Failed to launch pool2 kernel");

	setFcKernelArgs(fc1, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[2], 3, NULL, global_work_size, NULL, 1, &kernel_event[3], &kernel_event[4]);
	checkError(status, "Failed to launch fc1 kernel");

	setActKernelArgs(relu1, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[3], 3, NULL, global_work_size, NULL, 1, &kernel_event[4], &kernel_event[5]);
	checkError(status, "Failed to launch relu1 kernel");

	setFcKernelArgs(fc2, global_work_size);
	status = clEnqueueNDRangeKernel(queue, kernel[2], 3, NULL, global_work_size, NULL, 1, &kernel_event[5], &kernel_event[6]);
	checkError(status, "Failed to launch fc2 kernel");

	setActKernelArgs(smax, global_work_size);
	local_work_size[0] = global_work_size[0];
	local_work_size[1] = global_work_size[1];
	local_work_size[2] = global_work_size[2];
	status = clEnqueueNDRangeKernel(queue, kernel[4], 3, NULL, global_work_size, local_work_size, 1, &kernel_event[6], &kernel_event[7]);
	checkError(status, "Failed to launch smax kernel");

	const double end_time = getCurrentTimestamp();
	const double total_time = end_time - start_time;

	cl_ulong time_ns = getStartEndTime(kernel_event, 8);
	printf("%ld\n", time_ns);
	cout << "Kernel time(ms)" << double(time_ns) * 1e-6 << endl;
	cout << "\nTime(ms): " << total_time * 1e3 << endl;

	// Read the final results
	cout << "Reading the output from the device" << endl;
	status = clEnqueueReadBuffer(queue, *smax.d_output, CL_TRUE, 0,
		smax.top_shape.x * smax.top_shape.y * smax.top_shape.z * sizeof(DTYPE), *smax.h_output, 0, NULL, NULL);
	checkError(status, "Failed to read data from the device");
	for(unsigned i = 0; i < 10; i++ ) {
		cout << fc2.h_output[i] << endl;
	}
}
#endif
void print_shape(const std::string name, const DataShape &shape) {
        cout << name << ":" << shape.z <<  "x" << shape.y << "x" << shape.x << endl;
}

void printNetShapes() {

	print_shape("conv1", conv1.top_shape);
	print_shape("pool1", pool1.top_shape);
	print_shape("conv2", conv2.top_shape);
	print_shape("pool2", pool2.top_shape);
	print_shape("conv3", conv3.top_shape);
	print_shape("pool3", pool3.top_shape);
	print_shape("fc1", fc1.top_shape);
	print_shape("fc2", fc2.top_shape);
}

void initModel(DataShape &inputShape) {
	cout << "CNN model initialization\n";
	conv1.bot_shape = &inputShape;
	conv1.K = CONV1_FILTER_WIDTH;
	conv1.pad = 2;
	conv1.W = (WTYPE *)conv1_weights;
	conv1.b = (WTYPE *)conv1_bias;
	conv1.stride = 1;
	conv1.top_shape.z = CONV1_NO_OUTPUTS;
	conv1.top_shape.x = conv1.bot_shape->x - conv1.K + 1 + 2*conv1.pad;
	conv1.top_shape.y = conv1.bot_shape->y - conv1.K + 1 + 2*conv1.pad;

	pool1.bot_shape = &conv1.top_shape;
	pool1.type = MAX;
	pool1.stride = 2;
	pool1.winSize = 3;
	pool1.pad = 0;
	pool1.top_shape.x = ceil((pool1.bot_shape->x - pool1.winSize)/(float)pool1.stride) + 1;
	pool1.top_shape.y = ceil((pool1.bot_shape->y - pool1.winSize)/(float)pool1.stride) + 1;
	pool1.top_shape.z = pool1.bot_shape->z;

	relu1.bot_shape = &pool1.top_shape;
	relu1.type = RELU;
	relu1.top_shape.x = relu1.bot_shape->x;
	relu1.top_shape.y = relu1.bot_shape->y;
	relu1.top_shape.z = relu1.bot_shape->z;

	conv2.bot_shape = &relu1.top_shape;
	conv2.K = CONV2_FILTER_WIDTH;
	conv2.pad = 2;
	conv2.W = (WTYPE *)conv2_weights;
	conv2.b = (WTYPE *)conv2_bias;
	conv2.stride = 1;
	conv2.top_shape.z = CONV2_NO_OUTPUTS;
	conv2.top_shape.x = conv2.bot_shape->x - conv2.K + 1 + 2*conv2.pad;
	conv2.top_shape.y = conv2.bot_shape->y - conv2.K + 1 + 2*conv2.pad;

	relu2.bot_shape = &conv2.top_shape;
	relu2.type = RELU;
	relu2.top_shape.x = relu2.bot_shape->x;
	relu2.top_shape.y = relu2.bot_shape->y;
	relu2.top_shape.z = relu2.bot_shape->z;

	pool2.bot_shape = &relu2.top_shape;
	pool2.type = MAX;
	pool2.stride = 2;
	pool2.winSize = 3;
	pool2.pad = 0;
	pool2.top_shape.x = ceil((pool2.bot_shape->x - pool2.winSize)/(float)pool2.stride) + 1;
	pool2.top_shape.y = ceil((pool2.bot_shape->y - pool2.winSize)/(float)pool2.stride) + 1;
	pool2.top_shape.z = pool2.bot_shape->z;

	conv3.bot_shape = &pool2.top_shape;
	conv3.K = CONV3_FILTER_WIDTH;
	conv3.pad = 2;
	conv3.W = (WTYPE *)conv3_weights;
	conv3.b = (WTYPE *)conv3_bias;
	conv3.stride = 1;
	conv3.top_shape.z = CONV3_NO_OUTPUTS;
	conv3.top_shape.x = conv3.bot_shape->x - conv3.K + 1 + 2*conv3.pad;
	conv3.top_shape.y = conv3.bot_shape->y - conv3.K + 1 + 2*conv3.pad;

	relu3.bot_shape = &conv3.top_shape;
	relu3.type = RELU;
	relu3.top_shape.x = relu3.bot_shape->x;
	relu3.top_shape.y = relu3.bot_shape->y;
	relu3.top_shape.z = relu3.bot_shape->z;

	pool3.bot_shape = &relu3.top_shape;
	pool3.type = MAX;
	pool3.stride = 2;
	pool3.winSize = 3;
	pool3.pad = 0;
	pool3.top_shape.x = ceil((pool3.bot_shape->x - pool3.winSize)/(float)pool3.stride) + 1;
	pool3.top_shape.y = ceil((pool3.bot_shape->y - pool3.winSize)/(float)pool3.stride) + 1;
	pool3.top_shape.z = pool3.bot_shape->z;

	fc1.bot_shape = &pool3.top_shape;
	fc1.W = (WTYPE *)ip1_weights;
	fc1.b = (WTYPE *)ip1_bias;
	fc1.no_units = IP1_NO_OUTPUTS;
	fc1.top_shape.z = 1;
	fc1.top_shape.y = 1;
	fc1.top_shape.x = fc1.no_units;

	
	fc2.bot_shape = &fc1.top_shape;
	fc2.W = (WTYPE *)ip2_weights;
	fc2.b = (WTYPE *)ip2_bias;
	fc2.no_units = IP2_NO_OUTPUTS;
	fc2.top_shape.z = 1;
	fc2.top_shape.y = 1;
	fc2.top_shape.x = fc2.no_units;

	smax.bot_shape = &fc2.top_shape;
	smax.type = SOFTMAX;
	smax.top_shape.x = smax.bot_shape->x;
	smax.top_shape.y = smax.bot_shape->y;
	smax.top_shape.z = smax.bot_shape->z;

}
void allocateHostBuffer() {
	cout << "Allocating host memory for inputs and outputs\n";
	h_input_img.reset(conv1.bot_shape->x * conv1.bot_shape->y * conv1.bot_shape->z);
	conv1.h_input.reset((conv1.bot_shape->x+2*conv1.pad) * (conv1.bot_shape->y+2*conv1.pad) * conv1.bot_shape->z);
	conv1.h_output.reset(conv1.top_shape.x * conv1.top_shape.y * conv1.top_shape.z);

	pool1.h_input.reset((pool1.bot_shape->x+2*pool1.pad) * (pool1.bot_shape->y+2*pool1.pad) * pool1.bot_shape->z);
	pool1.h_output.reset(pool1.top_shape.x * pool1.top_shape.y * pool1.top_shape.z);

	relu1.h_input = &pool1.h_output;
	relu1.h_output = relu1.h_input;

	conv2.h_input.reset((conv2.bot_shape->x+2*conv2.pad) * (conv2.bot_shape->y+2*conv2.pad) * conv2.bot_shape->z);;
	conv2.h_output.reset(conv2.top_shape.x * conv2.top_shape.y * conv2.top_shape.z);

	relu2.h_input = &conv2.h_output;
	relu2.h_output = relu2.h_input;

	pool2.h_input.reset((pool2.bot_shape->x+2*pool2.pad) * (pool2.bot_shape->y+2*pool2.pad) * pool2.bot_shape->z);
	pool2.h_output.reset(pool2.top_shape.x * pool2.top_shape.y * pool2.top_shape.z);

	conv3.h_input.reset((conv3.bot_shape->x+2*conv3.pad) * (conv3.bot_shape->y+2*conv3.pad) * conv3.bot_shape->z);;
	conv3.h_output.reset(conv3.top_shape.x * conv3.top_shape.y * conv3.top_shape.z);

	relu3.h_input = &conv3.h_output;
	relu3.h_output = relu3.h_input;

	pool3.h_input.reset((pool3.bot_shape->x+2*pool3.pad) * (pool3.bot_shape->y+2*pool3.pad) * pool3.bot_shape->z);
	pool3.h_output.reset(pool3.top_shape.x * pool3.top_shape.y * pool3.top_shape.z);

	fc1.h_input = &pool3.h_output;
	fc1.h_output.reset(fc1.top_shape.x * fc1.top_shape.y * fc1.top_shape.z);

	fc2.h_input = &fc1.h_output;
	fc2.h_output.reset(fc2.top_shape.x * fc2.top_shape.y * fc2.top_shape.z);
	
	smax.h_input = &fc2.h_output;
	smax.h_output = smax.h_input;
}

void initInputImage(const Mat &img, int pad) {
	cout << "Normalizing the data and transferring to device memory" << endl;
	cl_int status;
#if 0
	// FIXME: handle color images
	// read image from pgm structure and normalize, populate into host buffer
	for(unsigned p = 0; p < input_pgm.height * input_pgm.width; p++) {
		h_input_img[p] = input_pgm.buf[p] / 255.0;
	}
	cout << "Transferring image to device memory" << endl;
	status = clEnqueueWriteBuffer(queue, d_input_img, CL_FALSE,
        0, conv1.bot_shape->x * conv1.bot_shape->y * conv1.bot_shape->z * sizeof(DTYPE), h_input_img, 0, NULL, NULL);
	checkError(status, "Failed to transfer input image to the device\n");
	clFinish(queue);
#endif
}

void zeropadAndTx(const scoped_array<DTYPE> &src, scoped_array<DTYPE> &dst, int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff) {
	unsigned dst_h = src_h + 2*pad_h;
	unsigned dst_w = src_w + 2*pad_w;
	cl_int status;
	for(int ch = 0; ch < n_ch; ch++) {
		for(int r = 0; r < src_h + 2*pad_h; r++) {
			for(int c = 0; c < src_w + 2*pad_w; c++) {
				if(r < pad_h || (r > src_h+pad_h-1) || c < pad_w || (c > src_w+pad_w-1)) {
					dst[ch*dst_h*dst_w + r*dst_w + c] = (DTYPE)0;
				}else {
					// TODO: use memcpy instead to do bulk copy of one full row
					dst[ch*dst_h*dst_w + r*dst_w + c] = src[ch*src_h*src_w + (r-pad_h)*src_w + c-pad_w];
				}
			}	
		}
	}
	status = clEnqueueWriteBuffer(queue, device_buff, CL_FALSE, 0,
		n_ch * dst_h * dst_w * sizeof(DTYPE), dst, 0, NULL, NULL);
	checkError(status, "Failed to transfer data to the device\n");
	clFinish(queue);
}

void allocateConvDeviceBuff(ConvLayer &conv) {
	cl_int status;
	// data is allocated in BANK1 and weights are in BANK2 for efficient access.
	conv.d_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
                (conv.bot_shape->x+2*conv.pad) * (conv.bot_shape->y+2*conv.pad) * conv.bot_shape->z * sizeof(DTYPE), NULL, &status);
	conv.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		conv.top_shape.x * conv.top_shape.y  * conv.top_shape.z * sizeof(DTYPE), NULL, &status);
	
	conv.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv.K * conv.K * conv.bot_shape->z * conv.top_shape.z * sizeof(WTYPE), conv.W, &status);
	conv.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv.top_shape.z * sizeof(WTYPE), conv.b, &status);
}

void allocateFcDeviceBuff(FcLayer &fc) {
	cl_int status;
	fc.d_input = &pool3.d_output;
	fc.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		fc.top_shape.x * fc.top_shape.y  * fc.top_shape.z * sizeof(DTYPE), NULL, &status);
	fc.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.bot_shape->x * fc.bot_shape->y * fc.bot_shape->z * fc.top_shape.x * sizeof(WTYPE), fc.W, &status);
	fc.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.top_shape.x * sizeof(WTYPE), fc.b, &status);
}

void createDeviceBuffer() {
	cl_int status;
	cout << "Allocating buffers on the device memory" << endl;
	allocateConvDeviceBuff(conv1);

	pool1.d_input = &conv1.d_output;
	pool1.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool1.top_shape.x * pool1.top_shape.y  * pool1.top_shape.z * sizeof(DTYPE), NULL, &status);
	// ActLayer performs in-place ops. No need of output buffer.
	relu1.d_input = &pool1.d_output;
	relu1.d_output = relu1.d_input;

	allocateConvDeviceBuff(conv2);

	relu2.d_input = &conv2.d_output;
	relu2.d_output = relu2.d_input;

	pool2.d_input = relu2.d_output;
	pool2.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool2.top_shape.x * pool2.top_shape.y  * pool2.top_shape.z * sizeof(DTYPE), NULL, &status);

	allocateConvDeviceBuff(conv3);
	relu3.d_input = &conv3.d_output;
	relu3.d_output = relu3.d_input;
	pool3.d_input = relu3.d_output;
	pool3.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool3.top_shape.x * pool3.top_shape.y  * pool3.top_shape.z * sizeof(DTYPE), NULL, &status);

	allocateFcDeviceBuff(fc1);
	allocateFcDeviceBuff(fc2);
	
	smax.d_input = &fc2.d_output;
	smax.d_output = smax.d_input;
    	checkError(status, "Failed to create buffers");
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
	
	std::string binary_file = getBoardBinaryFile("cnn_kernels", target_device);
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
	kernel[0] = clCreateKernel(program, "filter3D", &status);
	checkError(status, "Failed to create kernel");
	kernel[1] = clCreateKernel(program, "maxpool3D", &status);
	checkError(status, "Failed to create kernel");
	kernel[2] = clCreateKernel(program, "iplayer", &status);
	checkError(status, "Failed to create kernel");
	kernel[3] = clCreateKernel(program, "relu_layer", &status);
	checkError(status, "Failed to create kernel");
	kernel[4] = clCreateKernel(program, "softmax", &status);
	checkError(status, "Failed to create kernel");

	return true;
}

void freeConvDeviceBuff(const ConvLayer &conv) {

	clReleaseMemObject(conv.d_input);
	clReleaseMemObject(conv.d_output);
	clReleaseMemObject(conv.d_W);
	clReleaseMemObject(conv.d_b);
}

void freeFcDeviceBuff(const FcLayer &fc) {
	clReleaseMemObject(fc.d_output);
	clReleaseMemObject(fc.d_W);
	clReleaseMemObject(fc.d_b);
}
void cleanup() {
	cout << "Releasing all OpenCL objects" << endl;
	for(unsigned i = 0; i < num_kernels; ++i) {
		if(kernel && kernel[i]) {
			clReleaseKernel(kernel[i]);
		}
	}
	freeConvDeviceBuff(conv1);
	clReleaseMemObject(pool1.d_output);
	freeConvDeviceBuff(conv2);
	clReleaseMemObject(pool2.d_output);
	freeConvDeviceBuff(conv3);
	clReleaseMemObject(pool3.d_output);
	freeFcDeviceBuff(fc1);
	freeFcDeviceBuff(fc2);
	if(queue) {
		clReleaseCommandQueue(queue);
	}

	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
}
