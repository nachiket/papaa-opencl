#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "cnn_structs.h"
#include "pgm.h"
#include "lenet5_model.h"

using namespace aocl_utils;
using namespace std;

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
ConvLayer conv2;
PoolLayer pool2;
FcLayer fc1;
ActLayer relu1;
FcLayer fc2;
ActLayer smax;
cl_mem d_input_img;

void initModel(DataShape &inputShape);
void allocateHostBuffer();
void initInputImage(const pgm_t &input_pgm);
void createDeviceBuffer();
bool init_opencl();
unsigned int runApplication();
void cleanup();

int main(int argc, char **argv) {
	pgm_t input_pgm;
	DataShape input_shape;

	cout << "MNSIT Digit Classification using Altera FPGA and OpenCL\n";
	string img_path, mode;
	string list_file, img_dir;
	unsigned int no_samples, pred_digit;
	
	Options options(argc, argv);

	if(!options.has("mode")) {
		cout << "Please specify the application mode(sample OR test)" << endl;
		exit(1);
	} else {
		mode = options.get<string>("mode");
	}
	if(mode.compare("sample") == 0) {
		if(options.has("img")) {
			img_path = options.get<std::string>("img");
		} else {
			cout << "Please specify image path" << endl;
			exit(1);
		}
	} else if(mode.compare("test") == 0) {
		if(options.has("list") && options.has("dir") && options.has("n")) {
			list_file = options.get<string>("list");
			img_dir = options.get<string>("dir");
			no_samples = options.get<unsigned int>("n");
		} else {
			cout << "Need image list file, image directory and no of samples to test" << endl;
			exit(1);
		}
	} else {
		cout << "Invalid application mode(valid ones are <sample,test>)" << endl;
		exit(1);
	}

	input_shape.x = 28;//input_pgm.width;
	input_shape.y = 28;//input_pgm.height;
	input_shape.z = 1;

	// CNN model parameter init
	initModel(input_shape);
	// Host buffer allocation
	allocateHostBuffer();
	// OpenCL context init
	init_opencl();
	// Allocate all necessary buffers on the device global memory.
	createDeviceBuffer();
	if(mode.compare("sample") == 0) {
		// Sample test mode
		cout << img_path << endl;
		if(-1 == readPGM(&input_pgm, img_path.c_str())) {
			cout << "Failed to read the image. Specify correct image path" << endl;
			exit(1);
		}
		// This will normalize the image and transfer to the device memory
		initInputImage(input_pgm);
		// Main application run
		pred_digit = runApplication();
		cout << "Predicted digit = " << pred_digit << endl;
		// Release image buffer
		destroyPGM(&input_pgm);
	} else {
		cout << "Full test mode" << endl;
		std::ifstream list;
		std::vector<std::string> test_list;
		std::vector<unsigned int> target_labels;
		std::string csv_line, img_file, label;

		list.open(list_file.c_str());
		while(std::getline(list, csv_line)) {
			std::stringstream ss(csv_line);
			std::getline(ss, img_file, ',');
			std::getline(ss, label, ',');
			test_list.push_back(img_file);
			target_labels.push_back(atoi(label.c_str()));
		}
		unsigned int testset_size = target_labels.size();
		if(no_samples < 0 || no_samples > testset_size) {
			no_samples = testset_size;
		}
		cout << "No of samples under test = " << no_samples << endl;
		unsigned int mis_count = 0;	
		for(int img = 0; img < no_samples; img++) {
			img_file = img_dir + "/" + test_list[img];
			cout << img_file << endl;
			if(-1 == readPGM(&input_pgm, img_file.c_str())) {
				cout << "Failed to read image" << endl << img_file << endl;
			}
			// This will normalize the image and transfer to the device memory
			initInputImage(input_pgm);
			// Main application run
			pred_digit = runApplication();
			cout << "Actual = " << target_labels[img] << "Pred = " << pred_digit << endl;
			if(pred_digit != target_labels[img]) {
				mis_count++;
			}
			// Release image buffer
			destroyPGM(&input_pgm);
		}
		sleep(10);
		cout << "No images misclassified = " << mis_count << endl;
		cout << "Classification Error = " << float(mis_count)/no_samples << endl;
	}
	// Release OpenCL objects
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

unsigned int runApplication() {
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
	//printf("%ld\n", time_ns);
	//cout << "Kernel time(ms)" << double(time_ns) * 1e-6 << endl;
	//cout << "\nTime(ms): " << total_time * 1e3 << endl;

	// Read the final results
	//cout << "Reading the output from the device" << endl;
	status = clEnqueueReadBuffer(queue, *smax.d_output, CL_TRUE, 0,
		smax.top_shape.x * smax.top_shape.y * smax.top_shape.z * sizeof(DTYPE), *smax.h_output, 0, NULL, NULL);
	checkError(status, "Failed to read data from the device");
	/*for(unsigned i = 0; i < 10; i++ ) {
		cout << fc2.h_output[i] << endl;
	}*/
	DTYPE max_prob = 0.0;
	unsigned int pred = 10;
	for(unsigned p = 0; p < 10; p++) {
		if(fc2.h_output[p] > max_prob) {
			max_prob = fc2.h_output[p];
			pred = p;
		}
	}
	return pred;
}

void initModel(DataShape &inputShape) {
	cout << "CNN model initialization\n";
	conv1.bot_shape = &inputShape;
	conv1.K = CONV1_FILTER_WIDTH; // assume height == width for the filter
	conv1.W = (WTYPE *)conv1_weights;
	conv1.b = (WTYPE *)conv1_bias;
	conv1.stride = 1;
	conv1.top_shape.z = CONV1_NO_OUTPUTS;
	conv1.top_shape.x = conv1.bot_shape->x - conv1.K + 1;
	conv1.top_shape.y = conv1.bot_shape->y - conv1.K + 1;

	pool1.bot_shape = &conv1.top_shape;
	pool1.type = MAX;
	pool1.stride = 2;
	pool1.winSize = 2;
	pool1.top_shape.x = ((pool1.bot_shape->x - pool1.winSize)/pool1.stride) + 1;
	pool1.top_shape.y = ((pool1.bot_shape->y - pool1.winSize)/pool1.stride) + 1;
	pool1.top_shape.z = pool1.bot_shape->z;

	conv2.bot_shape = &pool1.top_shape;
	conv2.K = CONV2_FILTER_WIDTH; // assume height == width for the filter
	conv2.W = (WTYPE *)conv2_weights;
	conv2.b = (WTYPE *)conv2_bias;
	conv2.stride = 1;
	conv2.top_shape.z = CONV2_NO_OUTPUTS;
	conv2.top_shape.x = conv2.bot_shape->x - conv2.K + 1;
	conv2.top_shape.y = conv2.bot_shape->y - conv2.K + 1;

	pool2.bot_shape = &conv2.top_shape;
	pool2.type = MAX;
	pool2.stride = 2;
	pool2.winSize = 2;
	pool2.top_shape.x = ((pool2.bot_shape->x - pool2.winSize)/pool2.stride) + 1;
	pool2.top_shape.y = ((pool2.bot_shape->y - pool2.winSize)/pool2.stride) + 1;
	pool2.top_shape.z = pool2.bot_shape->z;

	fc1.bot_shape = &pool2.top_shape;
	fc1.W = (WTYPE *)ip1_weights;
	fc1.b = (WTYPE *)ip1_bias;
	fc1.no_units = IP1_NO_OUTPUTS;
	fc1.top_shape.z = 1;
	fc1.top_shape.y = 1;
	fc1.top_shape.x = fc1.no_units;

	relu1.bot_shape = &fc1.top_shape;
	relu1.type = RELU;
	relu1.top_shape.x = relu1.bot_shape->x;
	relu1.top_shape.y = relu1.bot_shape->y;
	relu1.top_shape.z = relu1.bot_shape->z;
	
	fc2.bot_shape = &relu1.top_shape;
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
	conv1.h_input = &h_input_img;
	conv1.h_output.reset(conv1.top_shape.x * conv1.top_shape.y * conv1.top_shape.z);

	pool1.h_input = &conv1.h_output;
	pool1.h_output.reset(pool1.top_shape.x * pool1.top_shape.y * pool1.top_shape.z);

	conv2.h_input = &pool1.h_output;
	conv2.h_output.reset(conv2.top_shape.x * conv2.top_shape.y * conv2.top_shape.z);

	pool2.h_input = &conv2.h_output;
	pool2.h_output.reset(pool2.top_shape.x * pool2.top_shape.y * pool2.top_shape.z);

	fc1.h_input = &pool2.h_output;
	fc1.h_output.reset(fc1.top_shape.x * fc1.top_shape.y * fc1.top_shape.z);

	relu1.h_input = &fc1.h_output;
	relu1.h_output = relu1.h_input;

	fc2.h_input = relu1.h_output;
	fc2.h_output.reset(fc2.top_shape.x * fc2.top_shape.y * fc2.top_shape.z);
	
	smax.h_input = &fc2.h_output;
	smax.h_output = smax.h_input;
}

void initInputImage(const pgm_t &input_pgm) {
	cout << "Normalizing the data and transferring to device memory" << endl;
	cl_int status;
	// FIXME: handle color images
	// read image from pgm structure and normalize, populate into host buffer
	for(unsigned p = 0; p < input_pgm.height * input_pgm.width; p++) {
		h_input_img[p] = input_pgm.buf[p] / 255.0;
	}
	cout << "Transferring image to device memory" << endl;
	status = clEnqueueWriteBuffer(queue, d_input_img, CL_TRUE,
        0, conv1.bot_shape->x * conv1.bot_shape->y * conv1.bot_shape->z * sizeof(DTYPE), h_input_img, 0, NULL, NULL);
	checkError(status, "Failed to transfer input image to the device\n");
//	clFinish(queue);
}

void createDeviceBuffer() {
	cl_int status;
	cout << "Allocating buffers on the device memory" << endl;
	// FIXME: Used CL_MEM_COPY_HOST_PTR to copy the model weights and biases to device memory. Not sure if this works for FPGAs
	// data is allocated in BANK1 and weights are in BANK2 for efficient access.
	d_input_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
		conv1.bot_shape->x * conv1.bot_shape->y  * conv1.bot_shape->z * sizeof(DTYPE), NULL, &status);
	conv1.d_input = &d_input_img;
	conv1.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		conv1.top_shape.x * conv1.top_shape.y  * conv1.top_shape.z * sizeof(DTYPE), NULL, &status);
	
	conv1.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv1.K * conv1.K * conv1.bot_shape->z * conv1.top_shape.z * sizeof(WTYPE), conv1.W, &status);
	conv1.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv1.top_shape.z * sizeof(WTYPE), conv1.b, &status);
	pool1.d_input = &conv1.d_output;
	pool1.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool1.top_shape.x * pool1.top_shape.y  * pool1.top_shape.z * sizeof(DTYPE), NULL, &status);
	conv2.d_input = &pool1.d_output;
	conv2.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		conv2.top_shape.x * conv2.top_shape.y  * conv2.top_shape.z * sizeof(DTYPE), NULL, &status);
	
	conv2.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv2.K * conv2.K * conv2.bot_shape->z * conv2.top_shape.z * sizeof(WTYPE), conv2.W, &status);
	conv2.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		conv2.top_shape.z * sizeof(WTYPE), conv2.b, &status);
	pool2.d_input = &conv2.d_output;
	pool2.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool2.top_shape.x * pool2.top_shape.y  * pool2.top_shape.z * sizeof(DTYPE), NULL, &status);

	fc1.d_input = &pool2.d_output;
	fc1.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		fc1.top_shape.x * fc1.top_shape.y  * fc1.top_shape.z * sizeof(DTYPE), NULL, &status);
	fc1.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc1.bot_shape->x * fc1.bot_shape->y * fc1.bot_shape->z * fc1.top_shape.x * sizeof(WTYPE), fc1.W, &status);
	fc1.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc1.top_shape.x * sizeof(WTYPE), fc1.b, &status);
	// ActLayer performs in-place ops. No need of output buffer.
	relu1.d_input = &fc1.d_output;
	relu1.d_output = relu1.d_input;
	
	fc2.d_input = relu1.d_output;
	fc2.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		fc2.top_shape.x * fc2.top_shape.y  * fc2.top_shape.z * sizeof(DTYPE), NULL, &status);
	fc2.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc2.top_shape.x * fc2.bot_shape->x * sizeof(WTYPE), fc2.W, &status);
	fc2.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc2.top_shape.x * sizeof(WTYPE), fc2.b, &status);
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
	
	std::string binary_file = getBoardBinaryFile("cnn_kernels.aocx", target_device);
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

void cleanup() {
	cout << "Releasing all OpenCL objects" << endl;
	for(unsigned i = 0; i < num_kernels; ++i) {
		if(kernel && kernel[i]) {
			clReleaseKernel(kernel[i]);
		}
	}
	clReleaseMemObject(d_input_img);
	clReleaseMemObject(conv1.d_output);
	clReleaseMemObject(conv1.d_W);
	clReleaseMemObject(conv1.d_b);
	clReleaseMemObject(pool1.d_output);
	clReleaseMemObject(conv2.d_output);
	clReleaseMemObject(conv2.d_W);
	clReleaseMemObject(conv1.d_b);
	clReleaseMemObject(pool2.d_output);
	clReleaseMemObject(fc1.d_output);
	clReleaseMemObject(fc1.d_W);
	clReleaseMemObject(fc1.d_b);
	clReleaseMemObject(fc2.d_output);
	clReleaseMemObject(fc2.d_W);
	clReleaseMemObject(fc2.d_b);
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
