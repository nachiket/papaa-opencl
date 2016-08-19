#include "device_utils.h"


void setKernelArgs(const ConvLayer &conv, const cl_kernel &kernel, const cl_mem &in_buff, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &in_buff);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &conv.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
		
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &conv.K);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &conv.stride);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &conv.bot_shape->z);
	checkError(status, "Failed to set argument %d", argi - 1);	

    global_ws[0] = conv.top_shape.x;
    global_ws[1] = conv.top_shape.y;
    global_ws[2] = conv.top_shape.z;
}
void setKernelArgs(const PoolLayer &pool, const cl_kernel &kernel, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), pool.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &pool.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &pool.bot_shape->x);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &pool.bot_shape->y);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &pool.winSize);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &pool.stride);
	checkError(status, "Failed to set argument %d", argi - 1);

    global_ws[0] = pool.top_shape.x;
    global_ws[1] = pool.top_shape.y;
    global_ws[2] = pool.top_shape.z;
}

void setKernelArgs(const FcLayer &fc, const cl_kernel &kernel, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;

	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), fc.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_W);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);

	unsigned int no_inputs = fc.bot_shape->x * fc.bot_shape->y * fc.bot_shape->z;
	
	status = clSetKernelArg(kernel, argi++, sizeof(unsigned int), &no_inputs);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &fc.d_b);
	checkError(status, "Failed to set argument %d", argi - 1);	

    global_ws[0] = fc.top_shape.x;
    global_ws[1] = fc.top_shape.y;
    global_ws[2] = fc.top_shape.z;
}

void setKernelArgs(const ActLayer &act, const cl_kernel &kernel, size_t *global_ws) {
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), act.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
    global_ws[0] = act.top_shape.x;
    global_ws[1] = act.top_shape.y;
    global_ws[2] = act.top_shape.z;
}

void setKernelArgs(const BatchNormLayer &bn, const cl_kernel &kernel, size_t *global_ws) {
	
	cl_int status;
	unsigned argi = 0;
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), bn.d_input);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bn.d_scale);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bn.d_offset);
	checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bn.d_output);
	checkError(status, "Failed to set argument %d", argi - 1);	

    global_ws[0] = bn.top_shape.x;
    global_ws[1] = bn.top_shape.y;
    global_ws[2] = bn.top_shape.z;
}

void allocConvDevBuff(cl_context &context, ConvLayer &conv) {
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

void allocFcDevBuff(cl_context &context, FcLayer &fc, cl_mem &prev_output) {
	cl_int status;
	fc.d_input = &prev_output;
	fc.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		fc.top_shape.x * fc.top_shape.y  * fc.top_shape.z * sizeof(DTYPE), NULL, &status);
	fc.d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.bot_shape->x * fc.bot_shape->y * fc.bot_shape->z * fc.top_shape.x * sizeof(WTYPE), fc.W, &status);
	fc.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		fc.top_shape.x * sizeof(WTYPE), fc.b, &status);
}

void allocPoolDevBuff(cl_context &context, PoolLayer &pool, cl_mem &prev_output) {
	cl_int status;
	pool.d_input = &prev_output;
	pool.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		pool.top_shape.x * pool.top_shape.y  * pool.top_shape.z * sizeof(DTYPE), NULL, &status);
}

void allocBatchNormDevBuff(cl_context &context, BatchNormLayer &bn, cl_mem &prev_output, unsigned int no_out) {
	cl_int status;
	bn.d_input = &prev_output;
	bn.d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA, 
		bn.top_shape.x * bn.top_shape.y  * bn.top_shape.z * sizeof(DTYPE), NULL, &status);
	bn.d_scale = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		no_out * sizeof(WTYPE), bn.scale, &status);
	bn.d_offset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA | CL_MEM_COPY_HOST_PTR, 
		no_out * sizeof(WTYPE), bn.offset, &status);
}

void allocConvHostBuff(ConvLayer &conv) {
	conv.h_input.reset((conv.bot_shape->x+2*conv.pad) * (conv.bot_shape->y+2*conv.pad) * conv.bot_shape->z);
	conv.h_output.reset(conv.top_shape.x * conv.top_shape.y * conv.top_shape.z);
}

void allocPoolHostBuff(PoolLayer &pool) {
	pool.h_input.reset((pool.bot_shape->x+2*pool.pad) * (pool.bot_shape->y+2*pool.pad) * pool.bot_shape->z);
	pool.h_output.reset(pool.top_shape.x * pool.top_shape.y * pool.top_shape.z);
}

void allocFcHostBuff(FcLayer &fc, scoped_aligned_ptr<DTYPE> &prev_h_output) {
	fc.h_input = &prev_h_output;
	fc.h_output.reset(fc.top_shape.x * fc.top_shape.y * fc.top_shape.z);
}

void allocNormHostBuff(BatchNormLayer &norm, scoped_aligned_ptr<DTYPE> &prev_h_output) {
	norm.h_input = &prev_h_output;
	norm.h_output.reset(norm.top_shape.x * norm.top_shape.y * norm.top_shape.z);
}
void freeConvDevBuff(const ConvLayer &conv) {

	clReleaseMemObject(conv.d_input);
	clReleaseMemObject(conv.d_output);
	clReleaseMemObject(conv.d_W);
	clReleaseMemObject(conv.d_b);
}

void freeFcDevBuff(const FcLayer &fc) {
	clReleaseMemObject(fc.d_output);
	clReleaseMemObject(fc.d_W);
	clReleaseMemObject(fc.d_b);
}

void freePoolDevBuff(const PoolLayer &pool) {
	clReleaseMemObject(pool.d_output);
}

void freeBatchNormDevBuff(const BatchNormLayer &norm) {
	clReleaseMemObject(norm.d_output);
	clReleaseMemObject(norm.d_scale);
	clReleaseMemObject(norm.d_offset);
}

