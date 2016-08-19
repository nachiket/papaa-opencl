#ifdef _DEVICE_UTILS_H_
#define _DEVICE_UTILS_H_
#include "cnn_structs.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

void setKernelArgs(const ConvLayer &conv, const cl_kernel &kernel, const cl_mem &in_buff, size_t *global_ws);

void setKernelArgs(const PoolLayer &pool, const cl_kernel &kernel, size_t *global_ws);

void setKernelArgs(const FcLayer &fc, const cl_kernel &kernel, size_t *global_ws);

void setKernelArgs(const ActLayer &act, const cl_kernel &kernel, size_t *global_ws);

void setKernelArgs(const BatchNormLayer &bn, const cl_kernel &kernel, size_t *global_ws);

void allocConvDevBuff(ConvLayer &conv);

void allocFcDevBuff(FcLayer &fc, cl_mem &prev_output);

void allocPoolDevBuff(PoolLayer &pool, cl_mem &prev_output);

void allocBatchNormDevBuff(BatchNormLayer &bn, cl_mem &prev_output, unsigned int no_out);

void allocConvHostBuff(ConvLayer &conv);

void allocPoolHostBuff(PoolLayer &pool);

void allocFcHostBuff(FcLayer &fc, scoped_aligned_ptr<DTYPE> &prev_h_output);

void allocNormHostBuff(BatchNormLayer &norm, scoped_aligned_ptr<DTYPE> &prev_h_output);

void freeConvDevBuff(const ConvLayer &conv);

void freeFcDevBuff(const FcLayer &fc);

void freePoolDevBuff(const PoolLayer &pool);

void freeBatchNormDevBuff(const BatchNormLayer &norm);
#endif// _KERNEL_UTILS_H_
