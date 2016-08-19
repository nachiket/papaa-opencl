#ifndef CNN_STRUCTS_H
#define CNN_STRUCTS_H
#include <stdio.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;
// data type
typedef float DTYPE;
// weight and bias type
typedef float WTYPE;

typedef enum {
	MAX,
	AVG
} PoolType;

typedef enum {
	RELU,
	TANH,
	SIGMOID,
	SOFTMAX
} ActType;

typedef struct {
	unsigned int x;
	unsigned int y;
	unsigned int z;
} DataShape;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape;
	unsigned int K;
	unsigned int stride;
	unsigned int pad;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> h_input;
	cl_mem d_output;
	cl_mem d_input;
	WTYPE *W;
	WTYPE *b;
	cl_mem d_W;
	cl_mem d_b;
} ConvLayer;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape;
	unsigned int winSize;
	unsigned int stride;
	unsigned int pad;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> h_input;
	cl_mem d_output;
	cl_mem *d_input;
	PoolType type;
} PoolLayer;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape;
	unsigned int no_units;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem d_output;
	cl_mem *d_input;
	WTYPE *W;
	WTYPE *b;
	cl_mem d_W;
	cl_mem d_b;
} FcLayer;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape;
	// ActLayer ops are in-place. No need to allocate separate buffer
	scoped_aligned_ptr<DTYPE> *h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem *d_output;
	cl_mem *d_input;
	ActType type;
} ActLayer;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem d_output;
	cl_mem *d_input;
	// scale = gamma * inv_std. These 2 params are combined to reduce the dynamic range during inference time.
	WTYPE *scale;
	// offset = -mean * inv_std * gamma + beta
	WTYPE *offset;
	cl_mem d_scale;
	cl_mem d_offset;
} BatchNormLayer;

typedef struct {
	DataShape *bot_shape;
	DataShape top_shape_0;
	DataShape top_shape_1;
} SliceLayer;

typedef struct {
	DataShape *bot_shape_0;
	DataShape *bot_shape_1;
	DataShape top_shape;
} ConcatLayer;
#endif // CNN_STRUCTS_H

