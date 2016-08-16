
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
	//scoped_array<DTYPE> h_output;
	//scoped_array<DTYPE> *h_input;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem d_output;
	cl_mem *d_input;
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
        //scoped_array<DTYPE> h_output;
        //scoped_array<DTYPE> *h_input;
	scoped_aligned_ptr<DTYPE> h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem d_output;
	cl_mem *d_input;
	PoolType type;
} PoolLayer;

typedef struct {
        DataShape *bot_shape;
        DataShape top_shape;
	unsigned int no_units;
        //scoped_array<DTYPE> h_output;
        //scoped_array<DTYPE> *h_input;
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
        //scoped_array<DTYPE> *h_output;
        //scoped_array<DTYPE> *h_input;
	scoped_aligned_ptr<DTYPE> *h_output;
	scoped_aligned_ptr<DTYPE> *h_input;
	cl_mem *d_output;
	cl_mem *d_input;
	ActType type;
} ActLayer;

#endif // CNN_STRUCTS_H

