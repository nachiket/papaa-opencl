/*Header file for model  weights and biases*/
#ifndef _CIFAR10_MODEL_H_
#define _CIFAR10_MODEL_H_
#include <stdio.h>

#define CONV1_NO_INPUTS  3

#define CONV1_NO_OUTPUTS  32

#define CONV1_FILTER_HEIGHT  5

#define CONV1_FILTER_WIDTH  5

extern const float conv1_weights[CONV1_NO_OUTPUTS][CONV1_NO_INPUTS*CONV1_FILTER_HEIGHT*CONV1_FILTER_WIDTH];

extern const float conv1_bias[CONV1_NO_OUTPUTS];

#define CONV2_NO_INPUTS  32

#define CONV2_NO_OUTPUTS  32

#define CONV2_FILTER_HEIGHT  5

#define CONV2_FILTER_WIDTH  5

extern const float conv2_weights[CONV2_NO_OUTPUTS][CONV2_NO_INPUTS*CONV2_FILTER_HEIGHT*CONV2_FILTER_WIDTH];

extern const float conv2_bias[CONV2_NO_OUTPUTS];

#define CONV3_NO_INPUTS  32

#define CONV3_NO_OUTPUTS  64

#define CONV3_FILTER_HEIGHT  5

#define CONV3_FILTER_WIDTH  5

extern const float conv3_weights[CONV3_NO_OUTPUTS][CONV3_NO_INPUTS*CONV3_FILTER_HEIGHT*CONV3_FILTER_WIDTH];

extern const float conv3_bias[CONV3_NO_OUTPUTS];

#define IP1_NO_INPUTS  1024

#define IP1_NO_OUTPUTS  64

extern const float ip1_weights[IP1_NO_OUTPUTS][IP1_NO_INPUTS];

extern const float ip1_bias[IP1_NO_OUTPUTS];

#define IP2_NO_INPUTS  64

#define IP2_NO_OUTPUTS  10

extern const float ip2_weights[IP2_NO_OUTPUTS][IP2_NO_INPUTS];

extern const float ip2_bias[IP2_NO_OUTPUTS];

#endif // _LENET5_MODEL_H_