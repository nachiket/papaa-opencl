/*Header file for convolution layer weights and biases*/
#ifndef _CONV_LAYER_WEIGHT_H_
#define _CONV_LAYER_WEIGHT_H_
#include <stdio.h>

#define CONV1_NO_INPUTS  3

#define CONV1_NO_OUTPUTS  5

#define CONV1_FILTER_HEIGHT  5

#define CONV1_FILTER_WIDTH  5

extern const float conv1_weights[CONV1_NO_OUTPUTS][CONV1_NO_INPUTS*CONV1_FILTER_HEIGHT*CONV1_FILTER_WIDTH];

extern const float conv1_bias[CONV1_NO_OUTPUTS];

#endif // _CONV_LAYER_WEIGHT_H_