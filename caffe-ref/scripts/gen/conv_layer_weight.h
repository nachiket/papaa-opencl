/*Header file for convolution layer weights and biases*/
#ifndef _CONV_LAYER_WEIGHT_H_
#define _CONV_LAYER_WEIGHT_H_
#include <stdio.h>

#define NO_INPUT_MAPS  3

#define NO_OUTPUT_MAPS  5

#define FILTER_HEIGHT  5

#define FILTER_WIDTH  5

extern const float conv_layer_weights[NO_OUTPUT_MAPS][NO_INPUT_MAPS*FILTER_HEIGHT*FILTER_WIDTH];

extern const float conv_layer_bias[NO_OUTPUT_MAPS];

#endif // _CONV_LAYER_WEIGHT_H_