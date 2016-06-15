/*Header file for inner-product layer weights and biases*/
#ifndef _IP_LAYER_WEIGHT_H_
#define _IP_LAYER_WEIGHT_H_
#include <stdio.h>

#define NO_INPUT  784

#define NO_OUTPUT  100

extern const float ip_layer_weights[NO_OUTPUT][NO_INPUT];

extern const float ip_layer_bias[NO_OUTPUT];

#endif // _IP_LAYER_WEIGHT_H_