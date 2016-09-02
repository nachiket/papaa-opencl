#if 0
#define FC_WG_SIZE			(256)			// compute 1024 outputs for last layer and discard 24 of them
#define FC_SIMD_ITEMS		(4)
#define FC_MAX_INPUT_SIZE	(256 * 8 * 8)	// nearest power of 2
__kernel
__attribute((max_work_group_size(4096)))
__attribute((num_simd_work_items(FC_SIMD_ITEMS)))
__attribute((reqd_work_group_size(FC_WG_SIZE,1,1)))
void fc_layer_relu(
	const __global float * restrict p_input,
	const __global float * restrict p_weights,
	__global float * restrict p_output,
	const int no_inputs,
	const __global float * restrict p_bias,
	const unsigned char act) {

	// TODO: partition into FC_SIMD_ITEMS banks so that they can have parallel access.
	// One read and one write port is enough
	__local float input_buff[FC_MAX_INPUT_SIZE];

	//////////////////int block_x = get_group_id(0);
	int local_x = get_local_id(0);
	int global_x = get_global_id(0);
	// copy inputs
	// FIXME: This assumes that no_inputs % FC_WG_SIZE = 0. This is true for alexnet
	//int inputs_per_item = no_inputs / FC_WG_SIZE;

	for(int in = 0; in < no_inputs; in+=FC_WG_SIZE) {
		input_buff[in + local_x] = p_input[in + local_x];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int w_start = no_inputs * global_x;
	float sum = 0.0f;
	float zero = 0.0f;
	#pragma unroll 32
	for(int i = 0; i < no_inputs; i++) {
		sum += p_weights[w_start + i] * input_buff[i];
	}
	sum += p_bias[global_x];
	if(act) {
		sum = fmax(zero, sum);
	}
	p_output[global_x] = sum;
}
#endif

#if 0
// Single work item kernel
__kernel void fc_layer_relu(
	const __global float * restrict p_input,
	const __global float * restrict p_weights,
	__global float * restrict p_output,
	const int no_inputs,
	const int no_outputs,
	const __global float * restrict p_bias,
	const unsigned char act) {

	float sum;
	float zero = 0;
	for(int out = 0; out < no_outputs; out++) {
		sum = 0;
		#pragma unroll 32
		for (int i = 0; i < no_inputs; i++) 
		{
			sum += p_weights[out * no_inputs + i] * p_input[i];
		}
		sum += p_bias[out];

		if(act == 1) {
			sum = fmax(zero, sum);
		}
		p_output[out] = sum;
	}
}
#endif

#if 1
/* Runtime
 * fc6: 22.79
 * fc7: 10.08
 * fc8: 2.80
 * Input is cached(100% cache hit). 14% Logic
 * weight read causing 76% worst case stall
 * weight read BW = 6784MB/s, burst size = 2
 * input read burst size = 2
 * output write burst size = 16
 * bias read burst size = 16
 */
__kernel void fc_layer_relu(
	const __global float * restrict pInput,
	const __global float * restrict pWeights,
	const __global float * restrict pBias,
	__global float * restrict pOutput,
	const int nInputs,
	const int act) {

	const int x = get_global_id(0);
	const int idxstart = x*nInputs;
	float sum = 0;
	float zero = 0;
	#pragma unroll 32
	for (int i = 0; i <nInputs; i++) 
	{
		sum += pWeights[idxstart+i]*pInput[i];
	}
	sum += pBias[x];
	if(act == 1) {
		pOutput[x] = fmax(zero, sum);
	} else {
		pOutput[x] = sum;
	}
}
#endif

#if 0
/* Runtime
 * fc6: 51.00
 * fc7: 22.99
 * fc8: 5.30
 * compiler is replicating all local buffers 60 times. 20% Logic
 * weight read BW = 3500MB/s, burst size = 2
 * input read burst size = 2
 * output write burst size = 6
 * bias read burst size = 16
 */
#ifndef NO_SIMD_ITEMS
#define NO_SIMD_ITEMS 4
#endif
#define BLOCK_SIZE	32

__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__attribute((num_simd_work_items(NO_SIMD_ITEMS)))
void fc_layer_relu(
	__global float * p_inputs,
	__global float * p_weights,
	__global float * p_bias,
	__global float * p_output,
	int no_inputs,
	int act
	 ) {
	
	// Local storage for weights. Even though the work items work on non-overlapping portion of this matrix, this will 
	// help in balancing the operand access latency as the input is shared among all work items.
	__local float local_weight[BLOCK_SIZE][BLOCK_SIZE];
	// Local storage for input and bias
	__local float local_input[BLOCK_SIZE];
	__local float local_bias[BLOCK_SIZE];
	
	int block_x = get_group_id(0);
	int local_x = get_local_id(0);
	int global_x = get_global_id(0);

	int weight_start = (block_x * BLOCK_SIZE) * no_inputs;
	float running_sum = 0.0f;
	float zero = 0.0f;
	// load bias 
	local_bias[local_x] = p_bias[global_x];
	for(int in = 0, w = weight_start; in < no_inputs; in += BLOCK_SIZE, w += BLOCK_SIZE) {		
		// load input, one element per work item
		local_input[local_x] = p_inputs[in + local_x];
		// load weights. BLOCK_SIZE no of weights per work item.
		// accessing the global memory is such that the consecutive locations are accessed by different work items.
		// #pragma unroll
		for(int bw = 0; bw < BLOCK_SIZE; bw++) {
			// To make global memory access 
			local_weight[bw][local_x] = p_weights[w + no_inputs * bw + local_x];
		}
		// wait for all work items to finish copying. 
		barrier(CLK_LOCAL_MEM_FENCE);
		// partial dot product
		#pragma unroll
		for(int e = 0; e < BLOCK_SIZE; e++) {
			running_sum += local_weight[local_x][e] * local_input[e];
		}
		// wait for the shared data to be consumed by all work items.
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	running_sum += local_bias[local_x];
	if(act == 1) {
		running_sum = fmax(zero, running_sum);
	}
	p_output[global_x] = running_sum;
}
#endif
