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
	#pragma unroll 4
	for(int out = 0; out < n_outputs; out++) {
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
