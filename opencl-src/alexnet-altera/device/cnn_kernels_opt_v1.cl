//__attribute__((num_compute_units(2))) // This fails to fit on DE5
__kernel void conv_3d_relu(
	const __global float * restrict p_maps,
	const __global float * restrict p_weights,
	const __global float * restrict p_bias,
	__global float * restrict p_output,
	const unsigned int K,
	const unsigned int stride,
	const unsigned int no_inputs) {

	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	
	const int out_width  = get_global_size(0);
	const int out_height = get_global_size(1);
	const int in_width = out_width + K - 1;
	const int in_height = out_height + K - 1;

	// Assume horizontal and vertical strides are same. Generally this is the case.
	// Same assumptions holds good for kernel dimensions as well.
	int wstart = x * stride;
	int hstart = y * stride;
	int wend = wstart + K;
	int hend = hstart + K;

	const int filter_start = z * K * K * no_inputs;
	int F = (int)K;
	float pix, w;
	float4 sum4 = 0.0;
	float zero = 0.0;
	float sum = 0.0;
	#pragma unroll 2
	for(unsigned int map = 0; map < no_inputs; map++) {
		#pragma unroll 3
		for(unsigned int r = 0; r < K; r++) {
			const int fstart = filter_start + map * K * K + r * K;
			const int map_start = ((map * in_height) + hstart + r) * in_width + wstart;
			int c = 0;
			int c4 = 0;
			// vector ops
			while(c <= F-4) {
				float4 filter4 = vload4(c4, p_weights + fstart);
				float4 data4 = vload4(c4, p_maps + map_start);
				sum4 += filter4 * data4;
				c += 4;
				c4++;
			}
			// remaining columns
			for(int c1 = c; c1 < K; c1++) {
				sum4.x += p_weights[fstart + c1] * p_maps[map_start + c1];
			}

		}
	}
	sum = sum4.x + sum4.y + sum4.z + sum4.w + p_bias[z];
	p_output[((z*out_height) + y) * out_width + x] = fmax(zero, sum);
}
//FIXME: If taking trained model from Lasagne, the conv filters flipped by default.
//Either perform flip here OR disable flip during training !!!
// 3D convolution + ReLU activation kernel
//__attribute__((max_work_group_size(256, 256, 1)))
/*__kernel void conv_3d_relu(
	const __global float * restrict p_maps,
	const __global float * restrict p_weights,
	const __global float * restrict p_bias,
	__global float * restrict p_output,
	const unsigned int K,
	const unsigned int stride,
	const unsigned int no_inputs,
	const unsigned int no_outputs) {

	// FIXME: local wg size must be equal to global qork group size for this to be functionally correct.
	const int x = get_local_id(0); 
	const int y = get_local_id(1);
//	const int z = get_global_id(2);
	
	const int out_width  = get_local_size(0);
	const int out_height = get_local_size(1);
	const int in_width = out_width + K - 1;
	const int in_height = out_height + K - 1;

	// Assume horizontal and vertical strides are same. Generally this is the case.
	// Same assumptions holds good for kernel dimensions as well.
	int wstart = x * stride;
	int hstart = y * stride;
	int total_work_items = out_width*out_height;
	__local float local_filter[9216];
	int no_weights = no_inputs * K * K;
	int weights_per_item = no_weights/total_work_items;
	int rem_weights = no_weights - weights_per_item * total_work_items;

	for(int z = 0; z < no_outputs; z++) {
		const int filter_start = z * no_weights;
		// copy weights to local buffer
		for(int w = 0 ; w < weights_per_item; w++) {
			local_filter[w*total_work_items + y*out_width + x] = p_weights[filter_start + w*total_work_items + y*out_width + x];
		}
		if(y*out_width + x < rem_weights) {
			local_filter[weights_per_item*total_work_items + y*out_width + x] = p_weights[filter_start + weights_per_item*total_work_items + y*out_width + x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		int F = (int)K;
		float pix, w;
		float4 sum4 = 0.0;
		float zero = 0.0;
		float sum = 0.0;
		#pragma unroll 2
		for(unsigned int map = 0; map < no_inputs; map++) {
			#pragma unroll 2
			for(unsigned int r = 0; r < K; r++) {
				const int fstart = map * K * K + r * K;
				const int map_start = ((map * in_height) + hstart + r) * in_width + wstart;
				int c = 0;
				int c4 = 0;
				// vector ops
				while(c <= F-4) {
					float4 filter4 = vload4(c4, local_filter + fstart);
					float4 data4 = vload4(c4, p_maps + map_start);
					sum4 += filter4 * data4;
					c += 4;
					c4++;
				}
				// remaining columns
				for(int c1 = c; c1 < K; c1++) {
					sum4.x += local_filter[fstart + c1] * p_maps[map_start + c1];
				}

			}
		}
		sum = sum4.x + sum4.y + sum4.z + sum4.w + p_bias[z];
		p_output[((z*out_height) + y) * out_width + x] = fmax(zero, sum);
	}
}*/
__kernel void maxpool_3d(
	const __global float * restrict pInput,
	__global float * restrict pOutput,
	const int iWidth,
	const int iHeight,
	const int nPoolsize,
	const int nStride) {

	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	
	const int oWidth  = get_global_size(0);
	const int oHeight = get_global_size(1);

	float maxval = -3.402823e+37;
	int hstart = y*nStride;
	int wstart = x*nStride;
	int hend = hstart+nPoolsize;
	int wend = wstart+nPoolsize;
	for(unsigned int r = hstart; r < hend; r++) {
		for(unsigned int c = wstart; c < wend; c++) {
			unsigned int idx = z*iHeight*iWidth + r * iWidth + c;
			maxval = fmax(maxval, pInput[idx]);
		}
	}
        pOutput[(((z*oHeight)+y)*oWidth)+x] = maxval;
}
#if 0
#define SINGLE_ITEM_FC
#define USE_LOCAL_MEM
// Perceptron layer + conditional ReLU activation
//__attribute__((max_work_group_size(256)))
#ifndef SINGLE_ITEM_FC
__attribute__((num_simd_work_items(4)))
__attribute__((reqd_work_group_size(8,1,1)))
__kernel void fc_layer_relu(
	const __global float * restrict pInput,
	const __global float * restrict pWeights,
	__global float * restrict pOutput,
	const int nInputs,
	const __global float * restrict pBias,
	const unsigned char act) {

	// Local RAM to share the input across local work items.
	// The input to fc6 has size = 9216 and all following layers have lesses size than this.
	const int gx = get_global_id(0);

	// Transfer input vector to local memory
#ifdef USE_LOCAL_MEM
	const int lx = get_local_id(0);
	const int ls = get_local_size(0);
	__local float local_input[256*6*6];
	int inputs_per_item = nInputs/ls;
	int rem_inputs = nInputs - inputs_per_item * ls;
	for(int item = 0; item < inputs_per_item; item++) {
		local_input[item*ls + lx] = pInput[item * ls + lx];
	}
	// Few work items read remaining inputs.
	if(lx < rem_inputs) {
		local_input[ls*inputs_per_item + lx] = pInput[ls*inputs_per_item + lx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif
	const int idxstart = gx*nInputs;
	float sum = 0;
	float zero = 0;
	#pragma 32
	for (int i = 0; i <nInputs; i++) 
	{
#ifdef USE_LOCAL_MEM
		sum += pWeights[idxstart+i]*local_input[i];
#else
		sum += pWeights[idxstart+i]*pInput[i];
#endif
	}
	sum += pBias[gx];
	if(act == 1) {
		pOutput[gx] = fmax(zero, sum);
	} else {
		pOutput[gx] = sum;
	}
}
#else // SINGLE_ITEM_FC
// Single work item Perceptron layer + conditional ReLU activation implementation
__kernel void fc_layer_relu(
	const __global float * restrict pInput,
	const __global float * restrict pWeights,
	__global float * restrict pOutput,
	const int nInputs,
	const int nOutputs,
	const __global float * restrict pBias,
	const unsigned char act) {

	float sum;
	float zero = 0;
	for(int out = 0; out < nOutputs; out++) {
		sum = 0;
		for (int i = 0; i <nInputs; i++) 
		{
			sum += pWeights[out*nInputs+i]*pInput[i];
		}
		sum += pBias[out];
		if(act == 1) {
			pOutput[out] = fmax(zero, sum);
		} else {
			pOutput[out] = sum;
		}
	}
}
#endif // SINGLE_ITEM_FC
#endif

#define FC_WG_SIZE			(256)			// compute 1024 outputs for last layer and discard 24 of them
#define FC_SIMD_ITEMS		(4)
#define FC_MAX_INPUT_SIZE	(256 * 8 * 8)	// nearest power of 2
__kernel
__attribute__((num_simd_work_items(FC_SIMD_ITEMS)))
__attribute__((reqd_work_group_size(FC_WG_SIZE,1,1)))
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

	int block_x = get_group_id(0);
	int local_x = get_local_id(0);
	int global_x = get_global_id(0);
	// copy inputs
	// FIXME: This assumes that no_inputs % FC_WG_SIZE = 0
	for(int in = 0; in < no_inputs; in += FC_WG_SIZE) {
		input_buff[in + local_x] = p_input[in + local_x];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int w_start = no_inputs * global_x;
	float sum = 0.0f;
	float zero = 0.0f;

	for(int i = 0; i < no_inputs; i++) {
		sum += p_weights[w_start + i] * input_buff[i];
	}
	sum += p_bias[global_x];
	if(act) {
		sum = fmax(zero, sum);
	}
	p_output[global_x] = sum;
}

// Need to do piecewise linear approximation for exp(x)
// Just implementing exp here. Normalizing probalities to be carried out on the host.
__attribute__((max_work_group_size(1000)))
__kernel void softmax(
	__global float * pdata) {

	//__local float sum, prob[1000];
	const int x = get_global_id(0);
	pdata[x] = exp(pdata[x]);

	/*barrier(CLK_LOCAL_MEM_FENCE);
	if(x == 0) {
		sum = 0;
		for(int i=0; i< get_local_size(0); i++) {
			sum += prob[i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	pdata[x] = prob[x]/sum;
	pdata[x] = prob[x];	*/
}


__kernel void batch_norm_layer(
	__global float * restrict pMaps,
	// TODO: use local memory for scale and offset
	__global float * restrict pScale,
	__global float * restrict pOffset,
	__global float * restrict pOutput) {
	
	const int map_no = get_global_id(2);
	const int row_no = get_global_id(1);
	const int col_no = get_global_id(0);
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	float norm_pix;
	const int pos = map_no*W*H + row_no*W + col_no;
	norm_pix = pMaps[pos] * pScale[map_no] + pOffset[map_no];
	pOutput[pos] = norm_pix;
}
