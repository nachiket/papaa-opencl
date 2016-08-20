//FIXME: If taking trained model from Lasagne, the conv filters flipped by default.
//Either perform flip here OR disable flip during training !!!
__kernel void conv_3d(
	const __global float *p_maps,
	const __global float *p_weights,
	const __global float *p_bias,
	__global float * p_output,
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

	int filter_start = z * K * K * no_inputs;
	float pix, w;
	float sum = 0.0;
	for(unsigned int map = 0; map < no_inputs; map++) {
		for(unsigned int r = hstart; r < hend; r++) {
			for(unsigned int c = wstart; c < wend; c++) {
				pix = p_maps[((map*in_height) + r )*in_width + c];
				w = p_weights[filter_start + map * K * K + (r-hstart)*K + c - wstart];
				sum += pix * w;
			}
		}
	}
	p_output[((z*out_height) + y) * out_width + x] = sum + p_bias[z];
}

__kernel void maxpool3D(
        const __global float * pInput,
        __global float * pOutput,
        const int iWidth,
        const int iHeight,
        const int nPoolsize,
        const int nStride)
{
        const int x = get_global_id(0); 
        const int y = get_global_id(1);
        const int z = get_global_id(2);

        const int oWidth  = get_global_size(0);
        const int oHeight = get_global_size(1);

	float maxval = -3.402823e+37;
	int hstart = y*nStride;
	int wstart = x*nStride;
	//int hend = min(hstart+nPoolsize, iHeight);
	//int wend = min(wstart+nPoolsize, iWidth);
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


__kernel void iplayer(
        const __global float * pInput,
        const __global float * pWeights,
        __global float * pOutput,
        const int nInputs,
        const __global float * pBias)
{
        const int x = get_global_id(0);
	const int idxstart = x*nInputs;
        float sum = 0;
        for (int i = 0; i <nInputs; i++) 
        {
           sum += pWeights[idxstart+i]*pInput[i];
        }
        pOutput[x] = sum + pBias[x];
}

__kernel void relu_layer (__global float * pData)
{
        const int x = get_global_id(0);
        float zero = 0.0;
        pData[x] = fmax(zero,pData[x]);
}

// Need to do piecewise linear approximation for exp(x)
__kernel void softmax(
	__global float * pdata) {

	__local float sum, prob[1000];
	const int x = get_global_id(0);
	prob[x] = exp(pdata[x]);

	barrier(CLK_LOCAL_MEM_FENCE);
	if(x == 0) {
		sum = 0;
		for(int i=0; i< get_global_size(0); i++) {
			sum += prob[i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	pdata[x] = prob[x]/sum; 
}

__kernel void batch_norm_layer(
	__global float *pMaps,
	// TODO: use local memory for scale and offset
	__global float *pScale,
	__global float *pOffset,
	__global float *pOutput) {
	
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
