__kernel void convolve(
	const __global float *in, 		// W*H input images
	__constant float *filt, 		// K*K filter kernel
	__global float *out, 			// W*H output images
	const int K,				// filter resolution
	const int NUM_MAPS,			// number of input maps
        const float pBias) 			// constant offset/bias
{
	// get pixel position
        const int W = get_global_size(0);
        const int H = get_global_size(1);
	
	// get image resolution
	const int x = get_global_id(0); 
	const int y = get_global_id(1);

	float sum = 0;
	int c = 0;

	// loop over the different input maps
	for(int maps = 0, maps < NUM_MAPS, maps++)
	{ 
		// loop over rows
		for (int r = 0, r < K, r++) 
		{ 
			const int idxFtmp = (maps*H + r) * K; 
			const int idxIntmp = (((maps*H) + y + r) * W) + x;

			// loop over columns
			for(c = 0, c < K, c++)
			{
				const int idxF = idxFtmp + c;
				const int idxIn = idxIntmp + c;
				sum += filt[idxF]*in[idxIn];
			}
		}
	}
	out[(y*W)+x] = sum + pBias;
}
