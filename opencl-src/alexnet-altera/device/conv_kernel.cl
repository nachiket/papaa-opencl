#if 0
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
__attribute__((reqd_work_group_size(1,1, 32)))
__kernel void conv_3d_relu(
	const __global float * restrict p_maps,
	const __global float * restrict p_weights,
	const __global float * restrict p_bias,
	__global float * restrict p_output,
	const unsigned int K,
	const unsigned int stride,
	const unsigned int no_inputs) {

	const int x = get_group_id(0); // This is = global id for local work group size = (1,1,32)
	const int y = get_group_id(1);
	const int z = get_local_id(2);
	const int gz = get_global_id(2);

	const int out_width  = get_global_size(0);
	const int out_height = get_global_size(1);
	const int in_width = out_width + K - 1;
	const int in_height = out_height + K - 1;
	__local float win_buff[256*3*3];	// This is the max size of the 3D patch coming from conv3

	// Assume horizontal and vertical strides are same. Generally this is the case.
	// Same assumptions holds good for kernel dimensions as well.
	int wstart = x * stride;
	int hstart = y * stride;
	int wend = wstart + K;
	int hend = hstart + K;
	// Each work item in z dimension loads  KxK patch corresponding to one or multiple input maps
	int patch_per_item = no_inputs >> 5;	// there are 32 items in z dim
	int rem_patches = no_inputs - 32*patch_per_item;
	int map_start, pr, pc;
	// load all input patches
	for(int patch = 0; patch < patch_per_item; patch++) {
		for(pr = 0; pr < K; pr++) {
			map_start = (((z+patch)*32 * in_height) + hstart + pr) * in_width + wstart;
			for(pc = 0; pc < K; pc++) {
				win_buff[((z+patch)*32*K + pr)*K + pc] = p_maps[map_start + pc];
			}
		}
	}
	// these work items need to load one extra patch
	if(z < rem_patches) {
		for(pr = 0; pr < K; pr++) {
			map_start = ((patch_per_item*32 * in_height) + hstart + pr) * in_width + wstart;
			for(pc = 0; pc < K; pc++) {
				win_buff[((patch_per_item*32 + z)*K + pr)*K + pc] = p_maps[map_start + pc];

			}
		}
	}
	const int filter_start =  gz * K * K * no_inputs;
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
			const int win_start = ((map * K) + r) * K;
			int c = 0;
			int c4 = 0;
			// vector ops
			while(c <= F-4) {
				float4 filter4 = vload4(c4, p_weights + fstart);
				float4 data4 = vload4(c4, win_buff + win_start);
				sum4 += filter4 * data4;
				c += 4;
				c4++;
			}
			// remaining columns
			for(int c1 = c; c1 < K; c1++) {
				sum4.x += p_weights[fstart + c1] * win_buff[win_start + c1];
			}

		}
	}
	sum = sum4.x + sum4.y + sum4.z + sum4.w + p_bias[gz];
	p_output[((gz*out_height) + y) * out_width + x] = fmax(zero, sum);
}
#endif

#if 0
#define MAX_INPUT_MAPS	(256)
#define MAX_KERNEL_SIZE	(11)
#define MAX_ROW_BUFF_SIZE 	(3*256*14)
#define MAX_KER_BUFF_SIZE	(3*3*256)
#define MAX_OUTPUT_ROW_SIZE	(55)
//__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void conv_3d_relu(
	__global float * restrict p_maps,
	__global float * restrict p_weights,
	__global float * restrict p_bias,
	__global float * restrict p_output,
	int K,	int stride,
	int no_inputs, int no_outputs, int H, int W, int out_height, int out_width
	) {

	int imap, row, col, d;
	//int out_height 	= H - K + 1;
	//int out_width 	= W - K + 1;
	const float zero = 0.0;
	// Local buffer to store K rows of all input maps.
	__local float row_buff[MAX_ROW_BUFF_SIZE];
	__local float ker_buff[MAX_KER_BUFF_SIZE];
	__local float out_row[MAX_OUTPUT_ROW_SIZE];
	// copy K-stride rows from each input before jumping on to main loop
	for(imap = 0; imap < no_inputs; imap++) {
		//async_work_group_copy(row_buff + imap * K * W, p_maps + imap * H  * W, W*(K-stride), 0);
		for(d = 0; d < W*(K-stride); d++) {
			row_buff[imap * K * W + d] = p_maps[imap * H * W + d];
		}
	}

	int buff_row_start = K-stride;
	for(row = 0; row < out_height; row++) {
		// copy 'stride' no of new rows from each input map
		//
		
		int row_start = row * stride + K - stride;
		for(int m = 0; m < no_inputs; m++) {
			int buff_row = buff_row_start;
			for(int r = row_start; r < row_start + stride; r++) {
				//async_work_group_copy(row_buff + (m*K + buff_row) * W, p_maps + (m * H + r) * W, W, 0);
				for(d = 0; d < W; d++) {
					row_buff[(m*K + buff_row) * W + d] = p_maps[(m * H + r) * W + d];
				}
				buff_row++;
				// To avoid costly mod operation
				if(buff_row > K) {
					buff_row = 0;
				}
			}
		}
		int ker_row_start = buff_row_start - K + stride;
		// Weight copy and compute loop
		for(int omap = 0; omap < no_outputs; omap++) {
			// copy the kernel for this output map
			//async_work_group_copy(ker_buff, p_weights + omap * K * K * no_inputs, K * K * no_inputs, 0);
			for(int w = 0; w < K * K * no_inputs; w++) {
				ker_buff[w] = p_weights[omap * K * K * no_inputs + w];
			}
			//barrier(CLK_LOCAL_MEM_FENCE);

			// fool the compiler fro optimizing
			/*for(int pix = 0; pix < no_inputs * K * W; pix++) {
				row_buff[pix] = row_buff[pix] + ker_buff[pix % W];
			}*/
			// loop over input maps
			int out_col = 0;
			for(col = 0; col < W; col+=stride) {
				float sum = 0;
				for(imap = 0; imap < no_inputs; imap++) {
					int ker_row = ker_row_start;
					//loop over kernel rows
					for(int kr = 0; kr < K; kr++) {
						for(int kc = 0; kc < K; kc++) {
							sum += ker_buff[(imap*K + ker_row)*K + kc] * row_buff[(imap*K + kr)*W + col + kc];
						}
						ker_row++;
						if(ker_row > K) {
							ker_row = 0;
						}
					}
				}
				out_row[out_col] = fmax(zero, sum + p_bias[omap]);
				out_col++;
			}
			for(out_col = 0; out_col < out_width; out_col++) {
				p_output[(omap*out_height + row)*out_width + out_col] = out_row[out_col];
			}
		}

		// Rotate buffer by stride no of rows.
		buff_row_start = buff_row_start + stride;
		// To avoid costly mod operation
		if(buff_row_start >= K) {
			buff_row_start = buff_row_start - K;
		}
	}
}
#endif

#if 1
#define NO_INPUT	256
#define NO_OUTPUT	384
#define K			3
#define STRIDE		1	
#define IN_WIDTH	14
#define OUT_WIDTH	13
#define IN_HEIGHT	14
#define OUT_HEIGHT	13

__attribute((task))
//__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void conv_3d_relu(
	__global float * restrict p_maps,
	__global float * restrict p_weights,
	__global float * restrict p_bias,
	__global float * restrict p_output
	) {

	// Local buffer to store K rows of all input maps.
	__local float row_buff[NO_INPUT][K][IN_WIDTH];	// some extra locatons for aligned transfers.
	__local float ker_buff[NO_INPUT * K * K];		// extra locations for aligned transfer.
	__local float out_row[OUT_WIDTH];
	// copy K-stride rows from each input before jumping on to main loop
	#pragma unroll
	for(int imap = 0; imap < NO_INPUT; imap++) {
		#pragma unroll
		for(int row = 0; row < K-STRIDE; row++) {
			//async_work_group_copy(row_buff[imap][row], p_maps + (imap * IN_HEIGHT + row) * IN_WIDTH, IN_WIDTH, 0);
			#pragma unroll
			for(int pix = 0; pix < IN_WIDTH; pix++) {
				row_buff[imap][row][pix] = p_maps[(imap * IN_HEIGHT + row) * IN_WIDTH + pix];
			}
		}
	}
	float zero = 0.0;
	int buff_row_start = K-STRIDE;
	for(int row = 0; row < OUT_HEIGHT; row++) {
		// copy 'stride' no of new rows from each input map
		//
		
		int row_start = row * STRIDE + K - STRIDE;
		for(int m = 0; m < NO_INPUT; m++) {
			int buff_row = buff_row_start;
			for(int r = row_start; r < row_start + STRIDE; r++) {
				//async_work_group_copy(row_buff[m][buff_row], p_maps + (m * IN_HEIGHT + r) * IN_WIDTH, IN_WIDTH, 0);
				for(int pix = 0; pix < IN_WIDTH; pix++) {
					row_buff[m][buff_row][pix] = p_maps[(m * IN_HEIGHT + r) * IN_WIDTH + pix];
				}
				buff_row++;
				// To avoid costly mod operation
				if(buff_row > K) {
					buff_row = 0;
				}
			}
		}
		// Weight copy and compute loop
		int ker_row_start = buff_row_start + STRIDE;
		if(ker_row_start >= K) {
			ker_row_start  = ker_row_start - K;
		}
		for(int omap = 0; omap < NO_OUTPUT; omap++) {
			// copy the kernel for this output map
			for(int coeff = 0; coeff < K * K * NO_INPUT; coeff++) {
				ker_buff[coeff] = p_weights[omap * K * K * NO_INPUT + coeff];
			}
			//async_work_group_copy(ker_buff, p_weights + omap * K * K * NO_INPUT, K * K * NO_INPUT, 0);
			//barrier(CLK_LOCAL_MEM_FENCE);

			// fool the compiler froM optimizing
			/*for(int col = 0; col < IN_WIDTH; col+=STRIDE) {
				int out_col = 0;
				for(int imap = 0; imap < NO_INPUT; imap++) {
					out_row[out_col] = row_buff[imap][imap%K][col] + ker_buff[out_col];
					out_col++;
				}
			}*/
			// loop over input maps
			int out_col = 0;
			for(int col = 0; col < IN_WIDTH; col+=STRIDE) {
				float sum = 0;
				for(int imap = 0; imap < NO_INPUT; imap++) {
					int ker_row = ker_row_start;
					//loop over kernel rows
					for(int kr = 0; kr < K; kr++) {
						//printf("row:%d\tker_row_start:%d\tker_row:%d\tcol: %d\n", row, ker_row_start, ker_row, col);
						for(int kc = 0; kc < K; kc++) {
							sum += ker_buff[(imap*K + kr)*K + kc] * row_buff[imap][ker_row][col + kc];
						}
						ker_row++;
						if(ker_row > K) {
							ker_row = 0;
						}
					}
				}
				out_row[out_col] = fmax(zero, sum + p_bias[omap]);
				out_col++;
			}
			for(out_col = 0; out_col < OUT_WIDTH; out_col++) {
				p_output[(omap*OUT_HEIGHT + row)*OUT_WIDTH + out_col] = out_row[out_col];
			}
		}

		// Rotate buffer by stride no of rows.
		buff_row_start = buff_row_start + STRIDE;
		// To avoid costly mod operation
		if(buff_row_start >= K) {
			buff_row_start = buff_row_start - K;
		}
	}
}
#endif

