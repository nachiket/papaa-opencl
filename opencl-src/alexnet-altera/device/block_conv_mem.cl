#define BLOCK_SIZE	16
#define K 3
#define NO_LOCAL_OUTPUT_MAPS	8
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, NO_LOCAL_OUTPUT_MAPS)))
__attribute((num_simd_work_items(4)))
void block_3d_conv(
	__global float * restrict p_maps
	, __global float * restrict p_weights
	, __global float * restrict p_bias
	, __global float * restrict p_output
	, int no_inputs
	, int H
	, int W
	) {

	// local storage for one block of one input map. Extra rows and columns for padding area.
	//__local float __attribute((memory, numbanks(8), bankwidth(64), doublepump))
	//__local float __attribute__((numbanks(8), bankwidth(4))) map_blk[2*BLOCK_SIZE][2*BLOCK_SIZE];
	__local float map_blk[2*BLOCK_SIZE][2*BLOCK_SIZE];
	// local buffer for weights corresponding to 1 input map. One KxK kernel for each output map
	//__local float __attribute((memory, numbanks(8), bankwidth(64), doublepump))
	//__local float __attribute__((numbanks(8), bankwidth(4))) map_ker[NO_LOCAL_OUTPUT_MAPS][BLOCK_SIZE/2][BLOCK_SIZE/2];
	__local float map_ker[NO_LOCAL_OUTPUT_MAPS][BLOCK_SIZE][BLOCK_SIZE];
	//__local float map_ker[NO_LOCAL_OUTPUT_MAPS][BLOCK_SIZE];

	// output block index of a block of a set of output map
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);

	// output map pixel offset within block
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);

	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gsx = get_global_size(0);
	int gsy = get_global_size(1);
	// current output map
	int out_map = get_global_id(2) & 0x1FF;		// we are not going to have more than 512 maps

	// bias unit for this output map which is common to all work items
	__local float local_bias[NO_LOCAL_OUTPUT_MAPS];

	// block start location in each input map
	int row_start = block_y * BLOCK_SIZE * W;
	int col_start = block_x * BLOCK_SIZE;

	float sum = 0.0f;
	float zero = 0.0f;
	
	//int filter_start = out_map * K * K * no_inputs;
	// set a flag if this work item is entitled to copy a weight coefficient.
	//const bool copy_ker = ((local_x < K) && (local_y < K));
	// Let the work items in the center of the block in each plane copy the bias
	// as the workload on these items is less.
	//const bool copy_bias = ((local_x == BLOCK_SIZE/2) && (local_y == BLOCK_SIZE/2));
	// first few work items in z=0 plane will copy 1 column on the block
	//const bool copy_col = (local_z == 0 && (local_y * BLOCK_SIZE + local_x) < (BLOCK_SIZE + K - 1));
	//int col_idx = local_y * BLOCK_SIZE + local_x;
	/*if(copy_bias) {
		local_bias[local_z] = p_bias[out_map];
	}*/

	for(uint imap = 0; imap < no_inputs; imap++) {
		//event_t events[2];
		// pointer to this input map in global memory
		global float *p_imap = p_maps + imap * H * W;
		int filter_start = out_map * K * K * no_inputs;
		/*if(copy_col) {
			#pragma unroll
			for(uint p = 0; p < BLOCK_SIZE + K - 1; p++) {
				//map_blk[p][col_idx] = p_imap[row_start + row_idx * W + col_start + p];
				map_blk[p][col_idx] = p_imap[row_start + p*W + col_start + col_idx];
			}
		}
		// copy kernel for input map
		if(copy_ker) {
			map_ker[local_z][local_y][local_x] = p_weights[filter_start + (imap * K + local_y) * K + local_x];
		}*/
		map_blk[local_y][local_x] = p_imap[row_start + local_y*W + col_start + local_x];
		map_ker[local_z][local_y][local_x] =  p_weights[filter_start + (imap * K + local_y) * K + local_x];
		barrier(CLK_LOCAL_MEM_FENCE);
		p_imap[row_start + local_y * W + col_start + local_x] = map_blk[local_y][local_x];
		p_weights[filter_start + (imap * K + local_y) * K + local_x] = map_ker[local_z][local_y][local_x];
	}
	//p_bias[out_map] = local_bias[local_z];
	// add bias unit and write back
	p_output[(out_map * gsy + gy) * gsx + gx] = fmax(zero, sum);
}
