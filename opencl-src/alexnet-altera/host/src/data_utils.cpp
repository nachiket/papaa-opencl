#include "data_utils.h"
template<typename T>
void showMat(T buff, int n_ch, int h, int w, int to_show=3) {
	for(unsigned int ch = 0; ch < to_show; ch++) {
		cout << "Channel: " << ch << endl;
		for(unsigned int r = 0; r < h; r++) {
			for(unsigned int c = 0; c < w; c++) {
				cout << buff[ch*h*w+r*w+c] << ",";
			}
			cout << endl;
		}
	} 
}

Mat & cropImage(const Mat &img, unsigned int H, unsigned int W, CROP_TYPE_E type) {
		switch(type) {
			case CENTER:
			{
				int top_x = (img.cols - W)/2;
				int top_y = (img.rows - W)/2;
				Rect window(top_x, top_y, W, H);
				Mat crop_img = img(window);
				return crop_img;
				break;
			}
			case RAND:
				cout << "Not implemented" << endl;
				exit(1);
				break;
			default:
				cout << "Invalid crop type" << endl;
				exit(1);
		}
}

void initInputImage(const Mat &img, const Mat &mean, scoped_aligned_ptr<DTYPE> &h_input_img) {
    uint8_t *p_img = (uint8_t *)img.data;
	uint8_t *p_mean = (uint8_t *)mean.data;
    uint8_t r,g,b, rm, gm, bm;
	assert(img.channels() == mean.channels() && img.rows == mean.rows && img.cols == mean.cols);
    unsigned int C = img.channels();
	unsigned int W = img.cols;
	unsigned int H = img.rows;
    for(int row = 0; row < H; row++) {
        for(int col = 0; col < W; col++) {
            b = p_img[row*W*C + col*C + 0];
            g = p_img[row*W*C + col*C + 1];
            r = p_img[row*W*C + col*C + 2];
            bm = p_mean[row*W*C + col*C + 0];
            gm = p_mean[row*W*C + col*C + 1];
            rm = p_mean[row*W*C + col*C + 2];
			h_input_img[0*H*W + row*W + col] = (DTYPE)(r - rm);
			h_input_img[1*H*W + row*W + col] = (DTYPE)(g - rm);
			h_input_img[2*H*W + row*W + col] = (DTYPE)(b - rm);
        }
    }
}

void zeropadAndTx(const scoped_aligned_ptr<DTYPE> &src, scoped_aligned_ptr<DTYPE> &dst,
	int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff, bool h2d_tx) {

	unsigned dst_h = src_h + 2*pad_h;
	unsigned dst_w = src_w + 2*pad_w;
	cl_int status;
	for(int ch = 0; ch < n_ch; ch++) {
		for(int r = 0; r < src_h + 2*pad_h; r++) {
			for(int c = 0; c < src_w + 2*pad_w; c++) {
				if(r < pad_h || (r > src_h+pad_h-1) || c < pad_w || (c > src_w+pad_w-1)) {
					dst[ch*dst_h*dst_w + r*dst_w + c] = (DTYPE)0;
				}else {
					// TODO: use memcpy instead to do bulk copy of one full row
					dst[ch*dst_h*dst_w + r*dst_w + c] = src[ch*src_h*src_w + (r-pad_h)*src_w + c-pad_w];
				}
			}	
		}
	}
	if(h2d_tx) {
		cout << "Sending data to device buffer" << endl;
		status = clEnqueueWriteBuffer(queue, device_buff, CL_FALSE, 0,
			n_ch * dst_h * dst_w * sizeof(DTYPE), dst, 0, NULL, NULL);
		checkError(status, "Failed to transfer data to the device\n");
		clFinish(queue);
	}
}
