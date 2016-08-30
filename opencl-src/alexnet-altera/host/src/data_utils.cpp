#include "data_utils.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"


void cropImage(const cv::Mat &img, cv::Mat &crop_img, unsigned int H, unsigned int W, CROP_TYPE_E type) {
		switch(type) {
			case CENTER:
			{
				int top_x = (img.cols - W)/2;
				int top_y = (img.rows - W)/2;
				cv::Rect window(top_x, top_y, W, H);
				crop_img = img(window);
				break;
			}
			case RAND:
				std::cout << "Not implemented" << std::endl;
				exit(1);
				break;
			default:
				std::cout << "Invalid crop type" << std::endl;
				exit(1);
		}
}

void initInputImage(const cv::Mat &img, DTYPE *mean, aocl_utils::scoped_aligned_ptr<DTYPE> &h_input_img) {
	// Resize image to be 256x256
	cv::Mat res_img;
	cv::Size size(256, 256);
	cv::resize(img, res_img, size);
	// mean normalization
	cv::Mat mean_norm = cv::Mat::zeros(size, CV_32FC3);
	DTYPE rm, gm, bm;
	cv::Vec3b pixel_u8;
	cv::Vec3f pixel_f;
	for(int r = 0; r < res_img.rows; r++) {
		for(int c = 0; c < res_img.cols; c++) {
			pixel_u8 = res_img.at<cv::Vec3b>(r,c);
			rm = mean[0*256*256+r*256+c];
			gm = mean[1*256*256+r*256+c];
			bm = mean[2*256*256+r*256+c];
			pixel_f.val[0] = pixel_u8.val[0] - bm;
			pixel_f.val[1] = pixel_u8.val[1] - gm;
			pixel_f.val[2] = pixel_u8.val[2] - rm;
			mean_norm.at<cv::Vec3f>(r,c) = pixel_f;
		}
	}	
	cv::Mat crop_img;

	cropImage(mean_norm, crop_img, 227, 227, CENTER);
    unsigned int C = crop_img.channels();
	unsigned int W = crop_img.cols;
	unsigned int H = crop_img.rows;
    for(int row = 0; row < H; row++) {
        for(int col = 0; col < W; col++) {
			pixel_f = crop_img.at<cv::Vec3f>(row,col);
			// BGR to RGB conversion will happen here by properly storing pixels in that manner.
			h_input_img[0*H*W + row*W + col] = pixel_f[2];
			h_input_img[1*H*W + row*W + col] = pixel_f[1];
			h_input_img[2*H*W + row*W + col] = pixel_f[0];
        }
    }
}

void zeropadAndTx(const aocl_utils::scoped_aligned_ptr<DTYPE> &src, aocl_utils::scoped_aligned_ptr<DTYPE> &dst,
	int n_ch, int src_h, int src_w, int pad_h, int pad_w, cl_mem &device_buff, cl_command_queue &queue, bool h2d_tx) {

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
		//std::cout << "Sending data to device buffer" << std::endl;
		status = clEnqueueWriteBuffer(queue, device_buff, CL_FALSE, 0,
			n_ch * dst_h * dst_w * sizeof(DTYPE), dst, 0, NULL, NULL);
		checkError(status, "Failed to transfer data to the device\n");
		clFinish(queue);
	}
}

void rand_init(aocl_utils::scoped_aligned_ptr<DTYPE> &buff, int len, int seed) {
	std::srand(seed);
	for(int i = 0; i < len; i++) {
		//buff[i] = (DTYPE)std::rand()/RAND_MAX;
		buff[i] = (DTYPE)(i % 10);
	}
}

void rand_init(WTYPE *buff, int len, int seed) {
	std::srand(seed);
	for(int i = 0; i < len; i++) {
		buff[i] = (DTYPE)std::rand()/RAND_MAX;
	}
}

// Each row will have no_input x K x K elements.
//
// This implementation reads single patch from each input maps and thus is a very 
// irregular read pattern. However this allows us to pad zeros at the end of each row of the
// matrix to make it multiple of block size
template<typename T>
void img2col(const T &img, T &mat,
	const int n_ch, const int in_h, const int in_w,
	const int K, const int stride, const int pad, const int zero_rows, const int zero_cols) {

	const int out_h = (in_h - K + 1 + 2*pad + stride - 1) / stride;
	const int out_w = (in_w - K + 1 + 2*pad + stride - 1) / stride;
	const int mat_w = K * K * n_ch + zero_cols;
	//std::cout << "H = " << out_h << std::endl;
	//std::cout << "W = " << out_w << std::endl;
	//std::cout << "Mat W = " << mat_w << std::endl;

	for(int out_row = 0; out_row < out_h; out_row++) {
		int in_row = -pad + out_row * stride; 
		for(int out_col = 0; out_col < out_w; out_col++) {
			int in_col = -pad + out_col * stride;
			int mat_row = out_row * out_w + out_col;
			for(int ch = 0; ch < n_ch; ch++) {
				// take one KxK from this input channel
				for(int kr = 0; kr < K; kr++) {
					int f_row = in_row + kr;
					for(int kc = 0; kc < K; kc++) {
						int f_col = in_col + kc;
						if(f_row < 0 || f_col < 0 || f_row >= in_h || f_col >= in_w) {
							mat[mat_row * mat_w + ch*K*K + kr*K + kc] = 0;
						} else {
							mat[mat_row * mat_w + ch*K*K + kr*K + kc] = img[(ch * in_h + f_row) * in_w + f_col];	
						}
					}
				}
			}
			// pad zeros to make the matrix width multiple of block size
			for(int p = 0; p < zero_cols; p++) {
				mat[mat_row * mat_w + n_ch * K * K + p] = 0;
			}
		}
	}
	// pad extra rows with zeros to make the matrix size to be multiple of block size
	int rows_till_now = out_h * out_w;
	for(int p = 0; p < zero_rows * mat_w; p++) {
		mat[rows_till_now * mat_w + p] = 0;
	}

}
// This is same as transpose of weight matrix containing 1 3D filter in 1 row
void weight2col(const WTYPE *W, WTYPE *wmat, const int n_ch, const int K,
	const int no_filters, const int zero_rows, const int zero_cols) {
	for(int f = 0; f < (no_filters + zero_cols); f++) {
		for(int k = 0; k < n_ch * K * K; k++) {
			if(f >= no_filters) {
				wmat[k * (no_filters + zero_cols) + f] = WTYPE(0);
			} else {
				wmat[k * (no_filters + zero_cols) + f] = W[f * n_ch * K * K + k];
			}
		}
	}
	// pad extra rows to make it multiple of block size
	memset(wmat + (no_filters + zero_cols)*n_ch*K*K, 0, zero_rows * (no_filters + zero_cols) * sizeof(WTYPE));
}
template void img2col(const  aocl_utils::scoped_aligned_ptr<DTYPE> &img, aocl_utils::scoped_aligned_ptr<DTYPE> &mat,
	const int n_ch, const int height, const int width,
	const int K, const int stride, const int pad, const int zero_rows, const int zero_cols);


template<typename T>
void col2img(const T & mat, T &img, int no_out, int out_h, int out_w, int no_pad_col) {
	for(int row = 0; row < out_h; row++) {
		for(int col = 0; col < out_w; col++) {
			int mat_offset = (row * out_w  + col)*(no_out + no_pad_col);
			for(int map = 0; map < no_out; map++) {
				img[(map * out_h + row)*out_w + col] = mat[mat_offset + map];
			}
		}
	}
}
template void col2img(const aocl_utils::scoped_aligned_ptr<DTYPE>& mat,
	aocl_utils::scoped_aligned_ptr<DTYPE> &img, int no_out, int out_h, int out_w, int no_pad_col);

