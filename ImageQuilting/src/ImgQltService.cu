/*
 * ImgQltService.cu
 *
 *  Created on: 07-Oct-2015
 *      Author: mobin
 */

#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include"opencv2/highgui/highgui.hpp"
#include<cuda_runtime.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include"opencv2/cudaarithm.hpp"
#include "tbb/tbb_stddef.h"

#define SAMPLE_SIZE 20
#define OVERLAP_SIZE 5
#define OUTPUTX_SIZE 250
#define OUTPUTY_SIZE 250

using std::cout;
using std::endl;
using namespace cv::cuda;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if(err!=cudaSuccess) {
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__device__ uchar* getSubImg(uchar* dSrc, int row, int col, int step) {
	return &dSrc[step * row + col * 3];
}

__device__ uchar getGrayElement(uchar* subArray, int row, int col, int step) {
	int b = subArray[row * step + col];
	int g = subArray[row * step + col + 1];
	int r = subArray[row * step + col + 2];
	return 0.2989 * r + 0.5870 * g + 0.1140 * b;
}

__global__ void cudaGetMinSSDImg(uchar* dSrc, uchar* preImg, uchar* topImg, int step, float* ssidArr) {

	int blkcolIdx = blockIdx.x;
	int blkrowIdx = blockIdx.y;

	int colIdx = threadIdx.x;
	int rowIdx = threadIdx.y;

	uchar* subArray = getSubImg(dSrc, blkrowIdx, blkcolIdx, step);

	__shared__ uchar subImgGray[SAMPLE_SIZE][SAMPLE_SIZE];
	__shared__ uchar preImgGray[SAMPLE_SIZE][SAMPLE_SIZE];
	__shared__ uchar topImgGray[SAMPLE_SIZE][SAMPLE_SIZE];

	subImgGray[rowIdx][colIdx] = getGrayElement(subArray, rowIdx, colIdx * 3, step);
	if (preImg != 0) {
		preImgGray[rowIdx][colIdx] = getGrayElement(preImg, rowIdx, colIdx * 3, step);
	}
	if (topImg != 0) {
		topImgGray[rowIdx][colIdx] = getGrayElement(topImg, rowIdx, colIdx * 3, step);
	}

	//only the first thread from each block need to work on the rest
	if (rowIdx == 0 && colIdx == 0) {

		__syncthreads();

		float ssid = 0;

		if (preImg != 0) {
			for(int i = 0; i < SAMPLE_SIZE; i++) {
				for(int j = 0; j < OVERLAP_SIZE; j++) {
					int diff = subImgGray[i][j] - preImgGray[i][SAMPLE_SIZE - OVERLAP_SIZE + j];
					ssid += sqrtf((float) (diff * diff));
				}
			}
		}

		if (topImg != 0) {
			for(int i = 0; i < OVERLAP_SIZE; i++) {
				for(int j = 0; j < SAMPLE_SIZE; j++) {
					int diff = subImgGray[i][j] - topImgGray[SAMPLE_SIZE - OVERLAP_SIZE + i][j];
					ssid += sqrtf((float) (diff * diff));
				}
			}
		}

		int idx = (blkrowIdx * gridDim.x) + blkcolIdx;
		ssidArr[idx] = ssid;
	}
}

cv::Mat getMinSSDImg(cv::Mat& prevImg, cv::Mat& topImg, cv::Mat& hSrc, cv::cuda::GpuMat& dSrc, int width, int height) {
	cv::cuda::GpuMat d_prevImg, d_topImg;

	//dSrc.upload(hSrc);
	d_prevImg.upload(prevImg);
	d_topImg.upload(topImg);

	cv::cuda::GpuMat d_curImg(SAMPLE_SIZE, SAMPLE_SIZE, CV_8UC3);
	const dim3 grid(width-SAMPLE_SIZE,height-SAMPLE_SIZE);
	const dim3 block(SAMPLE_SIZE,SAMPLE_SIZE);

	float h_ssidArr[height-SAMPLE_SIZE][width-SAMPLE_SIZE];

	float* d_ssidArr;
	size_t arraysize = (width - SAMPLE_SIZE) * (height - SAMPLE_SIZE) * sizeof(*d_ssidArr);

	SAFE_CALL(cudaMalloc<float>(&d_ssidArr,arraysize),"CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(d_ssidArr,h_ssidArr,arraysize,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	float ms;

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaGetMinSSDImg<<<grid,block>>>(dSrc.ptr(), d_prevImg.ptr(), d_topImg.ptr(), dSrc.step, d_ssidArr);
	//cudaDeviceSynchronize();

	//calculating time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout<<"Execution time:"<<ms<<"  ms"<<std::endl;

	SAFE_CALL(cudaMemcpy(h_ssidArr,d_ssidArr,arraysize,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	float minssid = FLT_MAX;
	int rowidx = 0;
	int colidx = 0;
	for(int i = 0; i < height - SAMPLE_SIZE; i++) {
		for(int j = 0; j < width - SAMPLE_SIZE; j++) {
			if(minssid > h_ssidArr[i][j]) {
				minssid = h_ssidArr[i][j];
				rowidx = i;
				colidx = j;
			}
		}
	}

	cv::Mat curImg = hSrc(cv::Range(rowidx, rowidx + SAMPLE_SIZE), cv::Range(colidx, colidx + SAMPLE_SIZE));
	return curImg;
}

cv::Mat getPreviousImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(j == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range((i*SAMPLE_SIZE)-(OVERLAP_SIZE*i),((i+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*i)),cv::Range(((j-1)*SAMPLE_SIZE)-(j-1)*OVERLAP_SIZE,(j*SAMPLE_SIZE)-(j-1)*OVERLAP_SIZE));
		return subImg;
	}
}

cv::Mat getTopImg(int i, int j, cv::Mat& hDst) {
	cv::Mat subImg;
	if(i == 0) {
		return subImg;
	} else {
		subImg = hDst(cv::Range(((i-1)*SAMPLE_SIZE)-(i-1)*OVERLAP_SIZE,(i*SAMPLE_SIZE)-(i-1)*OVERLAP_SIZE),cv::Range((j*SAMPLE_SIZE)-(OVERLAP_SIZE*j),((j+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*j)));
		return subImg;
	}
}

void placeImg(int row, int col, cv::Mat& tile, cv::Mat& lImg) {
	int x1 = (row*SAMPLE_SIZE)-(OVERLAP_SIZE*row);
	int x2 = ((row+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*row);
	int y1 = (col*SAMPLE_SIZE)-(OVERLAP_SIZE*col);
	int y2 = ((col+1)*SAMPLE_SIZE)-(OVERLAP_SIZE*col);
	if(row == 0) {
		x1 = (row*SAMPLE_SIZE);
		x2 = ((row+1)*SAMPLE_SIZE);
	}
	if(col == 0){
		y1 = (col*SAMPLE_SIZE);
		y2 = ((col+1)*SAMPLE_SIZE);
	}

	tile.copyTo(lImg(cv::Range(x1, x2), cv::Range(y1, y2)));
}

void imageQuilting(cv::Mat& hSrc, cv::Mat& hDst) {

	int height = hSrc.rows;
	int width = hSrc.cols;

	cv::cuda::GpuMat dSrc;
	dSrc.upload(hSrc);

	int nx = OUTPUTX_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);
	int ny = OUTPUTY_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);
	//int newx = nx + (height - nx * OVERLAP_SIZE) / SAMPLE_SIZE;
	//int newy = ny + (width - ny * OVERLAP_SIZE) / SAMPLE_SIZE;

	for(int i = 0; i < nx; i++ ) {
		for(int j = 0; j < ny; j++) {

			cv::Mat prevImg = getPreviousImg(i, j, hDst);

			cv::Mat topImg = getTopImg(i, j, hDst);

			cv::Mat currImg;

			currImg = getMinSSDImg(prevImg, topImg, hSrc, dSrc, width, height);

			placeImg(i, j, currImg, hDst);

		}
	}

	//dSrc.copyTo(dDst);
	//dDst.download(hDst);
}

int main() {


	int num_devices = getCudaEnabledDeviceCount();
	cout << "gpu count :" << num_devices << endl;

	std::string imageName = "image5.jpg";
	cv::Mat input = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);

	if (input.empty()) {
		cout << "Cannot read " + imageName << endl;
	} else {
		cout << imageName + " loaded" << endl;
	}
	cv::Mat output(OUTPUTY_SIZE, OUTPUTX_SIZE, CV_8UC3);

	clock_t start = clock();
	imageQuilting(input, output);

	clock_t end = clock();
	double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

	cout << "time elapsed :" << elapsed_secs << endl;
	cv::imshow("Output", output);

	cv::waitKey();

	return 0;
}
