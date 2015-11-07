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
#include <opencv2/imgproc/imgproc.hpp>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include<map>
#include<list>
#include<vector>

#define SAMPLE_SIZE 20
#define OVERLAP_SIZE 5
#define OUTPUTX_SIZE 250
#define OUTPUTY_SIZE 250

using std::cout;
using std::endl;
using namespace std;

std::vector<cv::Mat> createImageList(cv::Mat& hSrc) {
	int x_size = hSrc.rows;
	int y_size = hSrc.cols;
	std::vector<cv::Mat> imglist((x_size - SAMPLE_SIZE) * (y_size - SAMPLE_SIZE));
	for(int i = 0; i < x_size - SAMPLE_SIZE; i++) {
		for(int j = 0; j < y_size - SAMPLE_SIZE; j++) {
			imglist[(i * (y_size - SAMPLE_SIZE)) + j] = hSrc(cv::Range(i, i + SAMPLE_SIZE), cv::Range(j, j + SAMPLE_SIZE));
		}
	}
	return imglist;
}

int computeSSD(cv::Mat& overlap1, cv::Mat& overlap2) {
	double sum = 0;

	cv::Mat gray1, gray2;

	cv::cvtColor(overlap1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(overlap2, gray2, cv::COLOR_BGR2GRAY);

	cv::Mat diff;
	cv::absdiff(gray1, gray2, diff);
	sum = cv::sum(diff)[0];

	return sum;
}

int computeHorizontalSSD(cv::Mat& topImg, cv::Mat& randImg) {
	if(topImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = topImg(cv::Range(topImg.rows-OVERLAP_SIZE, topImg.rows), cv::Range(0,topImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0, OVERLAP_SIZE), cv::Range(0,topImg.cols));
	return computeSSD(overlap1, overlap2);
}

int computeVerticalSSD(cv::Mat& prevImg, cv::Mat& randImg) {
	if(prevImg.dims == 0) {
		return 0;
	}
	cv::Mat overlap1 = prevImg(cv::Range(0,prevImg.rows),cv::Range(prevImg.cols-OVERLAP_SIZE,prevImg.cols));
	cv::Mat overlap2 = randImg(cv::Range(0,randImg.rows),cv::Range(0,OVERLAP_SIZE));
	return computeSSD(overlap1, overlap2);
}

int computeCombinedSSD(cv::Mat& prevImg, cv::Mat& topImg, cv::Mat& randImg) {
	double verticalSSD = computeVerticalSSD(prevImg, randImg);
	double horizontalSSD = computeHorizontalSSD(topImg, randImg);
	return verticalSSD + horizontalSSD;
}

cv::Mat getMinSSDImg(cv::Mat& prevImg, cv::Mat& topImg, std::vector<cv::Mat>& imglist) {
	int minSSD = 0;
	int minIdx = 0;
	for(int i = 0; i < imglist.size(); i++) {
		if(i == 0) {
			minSSD = computeCombinedSSD(prevImg, topImg, imglist[i]);
			minIdx = i;
		} else {
			int ssd = computeCombinedSSD(prevImg, topImg, imglist[i]);
			if(ssd < minSSD) {
				minSSD = ssd;
				minIdx = i;
			}
		}
	}
	//cout << minSSD << "\n" << endl;
	//cout << minIdx << endl;
	return imglist[minIdx];
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
	//cout << "inside placeImage" << x1 << ":" << x2 << ":" << y1 << ":" << y2 << ":" << endl;
	tile.copyTo(lImg(cv::Range(x1, x2), cv::Range(y1, y2)));
}

double findVerticalMinCost(int i, int j, cv::Mat& diff, map<pair<int, int >, double>& costMap, map<pair<int, int >, vector<pair<int, int > > >& pathMap) {
	double cost, costjp1, costj, costjm1, minimum;
	int cols = diff.cols;
	vector<pair<int, int > > list1, list2;
	list1.push_back(make_pair(i,j));
	if( i > 0) {
		if(j > 0 && j < cols - 1) {
			costjp1 = costMap[make_pair(i-1,j+1)] != -1 ? costMap[make_pair(i-1,j+1)] : findVerticalMinCost(i-1,j+1, diff,costMap, pathMap);
			costj = costMap[make_pair(i-1,j)] != -1 ? costMap[make_pair(i-1,j)] : findVerticalMinCost(i-1,j, diff,costMap, pathMap);
			costjm1 = costMap[make_pair(i-1,j-1)] != -1 ? costMap[make_pair(i-1,j-1)] : findVerticalMinCost(i-1,j-1, diff,costMap, pathMap);
			double costarr[] = {costjp1,costj,costjm1};
			minimum = *min_element(costarr,costarr+3);
			if(minimum == costjp1) {
				list2 = pathMap[make_pair(i-1,j+1)];
			} else if (minimum == costjm1) {
				list2 = pathMap[make_pair(i-1,j-1)];
			} else {
				list2 = pathMap[make_pair(i-1,j)];
			}
		} else if (j == 0) {
			costjp1 = costMap[make_pair(i-1,j+1)] != -1 ? costMap[make_pair(i-1,j+1)] : findVerticalMinCost(i-1,j+1, diff,costMap, pathMap);
			costj = costMap[make_pair(i-1,j)] != -1 ? costMap[make_pair(i-1,j)] : findVerticalMinCost(i-1,j, diff,costMap, pathMap);
			double costarr[] = {costjp1,costj};
			minimum = *min_element(costarr,costarr+2);
			if(minimum == costjp1) {
				list2 = pathMap[make_pair(i-1,j+1)];
			} else {
				list2 = pathMap[make_pair(i-1,j)];
			}
		} else if (j == cols - 1) {
			costj = costMap[make_pair(i-1,j)] != -1 ? costMap[make_pair(i-1,j)] : findVerticalMinCost(i-1,j, diff,costMap, pathMap);
			costjm1 = costMap[make_pair(i-1,j-1)] != -1 ? costMap[make_pair(i-1,j-1)] : findVerticalMinCost(i-1,j-1, diff,costMap, pathMap);
			double costarr[] = {costjp1,costj};
			minimum = *min_element(costarr,costarr+2);
			if(minimum == costjm1) {
				list2 = pathMap[make_pair(i-1,j-1)];
			} else {
				list2 = pathMap[make_pair(i-1,j)];
			}
		}
		list1.insert(list1.end(), list2.begin(), list2.end());
		cost = minimum + static_cast<double>( diff.at<uchar>(i,j) );
		costMap[make_pair(i,j)] = cost;
		pathMap[make_pair(i,j)] = list1;
	} else {
		cost = static_cast<double>( diff.at<uchar>(i,j) );
		costMap[make_pair(i,j)] = cost;
		pathMap[make_pair(i,j)].push_back(make_pair(i,j));
	}
	return cost;
}

double findHorizontalMinCost(int i, int j, cv::Mat& diff, map<pair<int, int >, double>& costMap, map<pair<int, int >, vector<pair<int, int > > >& pathMap) {
	double cost, costip1, costi, costim1, minimum;
	int rows = diff.rows;
	vector<pair<int, int > > list1, list2;
	list1.push_back(make_pair(i,j));
	if( j > 0) {
		if(i > 0 && i < rows - 1) {
			costip1 = costMap[make_pair(i+1,j-1)] != -1 ? costMap[make_pair(i+1,j-1)] : findHorizontalMinCost(i+1,j-1, diff,costMap, pathMap);
			costi = costMap[make_pair(i,j-1)] != -1 ? costMap[make_pair(i,j-1)] : findHorizontalMinCost(i,j-1, diff,costMap, pathMap);
			costim1 = costMap[make_pair(i-1,j-1)] != -1 ? costMap[make_pair(i-1,j-1)] : findHorizontalMinCost(i-1,j-1, diff,costMap, pathMap);
			double costarr[] = {costip1,costi,costim1};
			minimum = *min_element(costarr,costarr+3);
			if(minimum == costip1) {
				list2 = pathMap[make_pair(i+1,j-1)];
			} else if (minimum == costim1) {
				list2 = pathMap[make_pair(i-1,j-1)];
			} else {
				list2 = pathMap[make_pair(i,j-1)];
			}
		} else if (i == 0) {
			costip1 = costMap[make_pair(i+1,j-1)] != -1 ? costMap[make_pair(i+1,j-1)] : findHorizontalMinCost(i+1,j-1, diff,costMap, pathMap);
			costi = costMap[make_pair(i,j-1)] != -1 ? costMap[make_pair(i,j-1)] : findHorizontalMinCost(i,j-1, diff,costMap, pathMap);
			double costarr[] = {costip1,costi};
			minimum = *min_element(costarr,costarr+2);
			if(minimum == costip1) {
				list2 = pathMap[make_pair(i+1,j-1)];
			} else {
				list2 = pathMap[make_pair(i,j-1)];
			}
		} else if (i == rows - 1) {
			costi = costMap[make_pair(i,j-1)] != -1 ? costMap[make_pair(i,j-1)] : findHorizontalMinCost(i,j-1, diff,costMap, pathMap);
			costim1 = costMap[make_pair(i-1,j-1)] != -1 ? costMap[make_pair(i-1,j-1)] : findHorizontalMinCost(i-1,j-1, diff,costMap, pathMap);
			double costarr[] = {costip1,costi};
			minimum = *min_element(costarr,costarr+2);
			if(minimum == costim1) {
				list2 = pathMap[make_pair(i-1,j-1)];
			} else {
				list2 = pathMap[make_pair(i,j-1)];
			}
		}
		list1.insert(list1.end(), list2.begin(), list2.end());
		cost = minimum + static_cast<double>( diff.at<uchar>(i,j) );
		costMap[make_pair(i,j)] = cost;
		pathMap[make_pair(i,j)] = list1;
	} else {
		cost = static_cast<double>( diff.at<uchar>(i,j) );
		costMap[make_pair(i,j)] = cost;
		pathMap[make_pair(i,j)].push_back(make_pair(i,j));
	}
	return cost;
}


vector<pair<int,int> > findVerticalMinPath(cv::Mat& diff) {

	map<pair<int, int >, double> costMap;

	for(int i = 0; i < diff.rows; i++) {
		for(int j =0; j < diff.cols; j++) {
			costMap[make_pair(i,j)] = -1;
		}
	}

	map<pair<int, int >, vector<pair<int, int > > > pathMap;

	vector<double> mincost;
	vector<vector<pair<int,int> > > minpaths;

	int lastrow = diff.rows - 1;
	for(int i = 0 ; i < diff.cols; i++) {
		mincost.push_back(findVerticalMinCost(lastrow,i, diff,costMap, pathMap));
		minpaths.push_back(pathMap[make_pair(lastrow,i)]);
	}

	int minidx = find(minpaths.begin(), minpaths.end(), *min_element(minpaths.begin(),minpaths.end())) - minpaths.begin();

	vector<pair<int,int> > minpath = minpaths[minidx];

	for(int i = 0; i < minpath.size(); i++) {
		cout << minpath[i].first << " : " << minpath[i].second << endl;
	}
	return minpath;
}

vector<pair<int,int> > findHorizontalMinPath(cv::Mat& diff) {

	map<pair<int, int >, double> costMap;

	for(int j = 0; j < diff.cols; j++) {
		for(int i = 0; i < diff.rows; i++) {
			costMap[make_pair(i,j)] = -1;
		}
	}

	map<pair<int, int >, vector<pair<int, int > > > pathMap;

	vector<double> mincost;
	vector<vector<pair<int,int> > > minpaths;

	int lastcol = diff.cols - 1;
	for(int i = 0 ; i < diff.rows; i++) {
		mincost.push_back(findHorizontalMinCost(i,lastcol, diff,costMap, pathMap));
		minpaths.push_back(pathMap[make_pair(i,lastcol)]);
	}

	int minidx = find(minpaths.begin(), minpaths.end(), *min_element(minpaths.begin(),minpaths.end())) - minpaths.begin();

	vector<pair<int,int> > minpath = minpaths[minidx];

	for(int i = 0; i < minpath.size(); i++) {
		cout << minpath[i].first << " : " << minpath[i].second << endl;
	}
	return minpath;
}

void blendVertically(cv::Mat& currImg, cv::Mat& prevImg) {
	cv::Mat overlap1 = prevImg(cv::Range(0,prevImg.rows),cv::Range(prevImg.cols-OVERLAP_SIZE,prevImg.cols));
	cv::Mat overlap2 = currImg(cv::Range(0,currImg.rows),cv::Range(0,OVERLAP_SIZE));
	cv::Mat gray1, gray2;

	cv::cvtColor(overlap1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(overlap2, gray2, cv::COLOR_BGR2GRAY);
	cv::Mat diff;

	cv::absdiff(gray1, gray2, diff);

	vector<pair<int,int> > verticalPath;

	//cv::imwrite("testimg.jpg",currImg);
	cout << "diff\n" << diff << endl;
	verticalPath = findVerticalMinPath(diff);

	for(int i = 0; i < verticalPath.size(); i++) {
		int rowIdx = verticalPath[i].first;
		int colIdx = verticalPath[i].second;
		for(int j = 0; j < colIdx; j++) {
			cv::Vec3b& currPixel = overlap2.at<cv::Vec3b>(rowIdx,j);
			cv::Vec3b& prePixel = overlap1.at<cv::Vec3b>(rowIdx,j);
			currPixel[0] = prePixel[0];
			currPixel[1] = prePixel[1];
			currPixel[2] = prePixel[2];
		}
	}
}

void blendHorizontally(cv::Mat& currImg, cv::Mat& topImg) {
	cv::Mat overlap1 = topImg(cv::Range(topImg.rows-OVERLAP_SIZE,topImg.rows),cv::Range(0,topImg.cols));
	cv::Mat overlap2 = currImg(cv::Range(0,OVERLAP_SIZE),cv::Range(0,topImg.cols));
	cv::Mat gray1, gray2;

	cv::cvtColor(overlap1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(overlap2, gray2, cv::COLOR_BGR2GRAY);
	cv::Mat diff;

	cv::absdiff(gray1, gray2, diff);

	vector<pair<int,int> > horizontalPath;

	//cv::imwrite("testimg.jpg",currImg);
	cout << "diff\n" << diff << endl;
	horizontalPath = findHorizontalMinPath(diff);

	for(int i = 0; i < horizontalPath.size(); i++) {
		int rowIdx = horizontalPath[i].first;
		int colIdx = horizontalPath[i].second;
		for(int j = 0; j < rowIdx; j++) {
			cv::Vec3b& currPixel = overlap2.at<cv::Vec3b>(j,colIdx);
			cv::Vec3b& prePixel = overlap1.at<cv::Vec3b>(j,colIdx);
			currPixel[0] = prePixel[0];
			currPixel[1] = prePixel[1];
			currPixel[2] = prePixel[2];
		}
	}
}

void blendImage(cv::Mat& currImg, cv::Mat& prevImg, cv::Mat& topImg) {
	if (prevImg.dims > 0) {
		blendVertically(currImg, prevImg);
	}
	if(topImg.dims > 0) {
		blendHorizontally(currImg, topImg);
	}
}

void imageQuilting(cv::Mat& hSrc, cv::Mat& hDst) {



	std::vector<cv::Mat> imglist = createImageList(hSrc);

	int nx = OUTPUTX_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);
	int ny = OUTPUTY_SIZE/(SAMPLE_SIZE - OVERLAP_SIZE);

	for(int i = 0; i < nx; i++ ) {
		for(int j = 0; j < ny; j++) {

			cout << "i , j : " << i << " : " << j << endl;
			cv::Mat prevImg = getPreviousImg(i, j, hDst);

			cv::Mat topImg = getTopImg(i, j, hDst);

			clock_t start = clock();
			cv::Mat currImg = getMinSSDImg(prevImg, topImg, imglist);

			//cv::imwrite("testimg.jpg",currImg);
			clock_t end = clock();
			double elapsed_secs = double(end - start) * 1000 / CLOCKS_PER_SEC;

			cout << "Execution time:" << elapsed_secs <<"  ms"<< endl;

			//placeImg(i, j, currImg, hDst);

			//cv::imwrite("testimg.jpg",hDst);
			cv::Mat currImgCopy;
			currImg.copyTo(currImgCopy);
			blendImage(currImgCopy,prevImg,topImg);
			//cout << currImg << endl;
			placeImg(i, j, currImgCopy, hDst);
			//cv::imshow("hDst", hDst);
			//cv::waitKey();
			//cv::imwrite("testimg.jpg",hDst);
		}
	}

	//dSrc.copyTo(dDst);
	//dDst.download(hDst);
}

int main() {

	std::cout << "Hello World" << std::endl;
	std::string imageName = "image1.png";
	cv::Mat input = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);

	//cv::Mat inputGray = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);


	//std::cout << "gray :\n" << inputGray << endl;

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

	//cv::imshow("Output", output);

	//cv::waitKey();

	cv::imwrite("testimg.jpg",output);

	return 0;
}
