#ifndef DH_H_
#define DH_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/objdetect.hpp>
using namespace cv;

void dummy();

void prepareData(cv::Mat & outputTrain,cv::Mat & outputTrainLabels,cv::Mat & outputTest,cv::Mat & outputTestLabels){
	// Hardcoded number of files per classes. Ideally would be using boost::filesystem or c++17 filesystem.
	const std::vector<size_t> cTrainTotals = {49,67,42,53,67,110};
	const std::vector<size_t> cTestTotals = {59,77,52,63,77,120};
	const size_t imWidth = 256;
    const size_t imHeight = 256;
    const Size winSize(imWidth,imHeight);
    const Size blockSize(64,64);
    const Size blockStride(32,32);
    const Size cellSize(32,32);
    const int nbins(8);
	for(int currentFolder = 0; currentFolder < 6; currentFolder++){
		for(size_t currentElement = 0; currentElement < cTrainTotals[currentFolder]; ++currentElement){
			// elems smaller than 10
			std::string currentImage;
			if(currentElement < 10){
			currentImage = "../data/task2/train/0" + std::to_string(currentFolder) + "/000" + std::to_string(currentElement) + ".jpg";
			}
			else if(currentElement > 99){
			currentImage = "../data/task2/train/0" + std::to_string(currentFolder) + "/0" + std::to_string(currentElement) + ".jpg";
			}
			else{
			currentImage = "../data/task2/train/0" + std::to_string(currentFolder) + "/00" + std::to_string(currentElement) + ".jpg";
			}
			Mat image = imread(currentImage, 1 );
			resize(image, image, Size(256,256),0,0, INTER_NEAREST);
			HOGDescriptor hogDesc(winSize, blockSize, blockStride, cellSize, nbins);
    		std::vector<float> descriptors;
    		hogDesc.compute(image,descriptors);
    		Mat matDescriptors(descriptors);
    		outputTrain.push_back(matDescriptors.t());
    		outputTrainLabels.push_back(static_cast<float>(currentFolder));
		}
		for(size_t currentElement = cTrainTotals[currentFolder]; currentElement < cTestTotals[currentFolder]; ++currentElement){
			// elems smaller than 10
			std::string currentImage;
			if(currentElement > 99){
			currentImage = "../data/task2/test/0" + std::to_string(currentFolder) + "/0" + std::to_string(currentElement) + ".jpg";
			}
			else{
			currentImage = "../data/task2/test/0" + std::to_string(currentFolder) + "/00" + std::to_string(currentElement) + ".jpg";
			}
			Mat image = imread(currentImage, 1 );
			resize(image, image, Size(256,256),0,0, INTER_NEAREST);
			HOGDescriptor hogDesc(winSize, blockSize, blockStride, cellSize, nbins);
    		std::vector<float> descriptors;
    		hogDesc.compute(image,descriptors);
    		Mat matDescriptors(descriptors);
    		outputTest.push_back(matDescriptors.t());
    		outputTestLabels.push_back(static_cast<float>(currentFolder));
		}
	}
};

#endif //DH_H_