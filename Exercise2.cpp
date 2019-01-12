#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include "hog_visualization.h"
#include "data_helper.h"
#include "RandomForest.h"
using namespace cv;

int Task1(){
    Mat image;
    image = imread("../data/task1/obj1000.jpg", 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    resize(image, image, Size(256,256),0,0, INTER_NEAREST);
    const int imWidth = image.cols;
    const int imHeight = image.rows;
    const Size winSize(imWidth,imHeight);
    const Size blockSize(64,64);
    const Size blockStride(64,64);
    const Size cellSize(64,64);
    const int nbins(8);
    HOGDescriptor hogDesc(winSize, blockSize, blockStride, cellSize, nbins);
    std::vector<float> descriptors;
    hogDesc.compute(image,descriptors);
    visualizeHOG(image, descriptors,hogDesc,1);
    return 0; 
}
int Task2(){

    // Image Read
    Mat outTrain;
    Mat outTrainLabels;
    Mat outTest;
    Mat outTestLabelsTrue;
    prepareData(outTrain,outTrainLabels,outTest,outTestLabelsTrue);

    // DTree
    Ptr<ml::DTrees> vTree = ml::DTrees::create();
    vTree->setCVFolds(1);
    vTree->setMaxCategories(6);
    vTree->setMaxDepth(16);
    vTree->setMinSampleCount(10);
    Ptr<ml::TrainData> vTrain  = ml::TrainData::create(outTrain,cv::ml::ROW_SAMPLE,outTrainLabels);
    vTree->train(vTrain);

    Mat outTestResult;
    vTree->predict(outTest,outTestResult);

    // RandomForest
    RandomForest<30> vRF;
    vRF.Create(1,6,16,10);
    vRF.Train(outTrain, outTrainLabels);

    Mat outTestAggregatedResult;
    outTestAggregatedResult = vRF.Predict(outTest);


    //Stats:
    double vHitCountTree = 0;
    double vHitCountRF = 0;
    for (int sampleIdx = 0; sampleIdx < outTest.rows; ++sampleIdx)
    {
       if (std::abs(outTestLabelsTrue.at<float>(sampleIdx) - outTestResult.at<float>(sampleIdx)) < 0.5) {
            ++vHitCountTree;
       }
       if (std::abs(outTestLabelsTrue.at<float>(sampleIdx) - outTestAggregatedResult.at<float>(sampleIdx)) < 0.5) {
            ++vHitCountRF;
       }
    }
    std::cout << "-- Task 2 stats--" << std::endl;
    std:: cout << "Number of Trees:"; 
    vRF.PrintNumberOfTrees();
    std::cout << "Accuracy Tree: " << vHitCountTree/static_cast<double>(outTest.rows) << std::endl;
    std::cout << "Accuracy RF: " << vHitCountRF/static_cast<double>(outTest.rows) << std::endl;
    std::cout << "outTrain.rows: " << outTrain.rows << std::endl;
    std::cout << "outTrain.cols: " << outTrain.cols << std::endl;
    std::cout << "outTest.rows: " << outTest.rows << std::endl;
    std::cout << "outTest.cols: " << outTest.cols << std::endl;

    return 0;
}
int main()
{
    Task1();
    Task2();
    return 0;


}
