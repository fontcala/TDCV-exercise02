#ifndef DT_H_
#define DT_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

template<int N>
class RandomForest{

private:
	std::vector<Ptr<ml::DTrees>> mRandomForest;
	Ptr<ml::TrainData> GenerateSubsetTrainData(const cv::Mat & outputTrain,const cv::Mat & outputTrainLabels);
	template<typename T>
	T ConsensusAcrossTrees(std::vector<T> input);
public:
	RandomForest(); //constructor
	void PrintNumberOfTrees() const;
	void Create(const int CVFolds, const int MaxCategories, const int MaxDepth, const int MinSampleCount);
	void Train(const cv::Mat & inputTrain,const cv::Mat & inputTrainLabels);
	cv::Mat Predict(const cv::Mat & inputTrain);
};

template<int N>
RandomForest<N>::RandomForest(){
	mRandomForest.reserve(N);
}

template<int N>
void RandomForest<N>::PrintNumberOfTrees() const{std::cout << N << std::endl;};

template<int N>
void RandomForest<N>::Create(const int CVFolds, const int MaxCategories, const int MaxDepth, const int MinSampleCount){
	for (int i = 0; i < N; ++i)
	{	
		Ptr<ml::DTrees> vTree = ml::DTrees::create();
    	vTree->setCVFolds(CVFolds);
    	vTree->setMaxCategories(MaxCategories);
    	vTree->setMaxDepth(MaxDepth);
    	vTree->setMinSampleCount(MinSampleCount);
		mRandomForest.push_back(vTree); 
	}
}
template<int N>
void RandomForest<N>::Train(const cv::Mat & inputTrain,const cv::Mat & inputTrainLabels){
	for(auto & vCurrentTree : mRandomForest){
		Ptr<ml::TrainData> vTrain = GenerateSubsetTrainData(inputTrain,inputTrainLabels);
		vCurrentTree->train(vTrain);
	}
}
template<int N>
cv::Mat RandomForest<N>::Predict(const cv::Mat & inputTrain){
	Mat allResults;
	for(auto & vCurrentTree : mRandomForest){
		if (!vCurrentTree->isTrained()){
			throw std::runtime_error("Predict(): Train the model first!");
		}
		Mat outTestResult;
    	vCurrentTree->predict(inputTrain,outTestResult);
    	allResults.push_back(outTestResult.t());
	}
	allResults = allResults.t();

    Mat outAggregatedTestResult(1,allResults.rows,CV_32F);
    //Find Consensus:
    for (int i = 0; i < allResults.rows; ++i)
    {
   		const float* p = allResults.ptr<float>(i);
   		std::vector<float> vector(p, p + allResults.cols);
   		const float sampleVote = ConsensusAcrossTrees(vector); // needs pass by value.
   		outAggregatedTestResult.at<float>(i) = sampleVote;
    }

    return outAggregatedTestResult;
}
template<int N>
Ptr<ml::TrainData> RandomForest<N>::GenerateSubsetTrainData(const cv::Mat & inputTrain,const cv::Mat & inputTrainLabels){
    std::vector <int> seeds;
    for (int cont = 0; cont < inputTrain.rows; cont++)
      seeds.push_back(cont);
    cv::randShuffle(seeds);
    const int subsetSize = static_cast<int>(static_cast<double>(inputTrain.rows)/3.0);
    Mat outputSubsetTrainLabels;
    Mat outputSubsetTrain;
    for (int cont = 0; cont < subsetSize; cont++){
     outputSubsetTrain.push_back(inputTrain.row(seeds[cont]));
     outputSubsetTrainLabels.push_back(inputTrainLabels.at<float>(seeds[cont]));
    }
    Ptr<ml::TrainData> vTrain = ml::TrainData::create(outputSubsetTrain,cv::ml::ROW_SAMPLE,outputSubsetTrainLabels);

   return vTrain;
}


template<int N>
template<typename T>
T RandomForest<N>::ConsensusAcrossTrees(std::vector<T> outAcrossTrees){
	std::sort(outAcrossTrees.begin(), outAcrossTrees.end()); 
	size_t n = outAcrossTrees.size();
	int max_count = 1, res = outAcrossTrees[0], curr_count = 1; 
    for (int i = 1; i < n; i++) { 
        if (outAcrossTrees[i] == outAcrossTrees[i - 1]) 
            curr_count++; 
        else { 
            if (curr_count > max_count) { 
                max_count = curr_count; 
                res = outAcrossTrees[i - 1]; 
            } 
            curr_count = 1; 
        } 
    } 
  
    // If last element is most frequent 
    if (curr_count > max_count) 
    { 
        max_count = curr_count; 
        res = outAcrossTrees[n - 1]; 
    }
    return res;
}


#endif //DT_H_

// for(auto & vCurrentTree : mRandomForest){
// 		std::cout << "eeh" << std::endl;
// 	}