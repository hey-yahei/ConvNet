#ifndef _ZTE_AI_H
#define _ZTE_AI_H

#include "Eltwise.h"
#include "CBConvolution.h"
#include "Convolution.h"
#include "CBFullyConnection.h"
#include <vector>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ZTE_AI {
private:
	int in_shape[4] = { 1,3,128,128 };
	Convolution* conv1;
	CBConvolution* conv2a_1, * conv2a_2a, * conv2a_2b, * conv2b_2a, * conv2b_2b;
	CBConvolution* conv3a_1, * conv3a_2a, * conv3a_2b, * conv3b_2a, * conv3b_2b;
	CBConvolution* conv4a_1, * conv4a_2a, * conv4a_2b, * conv4b_2a, * conv4b_2b;
	CBConvolution* conv5a_1, * conv5a_2a, * conv5a_2b, * conv5b_2a, * conv5b_2b;

	Eltwise* res_2a, * res_2b, * res_3a, * res_3b, * res_4a, * res_4b, * res_5a, * res_5b;

	CBFullyConnection* fc;

	void Build();
	void Load(string model_dir);

public:
	ZTE_AI(string model_dir){
		Build();
		Load(model_dir);
	}
	~ZTE_AI(){
		delete conv1;
		delete conv2a_1, conv2a_2a, conv2a_2b, conv2b_2a, conv2b_2b;
		delete conv3a_1, conv3a_2a, conv3a_2b, conv3b_2a, conv3b_2b;
		delete conv4a_1, conv4a_2a, conv4a_2b, conv4b_2a, conv4b_2b;
		delete conv5a_1, conv5a_2a, conv5a_2b, conv5b_2a, conv5b_2b;

		delete res_2a, res_2b, res_3a, res_3b, res_4a, res_4b, res_5a, res_5b;

		delete fc;
	}

	void GetOutputShape(int shape[]) {
		fc->GetOutputShape(shape);
	}
	float* Run(float* x);
	vector<float> GetFeature(Mat &image);
};

#endif  // ~~_ZTE_AI_H
