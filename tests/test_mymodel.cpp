#include "Convolution.h"
#include "Eltwise.h"
#include "Concat.h"
#include "Pooling.h"
#include "FullyConnection.h"
#include "helper.h"
#include <iostream>

class Model {
private:
	Convolution* conv1, * conv2, * conv3;
	FullyConnection* fc1, * fc2;
	Pooling* pool;
	Eltwise* elt;
	Concat* cat;
public:
	Model() {
		int in_shape[] = { 1,3,224,224 };
		int shape[4], shape_[4];

		conv1 = new Convolution(3, 2, 1, 12, 1, in_shape, true, false);
		conv1->GetOutputShape(shape);
		conv2 = new Convolution(5, 1, 2, 12, 2, shape, true, true);
		conv2->GetOutputShape(shape);
		elt = new Eltwise("add", shape, false);
		elt->GetOutputShape(shape_);
		conv3 = new Convolution(1, 1, 0, 6, 1, shape, true, false);
		conv3->GetOutputShape(shape);
		cat = new Concat(shape_, shape, nullptr);
		cat->SetKeepOrNot(false, false);
		cat->GetOutputShape(shape);
		pool = new Pooling("max", 7, 7, 0, shape);
		pool->GetOutputShape(shape);
		fc1 = new FullyConnection(100, shape, true, false, true);
		fc1->GetOutputShape(shape);
		fc2 = new FullyConnection(10, shape, false, true, false);

		conv1->LoadWeightFromFile("../models/my_testmodel/conv1.weight.dat");
		conv2->LoadWeightFromFile("../models/my_testmodel/conv2.weight.dat");
		conv2->LoadBiasFromFile("../models/my_testmodel/conv2.bias.dat");
		conv3->LoadWeightFromFile("../models/my_testmodel/conv3.weight.dat");
		fc1->LoadWeightFromFile("../models/my_testmodel/fc1.weight.dat");
		fc2->LoadWeightFromFile("../models/my_testmodel/fc2.weight.dat");
		fc2->LoadBiasFromFile("../models/my_testmodel/fc2.bias.dat");
	}
	~Model() {
		delete conv1, conv2, conv3;
		delete fc1, fc2;
		delete elt;
		delete cat;
	}
	float* Run(float* x) {
			int shape[4];
		float* x_ = conv1->Run(x);
			conv1->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/conv1.output.dat", (char*)x_, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x = conv2->Run(x_, true);
			conv2->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/conv2.output.dat", (char*)x, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x = elt->Run(x_, x);
			elt->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/elt.output.dat", (char*)x, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x_ = conv3->Run(x, true);
			conv3->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/conv3.output.dat", (char*)x_, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x = cat->Run(x, x_);	// clear all inputs
			cat->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/cat.output.dat", (char*)x, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x = pool->Run(x);
			pool ->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/pool.output.dat", (char*)x, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
		x = fc1->Run(x);
			fc1->GetOutputShape(shape);
			SaveToFile("../outputs/my_testmodel/fc1.output.dat", (char*)x, shape[0] * shape[1] * sizeof(float));
		x = fc2->Run(x);
		return x;
	}
	void GetOutputShape(int shape[]) {
		fc2->GetOutputShape(shape);
	}
};

void test_mymodel() {
	float* input = (float*)malloc(sizeof(float) * 1 * 3 * 224 * 224);
	LoadFromFile("../models/my_testmodel/input.dat", (char*)input);

	Model model;
	float* output;
	output = model.Run(input);

	int out_shape[4];
	unsigned long size;
	model.GetOutputShape(out_shape);
	// size = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
	size = out_shape[0] * out_shape[1];
	SaveToFile("../outputs/my_testmodel/output.dat", (char*)output, size * sizeof(float));
}

int main() {
	test_mymodel();
	return 0;
}
