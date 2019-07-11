#ifndef _FULLY_CONNECTION_H
#define _FULLY_CONNECTION_H

#include <iostream>
#include <cstring>

class FullyConnection{
private:
	int out_num;

	// Shape
	int in_shape[2];
	int out_shape[2];

	// Flag
	bool with_relu;
	bool with_bias;

	// Buffer
	float* weight_data = nullptr;
	float* bias_data = nullptr;
	float* out_data = nullptr;
public:
	FullyConnection(){}
	FullyConnection(int out_num, int ishape[], bool with_relu = false, bool wiht_bias = false, bool flatten = false);
	~FullyConnection(){
		if (weight_data != nullptr)
			free(weight_data);
		if (bias_data != nullptr)
			free(bias_data);
	}

	float* Run(float* in_data, bool keep_in = false);
	int LoadWeightFromFile(const char* path);
	int LoadBiasFromFile(const char* path);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 2); }
};

#endif   // ~~~_FULLY_CONNECTION_H