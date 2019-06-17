#include "FullyConnection.h"
#include "helper.h"
#include <iostream>

using namespace std;

FullyConnection::FullyConnection(int out_num, int ishape[], bool with_relu, bool wiht_bias, bool flatten) 
		: out_num(out_num), with_relu(with_relu), with_bias(with_bias){
	if (flatten) {
		in_shape[0] = ishape[0];
		in_shape[1] = ishape[1] * ishape[2] * ishape[3];
	} else {
		in_shape[0] = ishape[0];
		in_shape[1] = ishape[1];
	}

	out_shape[0] = ishape[0];
	out_shape[1] = out_num;

	weight_data = (float*)malloc(sizeof(float) * in_shape[1] * out_shape[1]);
	if (wiht_bias)
		bias_data = (float*)malloc(sizeof(float) * out_num);
}

float* FullyConnection::Run(float* in_data, bool keep_in) {
	float dat, wt, result;

	out_data = (float*)malloc(sizeof(float) * out_shape[0] * out_shape[1]);

	for (int idx_bs = 0; idx_bs < out_shape[0]; idx_bs++) {
		for (int idx_out = 0; idx_out < out_shape[1]; idx_out++) {
			result = 0.;
			for (int idx_in = 0; idx_in < in_shape[1]; idx_in++) {
				dat = in_data[idx_bs * in_shape[0] + idx_in];
				wt = weight_data[idx_out * in_shape[1] + idx_in];
				result += dat * wt;
			}
			if (with_bias)
				result += bias_data[idx_out];
			if (with_relu && result < 0)
				result = 0;
			out_data[idx_bs * out_shape[1] + idx_out] = result;
		}
	}

	if (!keep_in)
		free(in_data);

	return out_data;
}

int FullyConnection::LoadWeightFromFile(const char* path) {
	unsigned long cnt = LoadFromFile(path, (char*)weight_data);

	cnt /= sizeof(float);
	return (cnt == in_shape[1] * out_shape[1]) ? cnt : -1;
}

int FullyConnection::LoadBiasFromFile(const char* path) {
	unsigned long cnt = LoadFromFile(path, (char*)bias_data);

	cnt /= sizeof(float);
	return (cnt == out_shape[1]) ? cnt : -1;
}