#include "Convolution.h"
#include "helper.h"
#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;


Convolution::Convolution(int kernel, int stride, int pad, int out_channels, int groups, int ishape[], bool with_relu, bool with_bias)
		: kernel(kernel), stride(stride), pad(pad), in_channels(ishape[1]), 
	      out_channels(out_channels), groups(groups), with_relu(with_relu), with_bias(with_bias){
	if (in_channels % groups != 0) {
		cerr << "in_channels % groups != 0" << endl;
		exit(1);
	}
	if (out_channels % groups != 0) {
		cerr << "out_channels % groups != 0" << endl;
		exit(1);
	}

	ic_per_gp = in_channels / groups;
	oc_per_gp = out_channels / groups;

	memcpy(in_shape, ishape, sizeof(int) * 4);
	in_w = ishape[3];
	in_hw = ishape[2] * in_w;
	in_chw = ishape[1] * in_hw;

	wt_w = kernel;
	wt_hw = kernel * wt_w;
	wt_chw = ic_per_gp * wt_hw;

	out_shape[0] = ishape[0];		// batch size
	out_shape[1] = out_channels;	// out_channels
	out_shape[2] = (ishape[2] + 2 * pad - kernel) / stride + 1;   // out_height
	out_shape[3] = (ishape[3] + 2 * pad - kernel) / stride + 1;   // out_width
	out_w = out_shape[3];
	out_hw = out_shape[2] * out_w;
	out_chw = out_shape[1] * out_hw;

	weight_data = (float*)malloc(sizeof(float) * out_channels * ic_per_gp * kernel * kernel);
	if (with_bias)
		bias_data = (float*)malloc(sizeof(float) * out_channels);
}


float* Convolution::Run(float* in_data, bool keep_in){
	int idx_gp;
	int start_h, start_w;
	int idx_ih, idx_iw;
	float wt, dat, result;

	out_data = (float*)malloc(sizeof(float) * in_shape[0] * out_chw);

	// Do convolution
	for (int idx_bs = 0; idx_bs < in_shape[0]; idx_bs++) {
		for (int idx_oc = 0; idx_oc < out_shape[1]; idx_oc++) {
			idx_gp = idx_oc / oc_per_gp;
			for (int idx_oh = 0; idx_oh < out_shape[2]; idx_oh++) {
				for (int idx_ow = 0; idx_ow < out_shape[3]; idx_ow++) {
					start_h = idx_oh * stride - pad;
					start_w = idx_ow * stride - pad;
					result = 0;
					for (int idx_ic = ic_per_gp * idx_gp; idx_ic < ic_per_gp * (idx_gp + 1); idx_ic++) {
						for (int idx_kh = 0; idx_kh < kernel; idx_kh++) {
							for (int idx_kw = 0; idx_kw < kernel; idx_kw++) {
								idx_ih = start_h + idx_kh;
								idx_iw = start_w + idx_kw;
								if ((idx_ih < 0) || (idx_iw < 0) || (idx_ih >= in_shape[2]) || (idx_iw >= in_shape[3]))
									dat = 0.;
								else
									dat = get_in_data(in_data, idx_bs, idx_ic, idx_ih, idx_iw);
								wt = get_weight_data(idx_oc, idx_ic - idx_gp * ic_per_gp, idx_kh, idx_kw);
								result += dat * wt;
							}
						}
					}
					if (with_bias)
						result += bias_data[idx_oc];
					if (with_relu && result < 0)
						result = 0;
					set_out_data(idx_bs, idx_oc, idx_oh, idx_ow, result);
				}
			}
		}
	}

	if (!keep_in)
		free(in_data);

	return out_data;
}

int Convolution::LoadWeightFromFile(const char* path) {
	unsigned long cnt = LoadFromFile(path, (char *)weight_data);

	cnt /= sizeof(float);
	return (cnt == out_channels * ic_per_gp * kernel * kernel) ? cnt : -1;
}

int Convolution::LoadBiasFromFile(const char* path) {
	unsigned long cnt = LoadFromFile(path, (char*)bias_data);

	cnt /= sizeof(float);
	return (cnt == out_channels) ? cnt : -1;
}
