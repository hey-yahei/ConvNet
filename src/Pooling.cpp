#include "Pooling.h"
#include <iostream>
#include <string>

using namespace std;


Pooling::Pooling(const char* type, int kernel, int stride, int pad, int ishape[])
		: type(type), kernel(kernel), stride(stride), pad(pad), in_channels(ishape[1]){
	for (int i = 0; i < 4; i++)
		in_shape[i] = ishape[i];
	in_w = ishape[3];
	in_hw = ishape[2] * in_w;
	in_chw = ishape[1] * in_hw;

	out_shape[0] = ishape[0];		// batch size
	out_shape[1] = in_channels;	// out_channels
	out_shape[2] = (ishape[2] + 2 * pad - kernel) / stride + 1;   // out_height
	out_shape[3] = (ishape[3] + 2 * pad - kernel) / stride + 1;   // out_width
	out_w = out_shape[3];
	out_hw = out_shape[2] * out_w;
	out_chw = out_shape[1] * out_hw;
}


float* Pooling::Run(float* in_data, bool keep_in) {
	int start_h, start_w;
	int idx_ih, idx_iw;
	float dat, result;

	out_data = (float*)malloc(sizeof(float) * in_shape[0] * out_chw);

	// Do convolution
	for (int idx_bs = 0; idx_bs < in_shape[0]; idx_bs++) {
		for (int idx_oc = 0; idx_oc < out_shape[1]; idx_oc++) {
			for (int idx_oh = 0; idx_oh < out_shape[2]; idx_oh++) {
				for (int idx_ow = 0; idx_ow < out_shape[3]; idx_ow++) {
					start_h = idx_oh * stride - pad;
					start_w = idx_ow * stride - pad;
					result = 0.;
					for (int idx_kh = 0; idx_kh < kernel; idx_kh++) {
						for (int idx_kw = 0; idx_kw < kernel; idx_kw++) {
							idx_ih = start_h + idx_kh;
							idx_iw = start_w + idx_kw;
							if ((idx_ih < 0) || (idx_iw < 0) || (idx_ih >= in_shape[2]) || (idx_iw >= in_shape[3]))
								dat = 0.;
							else
								dat = get_in_data(in_data, idx_bs, idx_oc, idx_ih, idx_iw);
							if (!strcmp(type, "max")) {
								if (result < dat)
									result = dat;
							} else if(!strcmp(type, "avg")) {
								result += dat / kernel / kernel;
							} else {
								cerr << "Pooling: unknown type" << endl;
								exit(1);
							}
						}
					}
					set_out_data(idx_bs, idx_oc, idx_oh, idx_ow, result);
				}
			}
		}
	}

	if (!keep_in)
		free(in_data);

	return out_data;
}
