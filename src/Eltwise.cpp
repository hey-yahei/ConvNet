#include "Eltwise.h"
#include <iostream>
#include <string>

#define MAX(a,b) ((a>b)?a:b)

using namespace std;

Eltwise::Eltwise(const char* t, int ishape[], bool with_relu) : with_relu(with_relu) {
	elt_t = t;

	memcpy(in_shape, ishape, sizeof(int) * 4);
	memcpy(out_shape, ishape, sizeof(int) * 4);

	out_w = ishape[3];
	out_hw = ishape[2] * out_w;
	out_chw = ishape[1] * out_hw;
}

float* Eltwise::Run(float* in1_data, float* in2_data, bool keep_in1, bool keep_in2) {
	int elt_idx;

	if (!keep_in1) {
		out_data = in1_data;
		keep_in1 = true;
	} else if (!keep_in2) {   // keep_in1 == true
		out_data = in2_data;
		in2_data = in1_data;  // now, in2 -> in1
		keep_in2 = true;
	} else {
		out_data = (float*)malloc(sizeof(float) * out_chw * out_shape[0]);
		memcpy(out_data, in1_data, sizeof(float) * out_chw * out_shape[0]);
	}

	for (int idx_bs = 0; idx_bs < out_shape[0]; idx_bs++) {
		for (int idx_c = 0; idx_c < out_shape[1]; idx_c++) {
			for (int idx_h = 0; idx_h < out_shape[2]; idx_h++) {
				for (int idx_w = 0; idx_w < out_shape[3]; idx_w++) {
					elt_idx = idx_bs * out_chw + idx_c * out_hw + idx_h * out_w + idx_w;
					if (!strcmp(elt_t, "add")) {
						out_data[elt_idx] += in2_data[elt_idx];
						if (with_relu && out_data[elt_idx] < 0.)
							out_data[elt_idx] = 0.;
					} else if (!strcmp(elt_t, "max")){
						out_data[elt_idx] = MAX(out_data[elt_idx], in2_data[elt_idx]);
						if (with_relu && out_data[elt_idx] < 0.)
							out_data[elt_idx] = 0.;
					} else {
						cerr << "Eltwise: unknown type" << endl;
						exit(1);
					}
				}
			}
		}
	}

	if (!keep_in1)
		free(in1_data);
	if (!keep_in2)
		free(in2_data);

	return out_data;
}
