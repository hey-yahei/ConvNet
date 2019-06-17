# ifndef _CONVOLUTION_H
# define _CONVOLUTION_H

#include <cstring>
#include <iostream>


class Convolution{
private:
	// Basic attr
	int kernel;
	int stride;
	int pad;
	int in_channels;
	int out_channels;

	// Group
	int groups;
	int ic_per_gp;
	int oc_per_gp;

	// Shape
	int in_shape[4];
	int out_shape[4];

	// Offset
	int in_chw, in_hw, in_w;
	int wt_chw, wt_hw, wt_w;
	int out_chw, out_hw, out_w;

	// Flag
	bool with_relu;
	bool with_bias;

	// Buffer
	float* weight_data = nullptr;
	float* bias_data = nullptr;
	float* out_data = nullptr;

	// Helper function (INLINE)
	float get_in_data(float* data, int b, int c, int h, int w) {
		return data[b * in_chw + c * in_hw + h * in_w + w];
	}
	float get_weight_data(int oc, int ic, int kh, int kw) {
		return weight_data[oc * wt_chw + ic * wt_hw + kh * wt_w + kw];
	}
	void set_out_data(int b, int c, int h, int w, float d) {
		out_data[b * out_chw + c * out_hw + h * out_w + w] = d;
	}
public:
	Convolution(){}
	Convolution(int kernel, int stride, int pad, int out_channels, int groups, int ishape[], bool with_relu = false, bool with_bias = false);
	~Convolution() { 
		if (weight_data != nullptr) 
			free(weight_data); 
		if (bias_data != nullptr)
			free(bias_data);
	}

	float* Run(float* in_data, bool keep_in = false);
	int LoadWeightFromFile(const char* path);
	int LoadBiasFromFile(const char* path);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 4); }
};

# endif    // ~~~_CONVOLUTION_H