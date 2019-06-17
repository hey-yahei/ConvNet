#ifndef _POOLING_H
#define _POOLING_H

#include <string>

using namespace std;

class Pooling{
private:
	// Basic attr
	int kernel;
	int stride;
	int pad;
	int in_channels;

	// Shape
	int in_shape[4];
	int out_shape[4];

	// Offset
	int in_chw, in_hw, in_w;
	int out_chw, out_hw, out_w;

	// Buffer
	float* out_data = nullptr;

	// Type
	const char* type;

	// Helper function (INLINE)
	float get_in_data(float* data, int b, int c, int h, int w) {
		return data[b * in_chw + c * in_hw + h * in_w + w];
	}
	void set_out_data(int b, int c, int h, int w, float d) {
		out_data[b * out_chw + c * out_hw + h * out_w + w] = d;
	}

public:
	Pooling(){}
	Pooling(const char* type, int kernel, int stride, int pad, int ishape[]);
	~Pooling(){}

	float* Run(float* data, bool keep_in = false);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 4); }
};

#endif    // ~~~_POOLING_H