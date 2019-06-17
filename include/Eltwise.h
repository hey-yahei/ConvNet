#ifndef _ELTWISE_H
#define _ELTWISE_H

#include <iostream>
#include <string>
#include <cstring>

class Eltwise{
private:
	// Basic attr
	const char* elt_t;

	// Shape
	int in_shape[4];
	int out_shape[4];

	// Offset
	int out_w, out_hw, out_chw;

	// Buffer
	float* out_data = nullptr;

	// Flag
	bool with_relu;

	// Helper
	void set_out_data(int b, int c, int h, int w, float d) {
		out_data[b * out_chw + c * out_hw + h * out_w + w] = d;
	}

public:
	Eltwise(){}
	~Eltwise(){}
	Eltwise(const char* t, int in_shape[], bool with_relu = false);

	float* Run(float* in1_data, float* in2_data, bool keep_in1 = false, bool keep_in2 = false);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 4); }
};

#endif   // ~~~ELTWISE_H