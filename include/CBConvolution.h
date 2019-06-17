# ifndef _CBCONVOLUTION_H
# define _CBCONVOLUTION_H

#include <cstring>
#include <iostream>
#include "TarFile.h"


class CBConvolution {
private:
	// Basic attr
	int cb_bits;
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
	bool compressed_weight = false;

	// Buffer
	unsigned char* weight_data = nullptr;
	float* codebook = nullptr;
	float* bias_data = nullptr;
	float* out_data = nullptr;

	// Helper function (INLINE)
	float get_in_data(float* data, int b, int c, int h, int w) {
		return data[b * in_chw + c * in_hw + h * in_w + w];
	}
	unsigned char extract_data(unsigned char data, unsigned char start, unsigned char end) {
		unsigned char bits = end - start;
		unsigned char offset = 8 - bits - start;
		unsigned char mask = ((1 << bits) - 1) << offset;

		return (data & mask) >> offset;
	}
	float get_weight_data(int oc, int ic, int kh, int kw) {
		int idx = oc * wt_chw + ic * wt_hw + kh * wt_w + kw;
		if (compressed_weight) {
			int byte_idx = idx * cb_bits / 8, offset = idx * cb_bits % 8;
			
			if (offset > 8 - cb_bits) {
				unsigned char d1 = extract_data(weight_data[byte_idx], offset, 8);
				unsigned char d2 = extract_data(weight_data[byte_idx + 1], 0, (cb_bits + offset - 8));
				return codebook[(d1 << (cb_bits + offset - 8)) | d2];
			} else {
				return codebook[extract_data(weight_data[byte_idx], offset, offset + cb_bits)];
			}
		} else {
			if(codebook != nullptr){
				return codebook[weight_data[idx]];
			} else{
				return weight_data[idx];
			}
		}
	}
	void set_out_data(int b, int c, int h, int w, float d) {
		out_data[b * out_chw + c * out_hw + h * out_w + w] = d;
	}
public:
	CBConvolution() {}
	CBConvolution(int kernel, int stride, int pad, int out_channels, int groups, int ishape[], bool with_relu = false, bool with_bias = false, int cb_bits = 8);
	~CBConvolution() {
		if (weight_data != nullptr)
			free(weight_data);
		if (bias_data != nullptr)
			free(bias_data);
	}

	float* Run(float* in_data, bool keep_in = false);
	int LoadWeightFromFile(const char* path, bool compressed_data = false, bool sparse_huffman = false, TarFile* tarfile = nullptr);
	int LoadBiasFromFile(const char* path, TarFile* tarfile = nullptr);
	int LoadCodebookFromFile(const char* path, TarFile* tarfile = nullptr);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 4); }
};

# endif    // ~~~_CBCONVOLUTION_H