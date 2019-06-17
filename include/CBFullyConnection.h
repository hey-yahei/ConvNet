#ifndef _CBFULLY_CONNECTION_H
#define _CBFULLY_CONNECTION_H

#include <iostream>
#include <cstring>

class CBFullyConnection {
private:
	int out_num;
	int cb_bits;

	// Shape
	int in_shape[2];
	int out_shape[2];

	// Flag
	bool with_relu;
	bool with_bias;
	bool compressed_weight = false;

	// Buffer
	unsigned char* weight_data = nullptr;
	float* codebook = nullptr;
	float* bias_data = nullptr;
	float* out_data = nullptr;

	// Helper
	unsigned char extract_data(unsigned char data, unsigned char start, unsigned char end) {
		unsigned char bits = end - start;
		unsigned char offset = 8 - bits - start;
		unsigned char mask = ((1 << bits) - 1) << offset;

		return (data & mask) >> offset;
	}
	float get_weight_data(int o, int i) {
		int idx = o * in_shape[1] + i;
		if (compressed_weight) {
			int byte_idx = idx * cb_bits / 8, offset = idx * cb_bits % 8;

			if (offset > 8 - cb_bits) {
				unsigned char d1 = extract_data(weight_data[byte_idx], offset, 8);
				unsigned char d2 = extract_data(weight_data[byte_idx + 1], 0, (cb_bits + offset - 8));
				return codebook[(d1 << (cb_bits + offset - 8)) | d2];
			} else {
				if(codebook != nullptr){
				   return codebook[extract_data(weight_data[byte_idx], offset, offset + cb_bits)];
				}
			}
		} else {
			if(codebook != nullptr){
				return codebook[weight_data[idx]];
			}else{
				return weight_data[idx];
			}
		}
	}
public:
	CBFullyConnection() {}
	CBFullyConnection(int out_num, int ishape[], bool with_relu = false, bool with_bias = false, bool flatten = false, int cb_bits = 8);
	~CBFullyConnection() {
		if (weight_data != nullptr)
			free(weight_data);
		if (bias_data != nullptr)
			free(bias_data);
		if (codebook != nullptr)
			free(codebook);
	}

	float* Run(float* in_data, bool keep_in = false);
	int LoadWeightFromFile(const char* path, bool compressed_data = false, bool sparse_huffman = false);
	int LoadBiasFromFile(const char* path);
	int LoadCodebookFromFile(const char* path);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 2); }
};

#endif   // ~~~_CBFULLY_CONNECTION_H