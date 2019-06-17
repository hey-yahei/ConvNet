#include "CBFullyConnection.h"
#include "helper.h"
#include <iostream>
#include <math.h>

using namespace std;

CBFullyConnection::CBFullyConnection(int out_num, int ishape[], bool with_relu, bool with_bias, bool flatten, int cb_bits)
	: out_num(out_num), with_relu(with_relu), with_bias(with_bias), cb_bits(cb_bits) {
	if (flatten) {
		in_shape[0] = ishape[0];
		in_shape[1] = ishape[1] * ishape[2] * ishape[3];
	}
	else {
		in_shape[0] = ishape[0];
		in_shape[1] = ishape[1];
	}

	out_shape[0] = ishape[0];
	out_shape[1] = out_num;
}

float* CBFullyConnection::Run(float* in_data, bool keep_in) {
	float dat, wt, result;

	out_data = (float*)malloc(sizeof(float) * out_shape[0] * out_shape[1]);

	for (int idx_bs = 0; idx_bs < out_shape[0]; idx_bs++) {
		for (int idx_out = 0; idx_out < out_shape[1]; idx_out++) {
			result = 0.;
			for (int idx_in = 0; idx_in < in_shape[1]; idx_in++) {
				dat = in_data[idx_bs * in_shape[0] + idx_in];
				// wt = weight_data[idx_out * in_shape[1] + idx_in];
				wt = get_weight_data(idx_out, idx_in);
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

int CBFullyConnection::LoadWeightFromFile(const char* path, bool compressed_data, bool sparse_huffman) {
	unsigned long cnt;
	compressed_weight = compressed_data;

	/*if (compressed_data && sparse_huffman) {
		cerr << "CBConvolution: compressed=true and huffman=true" << endl;
		exit(1);
	}*/

	if (weight_data != nullptr)
		free(weight_data);
		if (sparse_huffman) {
			weight_data = (unsigned char*)malloc(sizeof(unsigned char) * in_shape[1] * out_shape[1]);
			if (nullptr == codebook) {
				cerr << "LoadWeightFromFile: codebook should be set before load weight with sparse huffman" << endl;
				exit(1);
			}

			int zero_label = -1;
			for (int i = 0; i < (1 << cb_bits); i++) {
				if (codebook[i] == 0) {
					zero_label = i;
					break;
				}
			}
			if (-1 == zero_label) {
				cerr << "LoadWeightFromFile: zero_label not found" << endl;
				exit(1);
			}

			memset(weight_data, zero_label, sizeof(unsigned char) * in_shape[1] * out_shape[1]);
			cnt = LoadFromSparseHuffmanFile(path, (char*)weight_data);
			if(compressed_data){
				int size1 = in_shape[1] * out_shape[1];
				int size2 = ceil(size1 * cb_bits / 8.);
				unsigned char *new_data = (unsigned char*)malloc(sizeof(unsigned char) * size2);
				memset(new_data, 0, sizeof(unsigned char) * size2);

				unsigned char *ptr = new_data;
				unsigned char offset = 0;
				unsigned char tmp;
				for(int i = 0; i < size1; i++){
					if(offset > 8 - cb_bits){
						unsigned char mask = ((1 << (8 - offset)) - 1) << (offset + cb_bits - 8);
						unsigned char d1 = weight_data[i] & mask;
						mask = (1 << (offset + cb_bits - 8)) - 1;
						unsigned char d2 = weight_data[i] & mask;
						
						*ptr |= d1;
						ptr++;
						*ptr |= d2 << (16 - offset - cb_bits);
						offset = offset + cb_bits - 8;
					}else{
						unsigned char mask = (1 << cb_bits) - 1;
						*ptr |= (weight_data[i] << 8 - offset - cb_bits);
						offset += cb_bits;
						if(offset == 8) 
						    offset = 0;
					}
				}
				free(weight_data);
				weight_data = new_data;
				return (cnt == ceil(in_shape[1] * out_shape[1] * cb_bits / 8.)) ? cnt : -1;
			}else{
				return (cnt == in_shape[1] * out_shape[1]) ? cnt : -1;
			}
		} else if (compressed_data) {
		    weight_data = (unsigned char*)malloc(sizeof(unsigned char) * ceil(in_shape[1] * out_shape[1] * cb_bits / 8.));
		    cnt = LoadFromFile(path, (char*)weight_data);
		    return (cnt == ceil(in_shape[1] * out_shape[1] * cb_bits / 8.)) ? cnt : -1;
	    } else {
			weight_data = (unsigned char*)malloc(sizeof(unsigned char) * in_shape[1] * out_shape[1]);
			cnt = LoadFromFile(path, (char*)weight_data);
			return (cnt == in_shape[1] * out_shape[1]) ? cnt : -1;
		}
}

int CBFullyConnection::LoadBiasFromFile(const char* path) {
	if (!with_bias) {
		cerr << "CBFullyConnection: without bias but try to load" << endl;
		exit(1);
	}
	if (bias_data != nullptr)
		free(bias_data);
	bias_data = (float*)malloc(sizeof(float) * out_num);

	unsigned long cnt = LoadFromFile(path, (char*)bias_data);

	cnt /= sizeof(float);
	return (cnt == out_shape[1]) ? cnt : -1;
}

int CBFullyConnection::LoadCodebookFromFile(const char* path) {
	if (codebook != nullptr)
		free(codebook);
	codebook = (float*)malloc(sizeof(float) * (1 << cb_bits));

	unsigned long cnt = LoadFromFile(path, (char*)codebook);

	cnt /= sizeof(float);
	return (cnt == (1 << cb_bits)) ? cnt : -1;
}
