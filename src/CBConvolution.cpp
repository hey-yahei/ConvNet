#include "CBConvolution.h"
#include "helper.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>

using namespace std;


CBConvolution::CBConvolution(int kernel, int stride, int pad, int out_channels, int groups, int ishape[], bool with_relu, bool with_bias, int cb_bits)
	: kernel(kernel), stride(stride), pad(pad), in_channels(ishape[1]),
	out_channels(out_channels), groups(groups), with_relu(with_relu), with_bias(with_bias), cb_bits(cb_bits) {
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
}


float* CBConvolution::Run(float* in_data, bool keep_in) {
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

int CBConvolution::LoadWeightFromFile(const char* path, bool compressed_data, bool sparse_huffman, TarFile* tarfile) {
	unsigned long cnt;
	compressed_weight = compressed_data;

	/*if (compressed_data && sparse_huffman) {
		cerr << "CBConvolution: compressed=true and huffman=true" << endl;
		exit(1);
	}*/

	if (weight_data != nullptr)
		free(weight_data);
	
		if (sparse_huffman) {
			weight_data = (unsigned char*)malloc(sizeof(unsigned char) * out_channels * ic_per_gp * kernel * kernel);
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

			memset(weight_data, zero_label, sizeof(unsigned char) * out_channels * ic_per_gp * kernel * kernel);
			cnt = LoadFromSparseHuffmanFile(path, (char*)weight_data, tarfile);
			if(compressed_data){
				int size1 = out_channels * ic_per_gp * kernel * kernel;
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

				return (cnt == ceil(out_channels * ic_per_gp * kernel * kernel * cb_bits / 8.)) ? cnt : -1;
			}else{
				return (cnt == out_channels * ic_per_gp * kernel * kernel) ? cnt : -1;
			}
		} else if (compressed_data) {
		    weight_data = (unsigned char*)malloc(sizeof(unsigned char) * ceil(out_channels * ic_per_gp * kernel * kernel * cb_bits / 8.));
		    cnt = LoadFromFile(path, (char*)weight_data, tarfile);
		    return (cnt == ceil(out_channels * ic_per_gp * kernel * kernel * cb_bits / 8.)) ? cnt : -1;
	    }else {
			weight_data = (unsigned char*)malloc(sizeof(unsigned char) * out_channels * ic_per_gp * kernel * kernel);
			cnt = LoadFromFile(path, (char*)weight_data, tarfile);
			return (cnt == out_channels * ic_per_gp * kernel * kernel) ? cnt : -1;
		}
}

int CBConvolution::LoadBiasFromFile(const char* path, TarFile* tarfile) {
	if (!with_bias) {
		cerr << "CBConvolution: without bias but try to load" << endl;
		exit(1);
	}
	if (bias_data != nullptr)
		free(bias_data);
	bias_data = (float*)malloc(sizeof(float) * out_channels);

	unsigned long cnt = LoadFromFile(path, (char*)bias_data, tarfile);

	cnt /= sizeof(float);
	return (cnt == out_channels) ? cnt : -1;
}

int CBConvolution::LoadCodebookFromFile(const char* path, TarFile* tarfile) {
	if (codebook != nullptr)
		free(codebook);
	codebook = (float*)malloc(sizeof(float) * (1 << cb_bits));

	unsigned long cnt = LoadFromFile(path, (char*)codebook, tarfile);

	cnt /= sizeof(float);
	return (cnt == (1 << cb_bits)) ? cnt : -1;
}
