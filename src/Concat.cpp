#include "Concat.h"
#include <cstdarg>
#include <iostream>

using namespace std;


Concat::Concat(int ishape[], ...) {
	va_list ap;
	int cnt = 0;
	int *temp;

	// First arg
	va_start(ap, ishape);
	temp = (int*)malloc(sizeof(int) * 4);
	memcpy(temp, ishape, sizeof(int) * 4);
	in_shape.push_back(temp);
	memcpy(out_shape, ishape, sizeof(int) * 4);
	cnt++;

	// Other args
	while(nullptr != (ishape=va_arg(ap, int*))){
		// Push into vector
		temp = (int*)malloc(sizeof(int) * 4);
		memcpy(temp, ishape, sizeof(int) * 4);
		in_shape.push_back(temp);

		// Check if shapes match
		if ((temp[0] != out_shape[0]) || (temp[2] != out_shape[2]) || (temp[3] != out_shape[3])) {
			cerr << "Concat: shapes of input not match." << endl;
			exit(1);
		}

		// Update out_channels and counter
		out_shape[1] += temp[1];
		cnt++;
	}
	va_end(ap);

	out_chw = out_shape[1] * out_shape[2] * out_shape[3];
}

float* Concat::Run(float* in_data, ...) {
	va_list ap;
	
	int offset = 0;
	int *ishape;
	int input_size;
	int idx_input = 0;
	int cnt = 0;

	out_data = (float*)malloc(sizeof(float) * out_shape[0] * out_chw);

	va_start(ap, in_data);
	while (cnt++ < in_shape.size()) {
		for (int idx_bs = 0; idx_bs < out_shape[0]; idx_bs++) {
			ishape = in_shape[idx_input];
			input_size = ishape[1] * ishape[2] * ishape[3];
			memcpy(out_data + idx_bs * out_chw + offset, in_data, sizeof(float) * input_size);
			offset += input_size;

			if (!keep_in[idx_input])
				free(in_data);
		}
		idx_input++;

		// Next input
		in_data = va_arg(ap, float*);
	}
	va_end(ap);

	if (idx_input != in_shape.size()) {
		cerr << "Concat: run failed" << endl;
		exit(1);
	}

	return out_data;
}

int Concat::SetKeepOrNot(bool keep, ...) {
	va_list ap;

	keep_in.clear();
	va_start(ap, keep);
	for (int i = 0; i < in_shape.size(); i++) {
		keep_in.push_back(keep);
		va_arg(ap, bool);
	}
	va_end(ap);

	return 0;
}
