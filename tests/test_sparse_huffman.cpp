#if 0

#include "helper.h"
#include "TarFile.h"
#include "CBConvolution.h"
#include <iostream>

void test_sparse_huffman() {
	// TarFile tarfile("../models/huffman/all.tar");
	// if (!tarfile.IsValidTarFile()) {
	// 	cerr << "tar file is invalid" << endl;
	// 	exit(1);
	// }

	int shape[4] = { 1,32,20,20 };
	float* out;
	float* in = (float*)malloc(sizeof(float) * 1 * 32 * 20 * 20);
	CBConvolution test_conv(3, 2, 1, 64, 1, shape, false, false, 3);

	
	LoadFromFile("../models/huffman/input.dat", (char*)in);
	test_conv.LoadCodebookFromFile("../models/huffman/cifarresnetv10_stage3_conv0.codebook.dat");
	test_conv.LoadWeightFromFile("../models/huffman/cifarresnetv10_stage3_conv0.weight", false, true);
	// LoadFromFile("input.dat", (char*)in, &tarfile);
	// test_conv.LoadCodebookFromFile("cifarresnetv10_stage3_conv0.codebook.dat", &tarfile);
	// test_conv.LoadWeightFromFile("cifarresnetv10_stage3_conv0.weight", false, true, &tarfile);
	out = test_conv.Run(in);
	test_conv.GetOutputShape(shape);
	SaveToFile("../models/huffman/out.dat", (char*)out, sizeof(float) * shape[0] * shape[1] * shape[2] * shape[3]);
}

int main() {
	test_sparse_huffman();
	return 0;
}

#endif