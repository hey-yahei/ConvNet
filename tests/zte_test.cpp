#include "ZTE_AI.h"
#include "helper.h"
#include <time.h>

int main() {
	string model_dir = "../compress_test";
	clock_t start, finish;
	double totaltime;
	ZTE_AI model(model_dir);
	float* out;
	float* in = (float*)malloc(sizeof(float) * 1 * 3 * 128 * 128);

	start = clock();
	LoadFromFile((model_dir + "/input.dat").c_str(), (char*)in);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "load data: " << totaltime << "s" << endl;

	start = clock();
	out = model.Run(in);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "run: " << totaltime << "s" << endl;
	int shape[2];
	model.GetOutputShape(shape);
	SaveToFile((model_dir + "/output.dat").c_str(), (char*)out, sizeof(float) * shape[0] * shape[1]);

	return 0;
}