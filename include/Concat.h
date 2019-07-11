#include <iostream>
#include <cstring>
#include <vector>

using namespace std;


class Concat{
private:
	// Shape
	vector<int *> in_shape;
	int out_shape[4];

	// Flags
	vector<bool> keep_in;

	// Offset
	int out_chw;

	// Buffer
	float* out_data = nullptr;
public:
	Concat(){}
	Concat(int ishape[], ...);
	~Concat(){
		// Free in_shape
		while (!in_shape.empty()) {
			free(in_shape.back());
			in_shape.pop_back();
		}
	}

	float* Run(float* in_data, ...);
	int SetKeepOrNot(bool keep_in, ...);
	float* GetOutputTensor() { return out_data; }
	void GetOutputShape(int shape[]) { memcpy(shape, out_shape, sizeof(int) * 4); }
};

