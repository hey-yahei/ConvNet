#include "ZTE_AI.h"
#include "helper.h"
#include <iostream>

void ZTE_AI::Build() {
	int s[4];

	conv1 = new Convolution(3, 2, 1, 64, 1, in_shape, true, true);
	conv1->GetOutputShape(s);

	conv2a_1 = new CBConvolution(1, 1, 0, 64, 1, s, false, true, 7);
	conv2a_2a = new CBConvolution(3, 1, 1, 64, 1, s, true, true, 7);
	conv2a_2a->GetOutputShape(s);
	conv2a_2b = new CBConvolution(3, 1, 1, 64, 1, s, false, true, 7);
	conv2a_2b->GetOutputShape(s);
	res_2a = new Eltwise("add", s, true);
	res_2a->GetOutputShape(s);
	conv2b_2a = new CBConvolution(3, 1, 1, 64, 1, s, true, true, 7);
	conv2b_2a->GetOutputShape(s);
	conv2b_2b = new CBConvolution(3, 1, 1, 64, 1, s, false, true, 7);
	conv2b_2a->GetOutputShape(s);
	res_2b = new Eltwise("add", s, true);

	conv3a_1 = new CBConvolution(1, 2, 0, 128, 1, s, false, true, 7);
	conv3a_2a = new CBConvolution(3, 2, 1, 128, 1, s, true, true, 7);
	conv3a_2a->GetOutputShape(s);
	conv3a_2b = new CBConvolution(3, 1, 1, 128, 1, s, false, true, 7);
	conv3a_2b->GetOutputShape(s);
	res_3a = new Eltwise("add", s, true);
	res_3a->GetOutputShape(s);
	conv3b_2a = new CBConvolution(3, 1, 1, 128, 1, s, true, true, 7);
	conv3b_2a->GetOutputShape(s);
	conv3b_2b = new CBConvolution(3, 1, 1, 128, 1, s, false, true, 7);
	conv3b_2a->GetOutputShape(s);
	res_3b = new Eltwise("add", s, true);

	conv4a_1 = new CBConvolution(1, 2, 0, 256, 1, s, false, true, 7);
	conv4a_2a = new CBConvolution(3, 2, 1, 256, 1, s, true, true, 7);
	conv4a_2a->GetOutputShape(s);
	conv4a_2b = new CBConvolution(3, 1, 1, 256, 1, s, false, true, 7);
	conv4a_2b->GetOutputShape(s);
	res_4a = new Eltwise("add", s, true);
	res_4a->GetOutputShape(s);
	conv4b_2a = new CBConvolution(3, 1, 1, 256, 1, s, true, true, 7);
	conv4b_2a->GetOutputShape(s);
	conv4b_2b = new CBConvolution(3, 1, 1, 256, 1, s, false, true, 7);
	conv4b_2a->GetOutputShape(s);
	res_4b = new Eltwise("add", s, true);

	conv5a_1 = new CBConvolution(1, 2, 0, 512, 1, s, false, true, 7);
	conv5a_2a = new CBConvolution(3, 2, 1, 512, 1, s, true, true, 7);
	conv5a_2a->GetOutputShape(s);
	conv5a_2b = new CBConvolution(3, 1, 1, 512, 1, s, false, true, 7);
	conv5a_2b->GetOutputShape(s);
	res_5a = new Eltwise("add", s, true);
	res_5a->GetOutputShape(s);
	conv5b_2a = new CBConvolution(3, 1, 1, 512, 1, s, true, true, 7);
	conv5b_2a->GetOutputShape(s);
	conv5b_2b = new CBConvolution(3, 1, 1, 512, 1, s, false, true, 7);
	conv5b_2a->GetOutputShape(s);
	res_5b = new Eltwise("add", s, false);

	fc = new CBFullyConnection(256, s, false, false, true, 4);
}

void ZTE_AI::Load(string model_dir) {
	conv1->LoadWeightFromFile((model_dir + "/ztemodel0_conv0.weight.dat").c_str());
	conv1->LoadBiasFromFile((model_dir + "/ztemodel0_conv0.bias.dat").c_str());

	conv2a_1->LoadCodebookFromFile((model_dir + "/ztemodel0_conv1.weight.codebook.dat").c_str());
	conv2a_1->LoadWeightFromFile((model_dir + "/ztemodel0_conv1.weight").c_str(), false, true);
	conv2a_1->LoadBiasFromFile((model_dir + "/ztemodel0_conv1.bias.dat").c_str());
	conv2a_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv2.weight.codebook.dat").c_str());
	conv2a_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv2.weight").c_str(), false, true);
	conv2a_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv2.bias.dat").c_str());
	conv2a_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv3.weight.codebook.dat").c_str());
	conv2a_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv3.weight").c_str(), false, true);
	conv2a_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv3.bias.dat").c_str());
	conv2b_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv4.weight.codebook.dat").c_str());
	conv2b_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv4.weight").c_str(), false, true);
	conv2b_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv4.bias.dat").c_str());
	conv2b_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv5.weight.codebook.dat").c_str());
	conv2b_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv5.weight").c_str(), false, true);
	conv2b_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv5.bias.dat").c_str());

	conv3a_1->LoadCodebookFromFile((model_dir + "/ztemodel0_conv6.weight.codebook.dat").c_str());
	conv3a_1->LoadWeightFromFile((model_dir + "/ztemodel0_conv6.weight").c_str(), false, true);
	conv3a_1->LoadBiasFromFile((model_dir + "/ztemodel0_conv6.bias.dat").c_str());
	conv3a_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv7.weight.codebook.dat").c_str());
	conv3a_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv7.weight").c_str(), false, true);
	conv3a_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv7.bias.dat").c_str());
	conv3a_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv8.weight.codebook.dat").c_str());
	conv3a_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv8.weight").c_str(), false, true);
	conv3a_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv8.bias.dat").c_str());
	conv3b_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv9.weight.codebook.dat").c_str());
	conv3b_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv9.weight").c_str(), false, true);
	conv3b_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv9.bias.dat").c_str());
	conv3b_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv10.weight.codebook.dat").c_str());
	conv3b_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv10.weight").c_str(), false, true);
	conv3b_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv10.bias.dat").c_str());

	conv4a_1->LoadCodebookFromFile((model_dir + "/ztemodel0_conv11.weight.codebook.dat").c_str());
	conv4a_1->LoadWeightFromFile((model_dir + "/ztemodel0_conv11.weight").c_str(), false, true);
	conv4a_1->LoadBiasFromFile((model_dir + "/ztemodel0_conv11.bias.dat").c_str());
	conv4a_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv12.weight.codebook.dat").c_str());
	conv4a_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv12.weight").c_str(), false, true);
	conv4a_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv12.bias.dat").c_str());
	conv4a_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv13.weight.codebook.dat").c_str());
	conv4a_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv13.weight").c_str(), false, true);
	conv4a_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv13.bias.dat").c_str());
	conv4b_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv14.weight.codebook.dat").c_str());
	conv4b_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv14.weight").c_str(), false, true);
	conv4b_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv14.bias.dat").c_str());
	conv4b_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv15.weight.codebook.dat").c_str());
	conv4b_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv15.weight").c_str(), false, true);
	conv4b_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv15.bias.dat").c_str());

	conv5a_1->LoadCodebookFromFile((model_dir + "/ztemodel0_conv16.weight.codebook.dat").c_str());
	conv5a_1->LoadWeightFromFile((model_dir + "/ztemodel0_conv16.weight").c_str(), false, true);
	conv5a_1->LoadBiasFromFile((model_dir + "/ztemodel0_conv16.bias.dat").c_str());
	conv5a_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv17.weight.codebook.dat").c_str());
	conv5a_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv17.weight").c_str(), false, true);
	conv5a_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv17.bias.dat").c_str());
	conv5a_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv18.weight.codebook.dat").c_str());
	conv5a_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv18.weight").c_str(), false, true);
	conv5a_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv18.bias.dat").c_str());
	conv5b_2a->LoadCodebookFromFile((model_dir + "/ztemodel0_conv19.weight.codebook.dat").c_str());
	conv5b_2a->LoadWeightFromFile((model_dir + "/ztemodel0_conv19.weight").c_str(), false, true);
	conv5b_2a->LoadBiasFromFile((model_dir + "/ztemodel0_conv19.bias.dat").c_str());
	conv5b_2b->LoadCodebookFromFile((model_dir + "/ztemodel0_conv20.weight.codebook.dat").c_str());
	conv5b_2b->LoadWeightFromFile((model_dir + "/ztemodel0_conv20.weight").c_str(), false, true);
	conv5b_2b->LoadBiasFromFile((model_dir + "/ztemodel0_conv20.bias.dat").c_str());

	fc->LoadCodebookFromFile((model_dir + "/ztemodel0_dense0.weight.codebook.dat").c_str());
	fc->LoadWeightFromFile((model_dir + "/ztemodel0_dense0.weight").c_str(), true, true);
}

#define SIZE4(s) (s[0]*s[1]*s[2]*s[3])
#define SIZE2(s) (s[0]*s[1])
float* ZTE_AI::Run(float* x) {
		// int s[4];
	float* x_;
	x = conv1->Run(x);
		// conv1->GetOutputShape(s);
		// SaveToFile("./outputs/conv1.outdat", (char*)x, sizeof(float)*SIZE4(s));

	x_ = conv2a_1->Run(x, true);
		// conv2a_1->GetOutputShape(s);
		// SaveToFile("./outputs/conv2a_1.outdat", (char*)x_, sizeof(float)*SIZE4(s));
	x = conv2a_2a->Run(x);
		// conv2a_2a->GetOutputShape(s);
		// SaveToFile("./outputs/conv2a_2a.outdat", (char*)x, sizeof(float)*SIZE4(s));
	x = conv2a_2b->Run(x);
		// conv2a_2b->GetOutputShape(s);
		// SaveToFile("./outputs/conv2a_2b.outdat", (char*)x, sizeof(float)*SIZE4(s));
	x = res_2a->Run(x_, x);
		// res_2a->GetOutputShape(s);
		// SaveToFile("./outputs/res_2a.outdat", (char*)x, sizeof(float)*SIZE4(s));
	x_ = x;
	x = conv2b_2a->Run(x, true);
		// conv2b_2a->GetOutputShape(s);
		// SaveToFile("./outputs/conv2b_2a.outdat", (char*)x, sizeof(float)*SIZE4(s));
	x = conv2b_2b->Run(x);
		// conv2b_2b->GetOutputShape(s);
		// SaveToFile("./outputs/conv2b_2b.outdat", (char*)x, sizeof(float)*SIZE4(s));
	x = res_2b->Run(x_, x);
		// res_2b->GetOutputShape(s);
		// SaveToFile("./outputs/res_2b.outdat", (char*)x, sizeof(float)*SIZE4(s));

	x_ = conv3a_1->Run(x, true);
	x = conv3a_2a->Run(x);
	x = conv3a_2b->Run(x);
	x = res_3a->Run(x_, x);
	x_ = x;
	x = conv3b_2a->Run(x, true);
	x = conv3b_2b->Run(x);
	x = res_3b->Run(x_, x);

	x_ = conv4a_1->Run(x, true);
	x = conv4a_2a->Run(x);
	x = conv4a_2b->Run(x);
	x = res_4a->Run(x_, x);
	x_ = x;
	x = conv4b_2a->Run(x, true);
	x = conv4b_2b->Run(x);
	x = res_4b->Run(x_, x);

	x_ = conv5a_1->Run(x, true);
	x = conv5a_2a->Run(x);
	x = conv5a_2b->Run(x);
	x = res_5a->Run(x_, x);
	x_ = x;
	x = conv5b_2a->Run(x, true);
	x = conv5b_2b->Run(x);
	x = res_5b->Run(x_, x);
		// res_5b->GetOutputShape(s);
		// SaveToFile("./outputs/res_5b.outdat", (char*)x, sizeof(float)*SIZE4(s));

	x = fc->Run(x);
		// fc->GetOutputShape(s);
		// SaveToFile("./outputs/fc.outdat", (char*)x, sizeof(float)*SIZE2(s));

	return x;
}

float* Preprocess(Mat &image, int img_h = 128, int img_w = 128){
	// resize(image, image, Size(img_h, img_w));
	unsigned char* img_data = ( unsigned char* )image.data;
	float* input_data = (float* )malloc(sizeof(float) * 3 * 128 * 128);
	int hw = img_h * img_w;

    for(int h = 0; h < img_h; h++) {
        for(int w = 0; w < img_w; w++) {
            for(int c = 0; c < 3; c++) {
                input_data[c * hw + h * img_w + w] = 0.0078125 * (*img_data - 127.5);
                img_data++;
            }
        }
    }
	return input_data;
}

vector<float> ZTE_AI::GetFeature(Mat &image){
	float* in = Preprocess(image);
	float* out = Run(in);
	int shape[2];
	GetOutputShape(shape);

	vector<float> vec(out, out + shape[0] * shape[1]);
	delete out;
	return vec;
}