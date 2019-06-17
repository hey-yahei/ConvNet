#include "ZTE_AI.h"
#include <vector>
#include <assert.h>
#include <math.h>
#include<opencv2/opencv.hpp>
#include "helper.h"

using namespace std;
using namespace cv;

float CosineVal(vector<float>& feature1, vector<float>& feature2)
{
    assert(feature1.size() == feature2.size());
    float sumAA = 0;
    float sumBB = 0;
    float sumAB = 0;
    for(int i=0; i<feature1.size(); ++i)
    {
        sumAA += feature1[i] * feature1[i];
        sumBB += feature2[i] * feature2[i];
        sumAB += feature1[i] * feature2[i];
    }
    float res = sumAB / (sqrt(sumAA*sumBB) + 1e-5);
    return res;
}

void GetTPR(ZTE_AI& ai, vector<string>& images_list1, vector<string>& images_list2)
{
    vector<vector<float> > features1, features2;
    assert(images_list1.size() == images_list2.size());
    for(int i=0; i<images_list1.size(); ++i)
    {
        cout << "forward(" << i << "): " << images_list1[i] << ", " << images_list2[i] << endl;
        Mat image1 = imread(images_list1[i]);
        vector<float> tmp_feature1 = ai.GetFeature(image1);
        features1.push_back(tmp_feature1);
        
        Mat image2 = imread(images_list2[i]);
        vector<float> tmp_feature2 = ai.GetFeature(image2);
        features2.push_back(tmp_feature2);
    }

    vector<float> posSims;
    vector<float> negSims;

    cout << endl << "Calculate cosine distance" << endl;
    for(int i=0; i<features1.size(); ++i)
    {
        for(int j=0; j<features2.size(); ++j)
        {
            float sim = CosineVal(features1[i],features2[j]);
            if(i == j)
            {
                posSims.push_back(sim);
            }
            else
            {
                negSims.push_back(sim);
            }
        }
    }

    sort(negSims.begin(),negSims.end(),greater<float>());
    sort(posSims.begin(),posSims.end(),less<float>());

    // cout << negSims.size() << endl;
    // cout << negSims[0] << " " << negSims[1] << " " << negSims[2] << endl;

    float threshold_00001=negSims[negSims.size()*0.0001];
    int posErrorNum_00001=0;
    
    cout << endl << "Calculate TPR" << endl;
    for(int i=0;i<posSims.size();i++)
    {
        if(posSims[i]<threshold_00001)
        {
            posErrorNum_00001++;
        }
        else
            break;
    }

    float TPR_00001=(posSims.size()-posErrorNum_00001)/(float)posSims.size();
    
    cout << "TPR is " << TPR_00001 << " @FPR=0.0001."<<endl;
}

int main(){
    ZTE_AI model("../compress_test");
    string root = "../test_2000/";
    char buffer[100];
    const char *sep = " \r\n";
    char *p;
    vector<string> list1;
    vector<string> list2;

    ifstream in("../test_2000_images_list.txt");
    if (!in.is_open()) { 
        cout << "Error opening images list"; 
        exit (1); 
    }  
    // while (!in.eof() ) {  
    for(int i = 0; i < 2; i++){     // test
        in.getline(buffer, 100);  
        p = strtok(buffer, sep);
        list1.push_back(root + string(p));
        p = strtok(NULL, sep);
        list2.push_back(root + string(p));
    }
    GetTPR(model, list1, list2);

    return 0;
}