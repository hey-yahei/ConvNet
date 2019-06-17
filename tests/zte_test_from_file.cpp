#include <vector>
#include <assert.h>
#include <math.h>
#include <string>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <fstream>
#include "helper.h"

using namespace std;

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

void GetTPR(vector<vector<float>>& features1, vector<vector<float>>& features2)
{
    assert(features1.size() == features2.size());

    vector<float> posSims;
    vector<float> negSims;

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

    float threshold_00001=negSims[negSims.size()*0.0001];
    int posErrorNum_00001=0;
    
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
    string output_root = "./outputs/";
    char buffer[100], buffer2[1500];
    const char *sep = " \r\n";
    char *p;
    vector<vector<float>> features1, features2;

    ifstream in("./test_2000_images_list.txt");
    if (!in.is_open()) { 
        cout << "Error opening images list"; 
        exit (1); 
    } 
    while (!in.eof() ){ 
        in.getline(buffer, 100);

        p = strtok(buffer, sep);
        string path = string(p);
        string out_path = output_root + path.replace(path.find("."), 4, ".dat");
        LoadFromFile(out_path.c_str(), buffer2);
        vector<float> feature1(buffer2, buffer2 + 256);
        features1.push_back(feature1);

        p = strtok(NULL, sep);
        path = string(p);
        out_path = output_root + path.replace(path.find("."), 4, ".dat");
        LoadFromFile(out_path.c_str(), buffer2);
        vector<float> feature2(buffer2, buffer2 + 256);
        features2.push_back(feature2);
    }
    cout << "123" << endl;
    GetTPR(features1, features2);

    return 0;
}