#include "ZTE_AI.h"
#include "helper.h"
#include <iostream>
#include <vector>
#include "time.h"

using namespace std;

int main(int argc, char * argv[]){
    if(argc != 3){
        cout << "Usage: ./build/zte_output_tofile <start> <end>" << endl;
        exit(1);
    }
    int start, end;
    sscanf(argv[1], "%d", &start);
    sscanf(argv[2], "%d", &end);

    clock_t start_time, finish_time;
	double totaltime;

    ZTE_AI model("./model");
    int shape[2];
    model.GetOutputShape(shape);

    string image_root = "./test_images/";
    string output_root = "./outputs/";
    char buffer[100];
    const char *sep = " \r\n";
    char *p;

    ifstream in("./test_2000_images_list.txt");
    if (!in.is_open()) { 
        cout << "Error opening images list"; 
        exit (1); 
    } 
    for(int i = 0; i < start; i++)
        in.getline(buffer, 100);  
    for(int i = 0; i < end - start; i++){ 
        start_time = clock();
        in.getline(buffer, 100);
        cout << "[" << i + 1 << "/" << (end-start) << "] " << buffer << endl;

        p = strtok(buffer, sep);
        for(int j = 0; j < 2; j++){
            string path = string(p);
            Mat image = imread(image_root + path);
            vector<float> feature = model.GetFeature(image);
            string out_path = output_root + path.replace(path.find("."), 4, ".dat");
            SaveToFile(out_path.c_str(), (char*)&feature[0], sizeof(float)*feature.size());
            p = strtok(NULL, sep);
        }
        finish_time = clock();
	    totaltime = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	    cout << ".... cost time: " << totaltime << "s" << endl;
    }

    return 0;
}