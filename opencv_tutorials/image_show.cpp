/*
  * png から mp4  へ変換
  *
  * argv[1] 1 or 2 (dataset) 
  * argv[2] pass1 ~ 10 (pass1~pass10) 
  * argv[3] 1 ~ 5 (1~5) 
  *
  */
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/core.hpp>

using namespace std;

const string output_pass = "/home/amsl/Videos/";

int main (int argc, char** argv)
{

    cv::VideoWriter writer(output_pass +  argv[1] +".mp4",cv::VideoWriter::fourcc('m', 'p', '4', 'v') , 10, cv::Size(808, 614), true);


    cv::Mat src_img;
    int a=0;
    

    while(1){

        
        src_img = cv::imread(output_pass + argv[1] + "/" + argv[2] + "/" + argv[3] + "/vision/"  + to_string(a)  +"_camera.png", 1); 
        // 画像が読み込まれなかったらプログラム終了
        if(src_img.empty()) return -1; 

        // 結果画像表示
        cv::namedWindow("Image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
        cv::imshow("Image", src_img);
        writer << src_img;

        cv::waitKey(1);
        a++;
        if(a>1000){
            cout<<"アカン"<<endl;
            break;
        }   
    }   

    return 0;
}

