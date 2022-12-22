#include <fstream>                   //C++ 文件操作
#include <iostream>                  //C++ input & output stream
#include <sstream>                   //C++ String stream, 读写内存中的string对象
#include "opencv2/opencv.hpp"       //OpenCV 头文件
#include "openvino/openvino.hpp"    //OpenVINO >=2022.1
 
using namespace std;
using namespace ov;
using namespace cv;
void letterbox(Mat& inframe,Mat& outframe,int x_size,int y_size) //定义一个用于将图片填充成正方形的函数 参数：（输入图像，输出图像）
{   
    cv::imshow("inframe",inframe);
	int x_ori = inframe.cols;  //得到图片宽度    
	int y_ori = inframe.rows;  //得到图片高度    
	cout<<"宽："<<x_ori<<endl;    
	cout<<"高："<<y_ori<<endl;        
	float r_x = (double)x_ori/x_size; // 1280/640    
	float r_y = (double)y_ori/y_size; // 768/640       确定缩放比，后图除以原图，值比较大（缩放没那么严重的）那个方向要padding(来多缩放些)    
	if(r_x >= r_y) // 在上下方向填充 那么y的理论高度应该是x_ori/x_size*y_size 
	{      
		int y0 = y_size * x_ori / x_size;      
		int y_pad = (y0 - y_ori)/2;        
		cv::copyMakeBorder(inframe,inframe,y_pad,y_pad,0,0, cv::BORDER_CONSTANT,20);
    }    
	else // 在左右方向填充 那么x的理论高度应该是y_ori/y_size*x_size
    {        
		int x0 = x_size * y_ori / y_size;        
		int x_pad = (x0 - x_ori)/2;        
		cv::copyMakeBorder(inframe,inframe,0,0,x_pad,x_pad, cv::BORDER_CONSTANT,20); 
	}
    cv::imshow("in",inframe);
}
int main(){
    cv::Mat grid = cv::imread("../293.jpg");
    //imshow("grid",grid);
    cv::Mat blob_image;
	letterbox(grid,blob_image,640,640);
    //cout<<blob_image.size();
    //imshow("blob_image",blob_image);
    waitKey(0);
}