#include <fstream>                   //C++ 文件操作
#include <iostream>                  //C++ input & output stream
#include <sstream>                   //C++ String stream, 读写内存中的string对象
#include "opencv2/opencv.hpp"       //OpenCV 头文件
#include "openvino/openvino.hpp"    //OpenVINO >=2022.1
 
using namespace std;
using namespace ov;
using namespace cv;
//安全帽数据集的标签
vector<string> class_names = { "BG","B1","B2","B3","B4","B5","B0","BBs","BBb","RG","R1","R2","R3","R4","R5","RO","RBs","RBb","NG","N1","N2","N3","N4","N5","NO","NBs","NBb","PG","P1","P2","P3","P4","P5","PO","PBs","PBb" };
//OpenVINO IR模型文件路径
string ir_filename = "/home/dqs/桌面/c++ 部署yolov5到openvino/best.onnx";
 
// @brief 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
// @param input_tensor 输入节点的tensor
// @param inpt_image 输入图片数据
void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
	// 获取输入节点要求的输入图片数据的大小
	ov::Shape tensor_shape = input_tensor.get_shape();
	const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
	const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
	const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
	auto image_size = width * height;
	// 读取节点数据内存指针
	float* input_tensor_data = input_tensor.data<float>();
	// 将图片数据填充到网络中
	// 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < height; h++) 
		{
			for (size_t w = 0; w < width; w++) 
			{
				input_tensor_data[c * image_size + h * width + w] = input_image.at<Vec3f>(h, w)[c];
			}
		}
	}
}
void letterbox(Mat& inframe,Mat& outframe, double k[], int x_size,int y_size) //定义一个用于将图片填充成正方形的函数 参数：（输入图像，输出图像）
{    
	int x_ori = inframe.cols;  //得到图片宽度    
	int y_ori = inframe.rows;  //得到图片高度    
	cout<<"宽："<<x_ori<<endl;    
	cout<<"高："<<y_ori<<endl;        
	float r_x = (double)x_ori/x_size; // 1280/640    
	float r_y = (double)y_ori/y_size; // 768/640 确定缩放比，后图除以原图，值比较大（缩放没那么严重的）那个方向要padding(来多缩放些)    
	if(r_x >= r_y) // 在上下方向填充 那么y的理论高度应该是x_ori/x_size*y_size 
	{      
		int y0 = y_size * x_ori / x_size;      
		int y_pad = (y0 - y_ori)/2;
		k[0] = 0;
		k[1] = 1.0*y_pad/y0*y_size;
		cv::copyMakeBorder(inframe,inframe,y_pad,y_pad,0,0, cv::BORDER_CONSTANT,20);
    }    
	else // 在上下方向填充 那么y的理论高度应该是x_ori/x_size*y_size
    {        
		int x0 = x_size * y_ori / y_size;        
		int x_pad = (x0 - x_ori)/2;
		k[0] = 1.0*x_pad/x0*x_size;
		k[1] = 0;   
		cv::copyMakeBorder(inframe,inframe,0,0,x_pad,x_pad, cv::BORDER_CONSTANT,20); 
	}
}
int main(int argc, char** argv) {
 
	//创建OpenVINO Core
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(ir_filename, "AUTO");
	ov::InferRequest infer_request = compiled_model.create_infer_request();
	cv::VideoCapture cap(0);
 
	// 预处理输入数据 - 格式化操作
	//cv::Mat grid = cv::imread("../293.jpg");
	//cout<<"图片大小"<<grid.size()<<endl;
    //imshow("grid",grid);
 
	//获取输入节点tensor
	Tensor input_image_tensor = infer_request.get_tensor("input");
	int input_h = input_image_tensor.get_shape()[2]; //获得"images"节点的Height
	int input_w = input_image_tensor.get_shape()[3]; //获得"images"节点的Width
	int input_c = input_image_tensor.get_shape()[1]; //获得"images"节点的channel
	cout << "input_h:" << input_h << "; input_w:" << input_w << endl;
	cout << "input_image_tensor's element type:" << input_image_tensor.get_element_type() << endl;
	cout << "input_image_tensor's shape:" << input_image_tensor.get_shape() << endl;
	// 获取输出节点tensor
	Tensor output_tensor = infer_request.get_tensor("output");
	int out_rows = output_tensor.get_shape()[1]; //获得"output"节点的out_rows
	int out_cols = output_tensor.get_shape()[2]; //获得"output"节点的Width
	cout << "out_cols:" << out_cols << "; out_rows:" << out_rows << endl;
 
	//连续采集处理循环
	while(true){
		double k[2];
		Mat frame, image;
		cap >> frame;
		imshow("frame", frame);

		int64 start = cv::getTickCount();
		frame.copyTo(image);
		imshow("image", image);
	
		cv::Mat blob_image;
		letterbox(image,image,k,640,640);
		image.copyTo(blob_image);
		cout<<k[0]<<"  "<<k[1]<<endl;
		cout<<"大小"<<blob_image.size();
		cv::resize(image, blob_image, cv::Size(input_w, input_h));
		imshow("src",blob_image);
		cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
		blob_image.convertTo(blob_image, CV_32F, 1.0 / 255);
	
		// 将图片数据填充到tensor数据内存中
		fill_tensor_data_image(input_image_tensor, blob_image);
	
		// 执行推理计算
		infer_request.infer();
	
		// 获得推理结果
		const ov::Tensor& output_tensor_1 = infer_request.get_tensor("output");
	
		// 解析推理结果，YOLOv5 output format: cx,cy,w,h,x1,y1,x2,y2,x3,y3,x4,y4,confidences 
		cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_tensor_1.data());
		vector<cv::Rect> boxes;
		vector<vector<Point2f>> points;
		vector<int> classIds;
		vector<float> confidences;
	
		for (int i = 0; i < det_output.rows; i++) 
		{
			float confidence = det_output.at<float>(i, 4);
			if (confidence < 0.6)
			{
				continue;
			}
			cv::Mat classes_scores = det_output.row(i).colRange(13, 49);
			// 获取最大classes_scores
			cv::Point classIdPoint;   
			double score;
			cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
	
			// 置信度 0～1之间
			if (score > 0.7)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				float x1 = det_output.at<float>(i, 5);
				float y1 = det_output.at<float>(i, 6);
				float x2 = det_output.at<float>(i, 7);
				float y2 = det_output.at<float>(i, 8);
				float x3 = det_output.at<float>(i, 9);
				float y3 = det_output.at<float>(i, 10);
				float x4 = det_output.at<float>(i, 11);
				float y4 = det_output.at<float>(i, 12);
				//cout<<det_output.at<float>(i, 4)<<endl;
				//cout<<"位置"<<cx_1<<"   "<<cy_1<<"   "<<x1_1<<"   "<<y1_1<<"   "<<x2_1<<"   "<<y2_1<<"   "<<x3_1<<"   "<<y3_1<<"   "<<x4_1<<"   "<<y4_1<<endl;
				cv::Rect box;
				box.x = x1;	
				box.y = y1;
				box.width = ow;
				box.height = oh;
				Point2f p1_1(x1,y1);
				Point2f p2_1(x2,y2);
				Point2f p3_1(x3,y3);
				Point2f p4_1(x4,y4);
				Point2f p5_1(cx,cy);
				vector<Point2f> point;
				point.push_back(p1_1);
				point.push_back(p2_1);
				point.push_back(p3_1);
				point.push_back(p4_1);
				point.push_back(p5_1);

				points.push_back(point);
				point.clear();
				boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}
		// NMS
		vector<int> indexes;
		cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
		for (size_t i = 0; i < indexes.size(); i++) 
		{
			int index = indexes[i];
			int idx = classIds[index];
			//cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
			//cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			//		cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
			//Point2f p1 = points[index][0];
			//Point2f p2 = points[index][1];
			// Point2f p3 = points[index][2];
			// Point2f p4 = points[index][3];
			for(int i = 0;i<5;i++){
				points[index][i].x = (points[index][i].x-k[0])*frame.cols/(640-2*k[0]);
				points[index][i].y = (points[index][i].y-k[1])*frame.rows/(640-2*k[1]);
				cout<<frame.size<<endl;
			}
			line(frame,points[index][0],points[index][1],Scalar(0,0,255),2);
			line(frame,points[index][1],points[index][2],Scalar(0,0,255),2);
			line(frame,points[index][2],points[index][3],Scalar(0,0,255),2);
			line(frame,points[index][3],points[index][0],Scalar(0,0,255),2);
			//cv::putText(frame, class_names[idx], cv::Point(boxes[index].tl().x*2, (boxes[index].tl().y-128)*768/384 - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
		}
		// 计算FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << endl;
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 3, 8);
		cv::imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", frame);
	
		char c = cv::waitKey(60);
		if (c == 27) 
		{ // ESC
			break;
		}
	}

 
	 cv::waitKey(0);
	
 
	return 0;
}