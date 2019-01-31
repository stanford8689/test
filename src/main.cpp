
/*
#include <opencv2/opencv.hpp>
using namespace cv;
#include "function.h"

int main(int argc, char** argv) {

    Mat image;
    image = imread("/Users/didi/Desktop/C_test/src/lena.jpg", 1);
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);


    test();

    int i = 0;


    while(1)
    {
        printf("%i\n",i);
        i++;
    }




    return 0;
}
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/legacy/constants_c.h>
#include <opencv2/videoio/videoio_c.h>

#include "../include/function.h"


#include<stdlib.h>
#include<stdio.h>
#include<vector>
#include<string>
#include<string.h>
#include<iostream>
#include<sys/time.h>
#include<thread>



using namespace std;
using namespace cv;

/*
IplImage* doCanny(IplImage* image_input,
                  double lowThresh,
                  double highThresh,
                  int aperture);



int main(int argc, char* argv[])
{
    cvNamedWindow("Camera" , CV_WINDOW_AUTOSIZE );

    CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY);

    assert(capture != NULL);

    IplImage *frame = 0;
    frame = cvQueryFrame(capture);

    IplImage *frame_edge = cvCreateImage(cvGetSize(frame),
                                         IPL_DEPTH_8U,
                                         1);
    while(1)
    {
        frame = cvQueryFrame(capture);
        if(!frame) break;

        //cvConvertScale(frame,frame_edge,0);
        frame = cvCloneImage(frame_edge);

        frame_edge = doCanny(frame_edge,70,90,3);

        cvShowImage("Camera",frame_edge);
        char c = cvWaitKey(15);
        if(c == 27)  break;
    }

    cvReleaseCapture(&capture);
    cvReleaseImage( &frame_edge );
    cvReleaseImage( &frame);


    return (int)0;
}





IplImage* doCanny(IplImage* image_input,
                  double lowThresh,
                  double highThresh,
                  int aperture)
{
    if(image_input->nChannels != 1)
        return (0);

    IplImage* image_output = cvCreateImage(cvGetSize(image_input),
                                           image_input->depth,
                                           image_input->nChannels);

    cvCanny(image_input,image_output,lowThresh,highThresh,aperture);

    return(image_output);
}
*/

int ImageCapture();
void test1();
void test2();
void test3();
double timeProcess(struct timeval begin, struct timeval end);
Mat get_perspective_mat();
int main()
{


    //ImageCaptrue();
    //thread t(test1);
    //test2();
    test3();


    return 0;
}




int ImageCapture()
{
	//打开摄像头
	VideoCapture capture;
	capture.open(0);
	//灰度图像
	Mat edge;
	//循环显示每一帧
	while (1)
	{
		//frame存储每一帧图像
		Mat frame;
		//读取当前帧
		capture >> frame;
		//显示当前视频
		imshow("正在录制", frame);
		//得到灰度图像
		cvtColor(frame, edge, CV_BGR2GRAY);
		//3*3降噪 （2*3+1)
		blur(edge, edge,Size(7,7));
		//边缘显示
		Canny(edge,edge,0,30,3);
		imshow("高斯模糊视频",edge);
		//延时30ms,按下任何键退出
		if (waitKey(30) >= 0)
			break;
	}

    return 0;
}


void test1(){


    struct timeval begin,end;
    gettimeofday(&begin,NULL);

    for(int i =0; i<10;i++)
    {
        cout<<i*i<<i<<"*"<<i<<endl;
    }

    gettimeofday(&end,NULL);

    cout<<"cost time :"<<timeProcess(begin,end)<<"ms"<<endl; 

}


double timeProcess(struct timeval begin, struct timeval end)
{
    return end.tv_sec*1000+end.tv_usec/1000.0 - begin.tv_sec*1000+begin.tv_usec/1000.0;
}



void test2()
{
    //透视变换的demo
	Mat image,M,perspective;
    image = imread("/Users/didi/Desktop/C_test/src/TransDemo.png", -1);
    namedWindow("before transform", WINDOW_AUTOSIZE);
    imshow("image",image);
    waitKey(0);

    M = get_perspective_mat();
	warpPerspective(image, perspective, M, Size(960, 270), INTER_LINEAR);

    cout<<"after transformed: "<<endl;
    cout<<M<<endl;
    namedWindow("after transform", WINDOW_AUTOSIZE);
    imshow("transformed ",perspective);

    waitKey(10000);

    destroyAllWindows();
    
}


Mat get_perspective_mat()
{
	Point2f src_points[] = { 
		cv::Point2f(165, 270),
		cv::Point2f(835, 270),
		cv::Point2f(360, 125),
		cv::Point2f(615, 125) };
 
	Point2f dst_points[] = {
		cv::Point2f(165, 270),
		cv::Point2f(835, 270),
		cv::Point2f(165, 30),
		cv::Point2f(835, 30) };
 
	Mat M = cv::getPerspectiveTransform(src_points, dst_points);
	
	return M;
 
}


void test3()
{
    vector<vector<int *> > result;

    vector<int *> sub_result;

    int temp = 0;
    int *p = NULL;

    for(int i=0;i<10;i++)
    {
        for(int j=0;j<7;j++)
        {
            temp++;
            p = &temp;
            cout<<*p<<" ";
            sub_result.push_back(p);
        }
        cout<<endl;
        result.push_back(sub_result);
        
    }
    cout<<result.back().size()<<endl;

    int *temp_result = NULL;
    for(int i=0;i<10;i++)
    {
        for(int j=0;j<7;j++)
        {
            temp_result = result[i][j];
            //cout<<result[i][j]<< " "; 
            cout<<*temp_result<<" ";
        }
        cout<<endl;
    }

    
}
