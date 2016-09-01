#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
#include "spline.h"

using namespace std;
using namespace cv;

//follow x,y
std::vector< pair<float, float> > left_lane;
std::vector< pair<float, float> > right_lane;

//splines
tk::spline s_left, s_right;

bool sort_pred(const pair<float, float>& left, const pair<float, float>& right)
{
    return left.first < right.first;
}

void cluster(Mat img)
{
    Vec3b colorTab[] =
    {
        Vec3b(0, 0, 255),
        Vec3b(0,255,0),
        Vec3b(255,100,100),
        Vec3b(255,0,255),
        Vec3b(0,255,255)
    };


    int k, clusterCount = 2, clusterCount_1 = 1;
    int i, sampleCount = 0;
    int j;
    std::vector<Point2f> points;
    Mat labels;

    for(i=0; i<img.rows; i++)
    {
        for(j=0; j<img.cols; j++)
        {
            Vec3b BGR = img.at<Vec3b>(i,j);
            if(BGR.val[0]>230 && BGR.val[1]>230 && BGR.val[2]>230) //detecting white points
            {
                Point2f p;
                p.x = i;
                p.y = j;
                points.push_back(p);
            }
        }
    }

    sampleCount = points.size();

    clusterCount = MIN(clusterCount, sampleCount);

    printf("Sample count = %d, Cluster count = %d\n", sampleCount, clusterCount);

    Mat inputData(points.size(), 2, CV_32F);
    for(size_t i = 0; i < points.size(); ++i) 
    {
        inputData.at<float>(i, 0) = points[i].y;
        inputData.at<float>(i, 1) = points[i].x;
    }

    Mat clustersCenters, clustersCenters_1;

    Mat inputData_1 = inputData;
    Mat labels_1 = labels;


    static const int cIterations = 10000;
    static const float cEps = 0.001;
    static const int cTrials = 5;

    double compactness_1 = cv::kmeans(inputData_1, clusterCount_1, labels_1, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, cIterations, cEps), cTrials, cv::KMEANS_PP_CENTERS, clustersCenters_1);
    double compactness = cv::kmeans(inputData, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, cIterations, cEps), cTrials, cv::KMEANS_PP_CENTERS, clustersCenters);

    //printf("%d, %d\n%d, %d\n",clustersCenters.at<uchar>(0,0),clustersCenters.at<uchar>(0,1),clustersCenters.at<uchar>(1,0),clustersCenters.at<uchar>(1,1));

    /* cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
    em_model->setClustersNumber(clusterCount);
    em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
    em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, cIterations, cEps));
    em_model->trainEM( inputData, cv::noArray(), labels, cv::noArray() );
    cv::Mat means = em_model->getMeans();
    std::vector<cv::Mat> covs;
    em_model->getCovs(covs);

    printf("size(covariance) = (%d, %d)\n", covs[0].rows, covs[0].cols);*/

    Mat clusters(img.rows,img.cols, CV_8UC3, Scalar(0,0,0));

    Point center1, center2;
    center1 = clustersCenters.at<Point2f>(0);
    center2 = clustersCenters.at<Point2f>(1);

    //cv::Point center1(means.at<double>(0,1), means.at<double>(0,0));
    //cv::Point center2(means.at<double>(1,1), means.at<double>(1,0));

    int indicator = (center1.x<center2.x)?0:1; // 0 => center1 (0) is left, 1 => center2 (1) is left

    //cv::Mat im(img.rows,img.cols,CV_8UC3, cv::Scalar(0));
    //cv::Mat predicted(img.rows,img.cols,CV_32FC1, cv::Scalar(-1));

    //for(int i=0;i<means.rows;i++)
    //    cv::circle(im,cv::Point(means.at<double>(i,0),means.at<double>(i,1)),10,colorTab[i],-1);
    
    for( i = 0; i < sampleCount; i++ )
    {
        int clusterIdx = labels.at<int>(i);
        Point ipt = inputData.at<Point2f>(i);
        
        if(clusterIdx == 0)
        {
            if(indicator == 0)
            {
                
                left_lane.push_back(make_pair(ipt.y,ipt.x));
            }
            else
            {
                right_lane.push_back(make_pair(ipt.y,ipt.x));
            }
        }
        if(clusterIdx == 1)
        {
            if(indicator == 0)
            {
                right_lane.push_back(make_pair(ipt.y,ipt.x));
            }
            else
            {
                left_lane.push_back(make_pair(ipt.y,ipt.x));
            }
        }
        //circle(clusters, ipt, 2, colorTab[clusterIdx]);//, CV_FILLED, CV_LINE_AA );
        clusters.at<Vec3b>(ipt) = colorTab[clusterIdx];

        /*cv::Mat sample( 1, 2, CV_32FC1 );
        sample.at<float>(0) = (float)(ipt.x);
        sample.at<float>(1) = (float)(ipt.y);
        int response = cvRound(em_model->predict2( sample, cv::noArray() )[1]);
        cv::Scalar c = colorTab[response];
        predicted.at<float>(ipt.y,ipt.x) = (float)response;
        cv::circle( im, cv::Point(ipt.x, ipt.y), 1, c*0.75, -1 );*/
    }

    std::sort(left_lane.begin(), left_lane.end(), sort_pred);
    std::sort(right_lane.begin(), right_lane.end(), sort_pred);

    circle(clusters, center1, 20, Scalar(255,255,0));
    circle(clusters, center2, 20, Scalar(0,255,255));
    
    imshow("Clusters", clusters);
    //imshow("Clusters", im);
}

void fit_spline(Mat img)
{
    std::vector<double> LX, LY, RX, RY;
    for(int i = 0; i<left_lane.size();i++)
    {
        if(i<left_lane.size()-1)
        {
            if(left_lane[i].first != left_lane[i+1].first)
            {
                //cout<<left_lane[i].first<<" ";
                LX.push_back(left_lane[i].first);
                LY.push_back(left_lane[i].second);
            }
        }
    }
    cout<<"\n";
    for(int i = 0; i<right_lane.size();i++)
    {
        if(i<right_lane.size()-1)
        {
            if(right_lane[i].first != right_lane[i+1].first)
            {
                RX.push_back(right_lane[i].first);
                RY.push_back(right_lane[i].second);
            }
        }
    }

    s_left.set_points(LX,LY);
    s_right.set_points(RX,RY);

    //display the splines
    Mat splines(img.rows,img.cols,CV_8UC3,cv::Scalar(0,0,0));
    for(int i=0; i<splines.rows; i++)
    {
        int col_l = round(s_left(i));
        int col_r = round(s_right(i));

        if(col_l>=0 && col_l<splines.cols)
        {
            splines.at<Vec3b>(i,col_l) = Vec3b(255,0,0);
        }

        if(col_r>=0 && col_r<splines.cols)
        {
            splines.at<Vec3b>(i,col_r) = Vec3b(0,255,0);
        }
    }
    
    imshow("Splines", splines);
    imwrite("splines.jpg", splines);
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: ./spline ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   

    if(! image.data )                              
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );                   

    //clustered and left, right vectors created
    cluster(image);

    //spline fit
    fit_spline(image);


    waitKey(0); 
    return 0;
}