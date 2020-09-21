/******************************************
 * OpenCV Tutorial: Ball Tracking using   *
 * Kalman Filter                          *
 ******************************************/

// Module "core"
#include <opencv2/core/core.hpp>

// Module "highgui"
#include <opencv2/highgui/highgui.hpp>

// Module "imgproc"
#include <opencv2/imgproc/imgproc.hpp>

// Module "video"
#include <opencv2/video/video.hpp>

// Output
#include <iostream>


#include <eigen3/Eigen/Dense>


using namespace std;
using namespace Eigen;
using namespace cv;

// >>>>> Color to be tracked
#define MIN_H_BLUE 200
#define MAX_H_BLUE 300
// <<<<< Color to be tracked

void kalmanUpd(bool, float, float);
void kalmanPred(bool);
bool start = false;

Matrix4f A;
Matrix4f state;
Matrix4f H;
Matrix4f P;
Matrix4f Q;
Matrix4f R;
Matrix4f y;
Matrix4f K, S;
Matrix4f Si;
Matrix4f I;
Matrix2f S1, S2;

int main()
{
    int h_min = 137, h_max = 255, s_min = 69, s_max = 255, v_min = 153, v_max = 255; //omni baru
    
    cv::VideoCapture cap("your source video.mp4");

    if(!cap.isOpened())

        return -1;


    double fps = cap.get(CV_CAP_PROP_FPS);

    Mat frame, hsv, threshold;

    int count = 0;
    
    int tick = 0;
    int test = 0;
    float averageFps = 0.0;
    float frameCounter = 0;

    bool found;

    namedWindow("hsv");

    while (true)

    {
        
        test++;
        count++;
        
        cap >> frame;

        cv::Mat res;
        frame.copyTo( res );

        cvtColor(res,hsv,CV_BGR2HSV);

        inRange(hsv,Scalar(h_min,s_min,v_min),Scalar(h_max,s_max,v_max),threshold);
    createTrackbar("h_min: ","hsv",&h_min,255);
        createTrackbar("h_max: ","hsv",&h_max,255);
        createTrackbar("s_min: ","hsv",&s_min,255);
        createTrackbar("s_max: ","hsv",&s_max,255);
        createTrackbar("v_min: ","hsv",&v_min,255);
        createTrackbar("v_max: ","hsv",&v_max,255);
        
        // imshow("thresh",threshold);
        //---------------------- operasi erode dan dilate -------------------------//
       
        erode(threshold, threshold, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) ); // perkecil noise

        dilate( threshold, threshold, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) ); //perbesar noise

        //-----------------------(pilih salah satu saja)------------------------//
       
        // imshow("dilasi",threshold);
    // >>>>> Contours detection
        vector<vector<Point> > contours;
        findContours(threshold, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);

    // >>>>> Filtering
        vector<vector<Point> > balls;
        vector<Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            Rect bBox;
            bBox = boundingRect(contours[i]);

            float ratio = (float) bBox.width / (float) bBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;

            // Searching for a bBox almost square
            if (bBox.area() >= 0)
            {
                if(ratio > 0.75 ) {
                    balls.push_back(contours[i]);
                    ballsBox.push_back(bBox);
                }
                
            }
        }

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            //cv::circle(res, center, 2, CV_RGB(20,150,20), -1);

            cv::circle(res, cv::Point(center.x, center.y), 20, CV_RGB(255,0,0), 2.5);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            cv::putText(res, sstr.str(),
                        cv::Point(center.x + 13, center.y - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);
        
        cv::putText(res, "found",
                        cv::Point(10, 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        found = true;

        }
        

        if (balls.size() == 0) {
            
            
        }
        else{
            found = true;
            
            cv::Point center;
            center.x = ballsBox[0].x + ballsBox[0].width / 2;
            center.y = ballsBox[0].y + ballsBox[0].height / 2;
            kalmanUpd(start, center.x, center.y);
            start = true;
        }
        // <<<<< Detection result
        kalmanPred(start);
        cv::circle(res, cv::Point(state(0,0), state(1,0)), 20, CV_RGB(0,0,255), 2.5);

    imshow("detect",res);
	

        if(waitKey(20) != -1){
            imwrite("result.jpg", res);
            imwrite("ori.jpg", frame);
            break;
        }
    }

    return 0;
}

void kalmanUpd(bool start, float measX, float measY){
    
    A << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;
       
    H << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

    Q << 0.1, 0, 0, 0,
         0, 0.1, 0, 0,
         0, 0, 0.1, 0,
         0, 0, 0, 0.1;

    I << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    R << 1, 1, 0, 0,
         1, 1, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;



      y << measX, 0, 0, 0,
           measY, 0, 0, 0,
           0, 0, 0, 0,
           0, 0, 0, 0;

      if (!start){
          state << 0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0;

          P << 1000, 0, 0, 0,
              0, 1000, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;
          
        //   start = true;
      }
      
      S = H * P * H.transpose() + R; 
      S1(0,0) = S(0,0);
      S1(1,0) = S(1,0);
      S1(0,1) = S(0,1);
      S1(1,1) = S(1,1);
     
     S2 = S1.inverse();
     Si(0,0) = S2(0,0);
     Si(0,1) = S2(0,1);
     Si(1,0) = S2(1,0);
     Si(1,1) = S2(1,1);
      K = P * H.transpose() * Si;
      state = state + K * (y - H * state);
      P = (I - K * H) * P;
      cout << state << endl << endl;
    
    // cout << measX << endl;
}

void kalmanPred(bool start){

    A << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;

    Q << 100, 0, 0, 0,
         0, 100, 0, 0,
         0, 0, 100, 0,
         0, 0, 0, 100;

    state = A * state;
    P = A * P * A.transpose() + Q;
}


