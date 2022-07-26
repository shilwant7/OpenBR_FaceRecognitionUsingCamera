#include <openbr/openbr_plugin.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include<iostream>
#include<fstream>
#include<string>
#define THRESHOLD 2
using namespace std;
using namespace cv;
ofstream outfile;

static void printTemplate(const br::Template &t)
{
    const QPoint firstEye = t.file.get<QPoint>("Affine_0");
    const QPoint secondEye = t.file.get<QPoint>("Affine_1");
    printf("%s eyes: (%d, %d) (%d, %d)\n", qPrintable(t.file.fileName()), firstEye.x(), firstEye.y(), secondEye.x(), secondEye.y());
}

void ShowTargetImg(const char *path, int flag = 0)
{

    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    cv::namedWindow( "Your Result", WINDOW_NORMAL /*WINDOW_AUTOSIZE */);
    imshow("Your Result", image);
    if(flag){ // if image is matched
     //write into file
    }
}

void NotMatch()
{
    cout<<"\n\nDo not match with gallery !"<<endl<<endl;
    ShowTargetImg("/home/openBRex/imgs/unknown.jpeg");
}

void Match(const br::Template &cam_frame, const br::Template &target, float match_score)
{
    printf("Images %s and %s have a similarity score of %.3f\n",
                qPrintable(cam_frame.file.name),
                qPrintable(target.file.name),
                match_score);
    ShowTargetImg(qPrintable(target.file.name), 1);
}

int main(int argc, char** argv)
{
    int count1 = 1;
    br::Context::initialize(argc, argv);
    br::Globals->enrollAll = true; // Enroll 0 or more faces per image

    outfile.open("face.txt", ios::out |ios::ate| ios::ate );
    if(outfile.is_open())
        cout<<"the file is open"<<endl;
    else
        cout<<"the file is not open"<<endl;

    cv::VideoCapture cap("rtsp://admin:password@ip/h264/ch1/main/av_stream?tcp");
//     VideoCapture cap("/home/vid.mp4");//for video

    // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm

    QSharedPointer<br::Transform> transform_fd = br::Transform::fromAlgorithm("Open+Cascade(FrontalFace)+ASEFEyes");//+ASEFEyes
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("FaceRecognition");
    QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");

    br::TemplateList target = br::TemplateList::fromGallery("/home/openBRex/gslab_dataset");

    br::Globals->enrollAll = true; // Enroll 0 or more faces per image
    target >> *transform;
//     cout<<"Printing Template"<<endl;
//     printTemplate(target);
    cv::namedWindow( "camera", WINDOW_NORMAL /*WINDOW_AUTOSIZE */);
    if(!cap.isOpened())
        return -1;

    for(;;)
    {

        Mat frame;
        cap >> frame;

        if( frame.empty() )
            break; // end of video stream

        br::Template queryB(frame);

        br::TemplateList cam_querylist;
        cam_querylist.push_back(queryB);//inserting queryB i.e. template in template list
        cam_querylist >> *transform_fd;

        cv::Mat vis = frame;

        for ( auto query : cam_querylist) {//to display faces
            const QPoint firstEye = query.file.get<QPoint>("First_Eye");
            const QPoint secondEye = query.file.get<QPoint>("Second_Eye");
            const QRect faceroi = query.file.get<QRect>("FrontalFace");

            cv::Point fe(firstEye.x(), firstEye.y());
            cv::Point se(secondEye.x(), secondEye.y());
            cv::Point te(firstEye.x() + 30, firstEye.y());

//             cv::circle(vis, fe, 5.0, cv::Scalar(255, 0, 0), 5);
//             cv::circle(vis, se, 5.0, cv::Scalar(0, 0, 255),5);
//              cv::rectangle(vis, Point (faceroi.x(),faceroi.y()), Point (faceroi.x()+faceroi.width(),faceroi.y()+faceroi.height()), Scalar(0x0,0xff,0xff),10,8,0);


//             cv::Mat ROI = vis(Rect(faceroi.x()-20, faceroi.y()-20 , faceroi.width()+20, faceroi.height()+20));
            cv::Mat ROI = vis(Rect(faceroi.x(), faceroi.y() , faceroi.width(), faceroi.height()));
            br::Template t1(ROI);
            t1 >> *transform;

            float scores;
            float prev_score = 0;
            // Print an example score
            br::Template t_target ;
            for ( auto trav_target : target) {//find best matched image
                scores = distance->compare(trav_target, t1/*queryB*/);
                if (scores > prev_score /*&& scores > 0*/){//for temp.storage
                    prev_score = scores;
                    t_target = trav_target;
                    if(scores > THRESHOLD) //no need to continue loop if matching score is good
                        break;

                }
                else{//
                    ShowTargetImg("/home/shilwant/openBRex/img_database_used/wait.jpeg");
                }
            }

            if(prev_score > 0){
                if(prev_score > THRESHOLD){//show result for > 1 value
                    cv::rectangle(vis, Point (faceroi.x(),faceroi.y()), Point (faceroi.x()+faceroi.width(),faceroi.y()+faceroi.height()), Scalar(0x0,0x80,0x0),10,8,0);
                    char text[20];
                    sprintf(text, "Found_%.3f", prev_score);
                    putText(vis, text, Point (faceroi.x(),faceroi.y()), FONT_HERSHEY_COMPLEX_SMALL, 4, cvScalar(0x0,0x80,0x0), 6, CV_AA);
                    Match(queryB, t_target, prev_score);

                }
                else{ //if matching % is < 1 Don't treat it
                    cv::rectangle(vis, Point (faceroi.x(),faceroi.y()), Point (faceroi.x()+faceroi.width(),faceroi.y()+faceroi.height()), Scalar(0,0,255),10,8,0);
                    putText(vis, "?", Point (faceroi.x(),faceroi.y()), FONT_HERSHEY_COMPLEX_SMALL, 5, cvScalar(0,0,255), 6, CV_AA);
                    NotMatch();
                }
            }
        }

        imshow("camera", vis);
        cv::waitKey(1);
//         if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    //   cap.close();

    br::Context::finalize();
    return 0;
}
