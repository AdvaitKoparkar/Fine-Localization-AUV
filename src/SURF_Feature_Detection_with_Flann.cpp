# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
#include  "opencv2/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <unistd.h>

#define DELAY 50
#define NUMBER_OF_FRAMES 2
#define MIN_HESSIAN 200

using namespace cv;

class Frame
{
private:
  int minHessian;
  cv::Mat frame[NUMBER_OF_FRAMES];
  std::vector<KeyPoint> keypoints[2];
  cv::Mat descriptors[2];
  std::vector<DMatch> matches;
  SurfDescriptorExtractor extractor;
public:
  Frame() : minHessian(MIN_HESSIAN) {}
  void setFrame(cv::Mat img, int i)
  {
    if(i>NUMBER_OF_FRAMES-1)
      return;
    cv::cvtColor(img, frame[i], CV_BGR2GRAY);
    // frame[i] = img.clone();
  }
  void getFrame(cv::Mat &img, int i)
  {
    if(i > NUMBER_OF_FRAMES-1)
      return;
    img = frame[i];
  }
  void shift()
  {
    frame[0] = frame[1];
  }
  void extractDescriptor(int i)
  {
    if (i > NUMBER_OF_FRAMES)
      return;

    if(i == -1)
    {
      for(int j = 0; j < NUMBER_OF_FRAMES; j++)
        extractor.compute(frame[j], keypoints[j], descriptors[j]);
      return;
    }
    extractor.compute(frame[i], keypoints[i], descriptors[i]);
  }
  void detectKeypoints(int i)
  {
    if (i > NUMBER_OF_FRAMES)
      return;

    cv::SurfFeatureDetector detector(minHessian);
    if (i == -1)
    {
      cv::SurfFeatureDetector detector(minHessian);
      for(int j = 0; j < NUMBER_OF_FRAMES; j++)
      {
        detector.detect(frame[j], keypoints[j]);
      }
      return;
    }

    detector.detect(frame[i], keypoints[i]);
    return;
  }
  void match(int i, int j)
  {
    FlannBasedMatcher matcher;
    matcher.match(descriptors[i], descriptors[j], matches);
  }
  void draw(cv::Mat &imgMatches)
  {
    cv::drawMatches(frame[0], keypoints[0], frame[1], keypoints[1], matches, imgMatches, Scalar::all(-1), Scalar::all(-1),
    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  }
};

int main( int argc, char** argv )
{
  cv::VideoCapture cap(0);
  if(!cap.isOpened()) return -1;

  cv::Mat img;
  cv::namedWindow("Output", 1);

  double t1;
  double t0;

  Frame frameObj;

  cv::Mat tmp;
  cap >> tmp;
  frameObj.setFrame(tmp, 0);
  frameObj.setFrame(tmp, 1);

  while(true)
  {
    t0 = cv::getTickCount();
    frameObj.shift();
    cv::waitKey(DELAY);
    cap >> tmp;
    frameObj.setFrame(tmp, 1);
    frameObj.detectKeypoints(-1);
    frameObj.extractDescriptor(-1);
    frameObj.match(0, 1);
    frameObj.draw(img);
    t1 = cv::getTickCount();
    std::cout << (t1-t0)/cv::getTickFrequency() << " SURF Detection with FlannMatching " <<  std::endl;
    cv::imshow("Output", img);

    if(cv::waitKey(DELAY) == 27)
      break;
  }

  return 0;
}
