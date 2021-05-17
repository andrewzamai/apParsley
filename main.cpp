// APParsley: un sistema di visione artificiale per l'individuazione di impurità tra foglie di prezzemolo essiccate
// Andrew Zamai
#include <opencv2/core.hpp> // openCV core module
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // openCV module used example for displaying an image
#include <opencv2/imgproc.hpp> // openCV module used example for calculating an image Histogram
#include <opencv2/features2d.hpp> // openCV module which contains SimpleBlobDetector
#include <iostream>
#include <string>
#include "parsleyLib.hpp" // my library of functions

using namespace std;
using namespace cv;

int main()
{
    cout << "APParsley: a computer vision system for impurities detection among dried parsley leaves." << endl;
    cout << "Insert path to the image you wish to process: " << endl;
    string inputImgPath;
    cin >> inputImgPath;
    
    cout << "Insert directory path where to save the output images for each program step: " << endl;
    string outputImgPath;
    cin >> outputImgPath;
    
    parsleyLib::processImage(inputImgPath, outputImgPath);
    
    return 0;
}

