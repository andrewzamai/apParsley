#include <opencv2/core.hpp> // openCV core module
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // openCV module used example for displaying an image
#include <opencv2/imgproc.hpp> // openCV module used example for calculating an image Histogram
#include <opencv2/features2d.hpp> // openCV module which contains SimpleBlobDetector

#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include "parsleyLib.hpp" // my library of functions

using namespace std;
using namespace cv;


void processImage(string imgPath, vector<Mat>& processedImages);


int main()
{
    // a vector where to save all processed images
    vector<Mat> processedImages;
    // FilePaths to the images I wish to process
    string rootPath = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/PerformanceSamples/";
    string imgPath = "";
    // constructs paths and process each single image
    for(int i=1; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            imgPath = rootPath + to_string(i) + to_string(j) + ".jpg";
    
            if(imgPath != "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/PerformanceSamples/13.jpg") //Layout1 has only 3 images
            {
                cout << imgPath << "\n";
                try
                {
                    //processedImages.clear();
                    processImage(imgPath, processedImages);
                    std::this_thread::sleep_for(1s);
                    // Saves all processed images
                    string pathToWrite = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/PerformanceSamples/ProcessedImages/" + to_string(i) + to_string(j) + ".tiff";
                    imwrite(pathToWrite, processedImages);
        
                }
                catch(exception &ex)
                {
                    cerr << ex.what() << "\n";
                }
            }
        }
    }
    
}

void processImage(string imgPath, vector<Mat>& processedImages)
{
    // Reads the image and saves it into img, an istance of OpenCV::Mat class
    Mat img = imread(imgPath, IMREAD_GRAYSCALE);
    
    // Starts calculating time to have an idea of processing time
    double ticks = (double) getTickCount();
    
    // Throws exception if couldn't load the specified image
    if(img.empty())
    {
        //cerr << "It wasn't possible to read the image at the specified path: " << imgPath << endl;
        throw runtime_error("It wasn't possible to read the image at the specified path: " + imgPath);
    }
    
    // Creates a named window in which to display the loaded image to be further processed
    //namedWindow("Displaying original image: ", WINDOW_AUTOSIZE);
    //imshow("Displaying original image: ", img);
    
    // adds original image to processed images
    processedImages.push_back(img);
    
    
    // Computes the image histogram and displays it in a new window
    Mat imgHistogram;
    parsleyLib::calculateCV_8UHistogram(img, imgHistogram);
    //imshow("Original image histogram: ", imgHistogram);
    processedImages.push_back(imgHistogram);
    
    
    // Applies gamma correction to the image
    double gamma = 1.5; // needs to be a double, a value between 1.0 and 3.0 is recommended. 1.5 is sperimentally choosen
    Mat gammaImage = img.clone();
    parsleyLib::applyGammaCorrection(gammaImage, gamma);
    //imshow("Image after Gamma Correction: ", img);
    
    // Computes new histogram after gamma correction
    Mat gammaImgHist; // image where to display gamma image histogram
    parsleyLib::calculateCV_8UHistogram(gammaImage, gammaImgHist);
    //imshow("New Histogram on Gamma Corrected img: ", gammaImgHist);
        
    // adds them to processed images
    processedImages.push_back(gammaImage);
    
    // Applies thresholding to get a Binary Image
    double thresholdingValue = parsleyLib::getAdaptiveThreshValue(gammaImage);
    // draws a thresholding vertical line on gammaImgHist at the calculated thresholdingValue by parsleyLib::getAdaptiveThreshValue // for debugging purposes
    int col = static_cast<int>(thresholdingValue*gammaImgHist.cols/256); // scales the thresholding value to the window size
    line(gammaImgHist, Point(col, 0), Point(col, gammaImgHist.rows-1), Scalar(0), 2, LINE_8, 0); // is not possible to draw a colored line on a CV_8U image
    //imshow("Thresholding value line on gamma img hist: ", gammaImgHist);
    processedImages.push_back(gammaImgHist);
    
    Mat binaryImage = gammaImage.clone();
    parsleyLib::toBinaryImage(binaryImage, thresholdingValue);
    //imshow("Binary Image: ", img);
    // adds binary image to processed images
    processedImages.push_back(binaryImage);
    

    // Blob detector and rotated bounding boxes steps
    
    // The following code is needed in order to create an istance of a SimpleBlobDetector
    SimpleBlobDetector::Params parameters = parsleyLib::instantiateBlobParams(); // only one instance is needed
    float minArea = 1500; // default value
    Ptr<SimpleBlobDetector> blobDetector;
    
    Mat imgWithKeypoints; // no needs for inizialization
    //Mat imgWithBoundingBoxes = Mat(img.size(), CV_8UC3, Scalar::all(255)); // to draw bounding boxes on a white image
    Mat imgWithBoundingBoxes;
    vector<KeyPoint> keypoints;
    
    blobDetector = parsleyLib::getBlobDetectorInstance(&parameters, minArea); // it is possible to change minArea filtering parameter at run time simply by calling getBlobDetectorInstance
    imgWithBoundingBoxes = imread(imgPath, IMREAD_COLOR); // to draw bounding boxes on the original image (to use different colors it needs to be read in IMREAD_COLOR MODE)
    //imgWithBoundingBoxes = binaryImage; // for drawing boxes on binary image
    //cvtColor(binaryImage, imgWithBoundingBoxes, COLOR_GRAY2BGR);
        
    keypoints = parsleyLib::boundingBlobDetect(blobDetector, binaryImage, imgWithKeypoints, imgWithBoundingBoxes);
        
    cout << "Number of detected objects for: "  << imgPath << ": " << keypoints.size() << endl;
        
    //imshow("Img with blob detected keypoints: ", imgWithKeypoints);
    //imshow("Img showing drawn bounding boxes: ", imgWithBoundingBoxes);
        
    // added to processed images
    processedImages.push_back(imgWithBoundingBoxes);
    
    // list of coordinates of all blob detected keypoints
    vector<Point2f> points2f;
    KeyPoint::convert(keypoints, points2f);
    cout << "The blob detected objects have center at the following coordinates: " << endl;
    for(int i=0; i<points2f.size();++i)
    {
        cout << "Object " << i << " : (" << points2f[i].x << ", " << points2f[i].y << ")" << "   (keypoint) Diameter: " << keypoints[i].size << endl; // float type values where (x,y) x is number of column, y number of row, from top left corner
    }
    
    
    // Blob detector and rotated bounding boxes steps for 1000 minArea
    
    // The following code is needed in order to create an istance of a SimpleBlobDetector
    parameters = parsleyLib::instantiateBlobParams(); // only one instance is needed
    minArea = 1000; // default value
    
    Mat img1000WithKeypoints; // no needs for inizialization
    //Mat imgWithBoundingBoxes = Mat(img.size(), CV_8UC3, Scalar::all(255)); // to draw bounding boxes on a white image
    Mat img1000WithBoundingBoxes;
    vector<KeyPoint> keypoints1000;
    
    blobDetector = parsleyLib::getBlobDetectorInstance(&parameters, minArea); // it is possible to change minArea filtering parameter at run time simply by calling getBlobDetectorInstance
    img1000WithBoundingBoxes = imread(imgPath, IMREAD_COLOR); // to draw bounding boxes on the original image (to use different colors it needs to be read in IMREAD_COLOR MODE)
    //imgWithBoundingBoxes = binaryImage; // for drawing boxes on binary image
    //cvtColor(binaryImage, imgWithBoundingBoxes, COLOR_GRAY2BGR);
        
    keypoints1000 = parsleyLib::boundingBlobDetect(blobDetector, binaryImage, img1000WithKeypoints, img1000WithBoundingBoxes);
        
    cout << "Number of detected objects for: "  << imgPath << ": " << keypoints1000.size() << endl;
        
    //imshow("Img with blob detected keypoints: ", imgWithKeypoints);
    //imshow("Img showing drawn bounding boxes: ", imgWithBoundingBoxes);
        
    // added to processed images
    processedImages.push_back(img1000WithBoundingBoxes);
    
    // list of coordinates of all blob detected keypoints
    points2f.clear();
    KeyPoint::convert(keypoints1000, points2f);
    cout << "The blob detected objects have center at the following coordinates: " << endl;
    for(int i=0; i<points2f.size();++i)
    {
        cout << "Object " << i << " : (" << points2f[i].x << ", " << points2f[i].y << ")" << "   (keypoint) Diameter: " << keypoints[i].size << endl; // float type values where (x,y) x is number of column, y number of row, from top left corner
    }
    
    // Stops the chrono and shows the elapsed time to process the image
    double elapsedTime = ((double)getTickCount() - ticks) / getTickFrequency();
    cout << "The program execution took " << elapsedTime << " seconds" << endl;

     
}


