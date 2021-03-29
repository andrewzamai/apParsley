#include <opencv2/core.hpp> // openCV core module
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // openCV module used example for displaying an image
#include <opencv2/imgproc.hpp> // openCV module used example for calculating an image Histogram
#include <opencv2/features2d.hpp> // openCV module which contains SimpleBlobDetector

#include <iostream>
#include <string>

#include "myLib.hpp" // my library of functions

using namespace std;
using namespace cv;


int main()
{
    cout << "Welcome, to start specify the path to the image you wish to process: " << endl;
    // FilePath to the image I wish to process
    string imgPath = "";
    cin >> imgPath;
    // some example paths
    //imgPath = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/2_Layout/Mono/IMG_9444.jpg";
    //imgPath = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/5_Layout/Mono/IMG_0197.jpg";
    
    // a vector for saving all produced images
    vector<Mat> processedImages;
    
    // Reads the image and saves it into img, an istance of OpenCV::Mat class
    Mat img = imread(imgPath, IMREAD_GRAYSCALE);
    
    // Starts calculating time to have an idea of processing time
    double ticks = (double) getTickCount();
    
    // Quits the program if couldn't load the specified image
    if(img.empty())
    {
        cerr << "It wasn't possible to read the image at the specified path: " << imgPath << endl;
        return 1;
    }
    
    // Creates a named window in which to display the loaded image to be further processed
    namedWindow("Displaying original image: ", WINDOW_AUTOSIZE);
    imshow("Displaying original image: ", img);
    
    // added to processed images
    processedImages.push_back(img);
    
    // Shows some info about the loaded image
    cout << "Image type: " << img.depth() << endl;
    cout << "Size: " << img.size << endl;
    cout << "Dimension: " << img.dims << endl;
    cout << "Number of rows: " << img.rows << ", number of columns: " << img.cols << endl;
    
    // Computes the image histogram and displays it in a new window
    Mat imgHistogram;
    myLib::calculateCV_8UHistogram(img, imgHistogram);
    imshow("Original image histogram: ", imgHistogram);
    
    
    // Applies gamma correction to the image, the program promps the user to specify manually the gamma value
    bool repeat = true; // ask again by default
    string user = "";
    double gamma = 0.0; // needs to be a double, a value between 1.0 and 3.0 is recommended
    Mat gammaImage;
    
    do{
        cout << "Specify a gamma value for gamma correction (a value between 1.0 and 3.0 is recommended): " << endl;
        cin >> gamma;
        
        // Applies gamma correction to the image
        gammaImage = img.clone(); // cloning the image can require computational time, however not doing so will cause the program in this loop to apply a new gamma correction on an alredy gamma corrected image
        myLib::applyGammaCorrection(gammaImage, gamma); //gamma value is chosen after some sperimental attempts
        imshow("Image after Gamma Correction: ", gammaImage);
        
        // Computes new histogram after gamma correction
        Mat gammaImgHist;
        myLib::calculateCV_8UHistogram(gammaImage, gammaImgHist);
        imshow("New Histogram on Gamma Corrected img: ", gammaImgHist);
        
        waitKey(1);
        
        cout << "Proceed? ( 'y' for yes, 'r' for repeat) " << endl;
        cin >> user;
        if(user.compare("y") == 0)
            repeat = false;
        
    } while(repeat);
    
    // added to processed images
    processedImages.push_back(gammaImage);
    
    // Applies thresholding to get a Binary Image
    repeat = true;
    user.clear();
    double thresholdingValue = 0.0;
    Mat binaryImage;
    
    do{
        cout << "Specify a thresholding value: " << endl;
        cin >> thresholdingValue;
        
        binaryImage = gammaImage.clone();
        myLib::toBinaryImage(binaryImage, thresholdingValue);
        imshow("Binary Image: ", binaryImage);
        
        waitKey(1);
        
        cout << "Proceed? ( 'y' for yes, 'r' for repeat) " << endl;
        cin >> user;
        if(user.compare("y") == 0)
            repeat = false;
        
    } while(repeat);
    
    // added to processed images
    processedImages.push_back(binaryImage);
    
    // The following code is needed in order to create an istance of a SimpleBlobDetector
    SimpleBlobDetector::Params parameters = myLib::instantiateBlobParams(); // only one instance is needed
    float minArea = 250; // default value
    Ptr<SimpleBlobDetector> blobDetector;
    
    Mat imgWithKeypoints; // no needs for inizialization
    //Mat imgWithBoundingBoxes = Mat(img.size(), CV_8UC3, Scalar::all(255)); // to draw bounding boxes on a white image
    Mat imgWithBoundingBoxes;
    vector<KeyPoint> keypoints;
    repeat = true;
    user.clear();
    
    do {
        cout << "Specify a minimum Area for the object to be detected: " << endl;
        cin >> minArea;
        
        blobDetector = myLib::getBlobDetectorInstance(&parameters, minArea); // it is possible to change minArea filtering parameter at run time simply by calling getBlobDetectorInstance
        
        imgWithBoundingBoxes = imread(imgPath, IMREAD_COLOR); // to draw bounding boxes on the original image (to use different colors it needs to be read in IMREAD_COLOR MODE)
        
        keypoints = myLib::boundingBlobDetect(blobDetector, binaryImage, imgWithKeypoints, imgWithBoundingBoxes);
        
        cout << "Number of detected objects: " << keypoints.size() << endl;
        
        imshow("Img with blob detected keypoints: ", imgWithKeypoints);
        imshow("Img showing drawn bounding boxes: ", imgWithBoundingBoxes);
        
        waitKey(1);
        
        cout << "Proceed? ( 'y' for yes, 'r' for repeat) " << endl;
        cin >> user;
        if(user.compare("y") == 0)
            repeat = false;
        
    } while(repeat);
    
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
    
    
    // Stops the chrono and shows the elapsed time to process the image
    double elapsedTime = ((double)getTickCount() - ticks) / getTickFrequency();
    cout << "The program execution took " << elapsedTime << " seconds" << endl;

    // Saves all processed images
    imwrite("/Users/andrew/Desktop/ProcessedImages/ProcessedImages.tiff", processedImages);
    
    
    // Waits an indeterminate time that the users presses any key (this is to let time to display the image)
    int k = waitKey(0);
    if(k == 'q')
    {
        cout << "Program has been quitted correctly by the user." << endl;
        return 0;
    }
    
    return 0;
}


