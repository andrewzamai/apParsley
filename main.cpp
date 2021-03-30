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
    // FilePath to the image I wish to process
    string path0 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/1_Layout/Mono/IMG_9234.jpg";
    string path1 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/2_Layout/Mono/IMG_9444.jpg";
    string path2 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/4_Layout/Mono/IMG_0041.jpg";
    string path3 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/4_Layout/Mono/IMG_0011.jpg";
    string path4 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/4_Layout/Mono/IMG_0101.jpg";
    string path5 = "/Users/andrew/Desktop/Università/Laurea/ProgettoPrezzemolo/FotoPrezzemolo0903/5_Layout/Mono/IMG_0197.jpg";
    
    string paths[] = {path0, path1, path2, path3, path4, path5};
    
    cout << "Welcome, to start digit a number from 0 to 5 corresponding to a path to an image you wish to process: " << endl;
    int selection = 0;
    cin >> selection;
    
    string imgPath = paths[selection];
    
    
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
    parsleyLib::calculateCV_8UHistogram(img, imgHistogram);
    imshow("Original image histogram: ", imgHistogram);
    
    
    // Applies gamma correction to the image, the program promps the user to specify manually the gamma value
    bool repeat = true; // ask again by default
    string user = "";
    double gamma = 0.0; // needs to be a double, a value between 1.0 and 3.0 is recommended
    Mat gammaImage;
    Mat gammaImgHist; // image where to display histogram
    
    do{
        cout << "Specify a gamma value for gamma correction (a value between 1.0 and 3.0 is recommended): " << endl;
        cin >> gamma;
        
        // Applies gamma correction to the image
        gammaImage = img.clone(); // cloning the image can require computational time, however not doing so will cause the program in this loop to apply a new gamma correction on an alredy gamma corrected image
        parsleyLib::applyGammaCorrection(gammaImage, gamma); //gamma value is chosen after some sperimental attempts
        imshow("Image after Gamma Correction: ", gammaImage);
        
        // Computes new histogram after gamma correction
        
        parsleyLib::calculateCV_8UHistogram(gammaImage, gammaImgHist);
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
    double thresholdingValue = parsleyLib::getAdaptiveThreshValue(gammaImage);
    Mat binaryImage;
    
    binaryImage = gammaImage.clone();
    
    // draw thresholding line on gammaImgHist
    int col = static_cast<int>(thresholdingValue*gammaImgHist.cols/256); // scales the thresholding value to the window size
    line(gammaImgHist, Point(col, 0), Point(col, gammaImgHist.rows-1), Scalar(255), 2, LINE_8, 0); // is not possible to draw a colored line on a CV_8U image
    imshow("Thresholding value line on gamma img hist: ", gammaImgHist);
    
    parsleyLib::toBinaryImage(binaryImage, thresholdingValue);
    imshow("Binary Image: ", binaryImage);
    
    waitKey(1);
        
    // added to processed images
    processedImages.push_back(binaryImage);
    
    cout << "Alredy displayed image above" << endl;
    
    // The following code is needed in order to create an istance of a SimpleBlobDetector
    SimpleBlobDetector::Params parameters = parsleyLib::instantiateBlobParams(); // only one instance is needed
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
        
        blobDetector = parsleyLib::getBlobDetectorInstance(&parameters, minArea); // it is possible to change minArea filtering parameter at run time simply by calling getBlobDetectorInstance
        
        imgWithBoundingBoxes = imread(imgPath, IMREAD_COLOR); // to draw bounding boxes on the original image (to use different colors it needs to be read in IMREAD_COLOR MODE)
        
        keypoints = parsleyLib::boundingBlobDetect(blobDetector, binaryImage, imgWithKeypoints, imgWithBoundingBoxes);
        
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


