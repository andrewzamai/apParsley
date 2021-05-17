// APParsley: un sistema di visione artificiale per l'individuazione di impurità tra foglie di prezzemolo essiccate
// Andrew Zamai
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "parsleyLib.hpp"
#include <iostream>

using namespace std;
using namespace cv;

namespace parsleyLib {

void processImage(string inputImgPath, string outputImgPath)
{
    // Starts calculating time to have an idea of processing time
    double ticks = (double) getTickCount();
    
    // a vector where to save all processed images
    vector<Mat> processedImages;
    // Reads the image and saves it into img, an istance of OpenCV::Mat class
    Mat img = imread(inputImgPath, IMREAD_GRAYSCALE);
    
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
    

    // Blob detector and rotated bounding boxes steps:
    
    // The following code is needed in order to create an istance of a SimpleBlobDetector
    SimpleBlobDetector::Params parameters = parsleyLib::instantiateBlobParams(); // only one instance is needed
    float minArea = 1500; // default value
    Ptr<SimpleBlobDetector> blobDetector;
    
    Mat imgWithKeypoints; // no needs for inizialization
    //Mat imgWithBoundingBoxes = Mat(img.size(), CV_8UC3, Scalar::all(255)); // to draw bounding boxes on a white image
    Mat imgWithBoundingBoxes;
    vector<KeyPoint> keypoints;
    
    blobDetector = parsleyLib::getBlobDetectorInstance(&parameters, minArea); // it is possible to change minArea filtering parameter at run time simply by calling getBlobDetectorInstance
    imgWithBoundingBoxes = imread(inputImgPath, IMREAD_COLOR); // to draw bounding boxes on the original image (to use different colors it needs to be read in IMREAD_COLOR MODE)
    //imgWithBoundingBoxes = binaryImage; // for drawing boxes on binary image
    //cvtColor(binaryImage, imgWithBoundingBoxes, COLOR_GRAY2BGR);
        
    keypoints = parsleyLib::boundingBlobDetect(blobDetector, binaryImage, imgWithKeypoints, imgWithBoundingBoxes);
        
    cout << "Number of detected objects for: "  << inputImgPath << ": " << keypoints.size() << endl;
        
    //imshow("Img with blob detected keypoints: ", imgWithKeypoints);
    //imshow("Img showing drawn bounding boxes: ", imgWithBoundingBoxes);
        
    // added to processed images
    processedImages.push_back(imgWithBoundingBoxes);
    
    // list of coordinates of all blob detected keypoints
    vector<Point2f> points2f;
    KeyPoint::convert(keypoints, points2f);
    cout << "The detected objects have centers at the following coordinates (pixelCol, pixelRow): " << endl;
    for(int i=0; i<points2f.size();++i)
    {
        cout << "Object " << i << " : (" << (int)points2f[i].x << ", " << (int)points2f[i].y << ")" << endl; // float type values where (x,y) x is number of column, y number of row, from top left corner
    }
    
    string outputPath = outputImgPath + "/processedImages.tiff";
    imwrite(outputPath, processedImages);
    
    
    // Stops the chrono and shows the elapsed time to process the image
    double elapsedTime = ((double)getTickCount() - ticks) / getTickFrequency();
    cout << "Single image execution time: " << elapsedTime << " seconds" << endl;
    
    
}


void calculateCV_8UHistogram(const Mat& image, Mat& imageToDisplayWhere)
{
    const int histSize = 256; // being an 8 bit image there will be 256 intensities of gray
    
    float range[] = {0, 256}; // array of float to specify x-axis values
    const float* histRange = {range}; // array of pointers, cause cv::calcHist need a double pointer param
    
    bool uniform = true; // bins (columns) to have all same width
    bool accumulate = false; // to clear all data from previous histograms
    
    // calling cv::calcHist static function with the above specified parameters
    Mat imgDataHist;
    calcHist(&image, 1, 0, Mat(), imgDataHist, 1, &histSize, &histRange, uniform, accumulate);
    
    // creates new Mat type image in which to display the hist given the calculated imgDataHist
    int histHeight = 3024;
    int histWidth = 3024;
    int binWidth = cvRound( (double) histWidth/histSize); // each bin width
    Mat histImageToDisplay(histHeight, histWidth, CV_8U, Scalar(255));
    
    // normalizes histogram to fit into histImageToDisplay
    normalize(imgDataHist, imgDataHist, 0, histImageToDisplay.rows, NORM_MINMAX, -1, Mat());
    
    // uses cv::line function to draw the computed histogram on the to display image
    for(int i = 1; i < histSize; i++)
        {
            line(histImageToDisplay, Point( binWidth*(i-1), histHeight - cvRound(imgDataHist.at<float>(i-1))),
                  Point(binWidth*(i), histHeight - cvRound(imgDataHist.at<float>(i))),
                  Scalar(0, 0, 0), 3, LINE_8, 0);
            //cout << " Value " << i-1 << ": " << cvRound(imgDataHist.at<float>(i-1)) << endl;
        }
    
    imageToDisplayWhere = histImageToDisplay; // uses Mat class assigment operator
}


void applyGammaCorrection(Mat& image, double gamma)
{
    // 1x256 image used as lookUpTable vector, static so it fill be inizialez only once ?!
    static Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    
    for(int i=0; i<256; ++i)
    {
        p[i] = saturate_cast<uchar>(pow(i/255.0, gamma) * 255.0); // saturate_cast ensures that the calculated value is in the 0-255 char range
    }
    
    LUT(image, lookUpTable, image);
}

// Adapter like function
void toBinaryImage(Mat& image, double thresholdingValue)
{
    double maxValue = 255; // 8-bit image CV_8U
    int type = 0; // Binary thresholding
    
    threshold(image, image, thresholdingValue, maxValue, type);
}


SimpleBlobDetector::Params instantiateBlobParams()
{
    SimpleBlobDetector::Params parameters = SimpleBlobDetector::Params();
    
    parameters.filterByColor = true;
    parameters.blobColor = 255;
    parameters.minDistBetweenBlobs = 1;
    parameters.minThreshold = 1;
    parameters.maxThreshold = 255;
    
    parameters.filterByArea = true;
    parameters.minArea = 1500;       // 1500 default value
    parameters.maxArea = 500000;
    
    parameters.filterByCircularity = false;
    parameters.minCircularity = 0.1;
    parameters.filterByConvexity = false;
    parameters.minConvexity = 0.87;
    parameters.filterByInertia = false;
    parameters.minInertiaRatio = 0.01;
    
    return parameters;
}


Ptr<SimpleBlobDetector> getBlobDetectorInstance(SimpleBlobDetector::Params* parameters, float minArea)
{
    parameters->minArea = minArea;
    return SimpleBlobDetector::create(*parameters);
}


vector<KeyPoint> boundingBlobDetect(Ptr<SimpleBlobDetector> blobDetector, Mat& binaryImage, Mat& imgWithKeypoints, Mat& imgWithBoundingBoxes)
{
    // calling detect function of blobDetector
    vector<KeyPoint> keypoints;
    blobDetector -> detect(binaryImage, keypoints);
    // draws red circles on detected keypoints
    drawKeypoints(binaryImage, keypoints, imgWithKeypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
    // Uses cv::findContours method applied to the Binary Image, to locate all patches of white pixels
    vector<vector<Point>> contoursSet; // a vector of patches, eatch patch of points it's in turn described by a vector of Point
    findContours(binaryImage, contoursSet, RETR_EXTERNAL, CHAIN_APPROX_NONE); // RETR_EXTERNAL to reject inner contours
    
    // Given each patch creates the minimum rotated bounding rectangle which contains it
    vector<RotatedRect> boundingRects(contoursSet.size());
    for(int i=0; i<contoursSet.size(); ++i)
    {
        boundingRects[i] = minAreaRect(contoursSet[i]);
    }
    
    
    vector<RotatedRect> boundingRectsCopy(boundingRects); // clones boundingRects vector before vector difference
    eliminateBlobDetectedRect(keypoints, boundingRectsCopy); // vectors of non blob detected bounding rectangles
    vectorsDifference(boundingRects, boundingRectsCopy); // vectors of only blob detected bounding rectangles
    
    drawRotatedBoundingBoxes(imgWithBoundingBoxes, boundingRects);
    
    //cout << "Total number of bounded pixels: " << totalPixels(boundingRects) << endl;
    
    return keypoints;
}





/*------------------------------------------- Helper functions below: --------------------------------------------------*/





double getAdaptiveThreshValue(const cv::Mat& image)
{
    const int histSize = 256; // being an 8 bit image there will be 256 intensities of gray
    
    float range[] = {0, 256}; // array of float to specify x-axis values
    const float* histRange = {range}; // array of pointers, cause cv::calcHist need a double pointer param
    
    bool uniform = true; // bins (columns) to have all same width
    bool accumulate = false; // to clear all data from previous histograms
    
    // calling cv::calcHist static function with the above specified parameters
    Mat imgDataHist;
    calcHist(&image, 1, 0, Mat(), imgDataHist, 1, &histSize, &histRange, uniform, accumulate);
    
    // each element of the vector will represent the number of pixel of intensity same as the element position i
    MatIterator_<float> it, end;
    vector<float> intensities; // cv::calcHist fills a Mat object of float elements
    for( it=imgDataHist.begin<float>(), end=imgDataHist.end<float>(); it != end; ++it)
    {
        intensities.push_back(*it);
    }
    
    // Prints intensity values // Just for debugging purposes
    /*
    cout << "Number of intensities: " << intensities.size() << endl;
    cout << "List of all intensities: " << endl;
    for(int i=0; i<intensities.size(); ++i)
    {
        cout << intensities.at(i) << endl;
    }
    */
    double y2 = 0.0;
    double y1 = 0.0;
    double m = 0.0; // slope
    // max slope over which the algotithm ends, returning the thresholding value equal to the i position reached
    // experimental value founded by analyzing histograms of some gamma corrected images, seeing that all hists present this thresholding slope where impurities intensities end and parsley/background intensities start
    double thresholdingSlope = 1000000;
    int window = 2; // due to the gamma image histogram irregolarity, calculates the mean of 2 points instead of 1, to eventually calculate the slope by (y2-y1)/window
    
    // i acts as an iterator over the vector, scanning it from right to left, calculating for each iteration the slope of the gamma histogram at that point
    // y1 represents the mean of the intensities at positions i and i+1    [i to i+window[ if window > 2
    // y2 represents the mean of the intensities at positions i-1 and i-2   [i-window, i[ if window > 2
    for(int i=intensities.size()-window; i>window; --i)
    {
        y1 = 0.0;
        for(int j=i; j<i+window; ++j)
        {
            y1 += intensities.at(j);
        }
        y1 = y1/window; // calculates mean
        
        y2 = 0.0;
        for(int j=i-1; j>i-window; --j)
        {
            y2 += intensities.at(j);
        }
        y2 = y2/window;
        
        m = - ((y2-y1)/window) * 100; // calculates slope in percentage, minus because we are going from right to left (x2 < x1), divided by window as we can assume x2-x1 = window
        
        //cout << "Slope: " << m << endl; // Just for debugging purposes
        
        // stops after a big slope: impurities are just a small percentage of overall image, yet not negligible. A big positive slope separates a first number of significant pixels (impurities) at its right
        if(m > thresholdingSlope)
        {
            //cout << "Computed Thresholding value: " << i << endl;
            return i;
        }

    }
    return -1;
}


bool pointBelongsToRect(const RotatedRect& rect, Point2f point)
{
    Rect minBoundingRect = rect.boundingRect(); // gets the best non rotated rectangle which contains rect
    return minBoundingRect.contains(point); // uses alredy implemented Rect member function
}


void eliminateBlobDetectedRect(const vector<KeyPoint>& keypoints, vector<RotatedRect>& rotRects)
{
    for(int i=0; i<keypoints.size(); ++i)
    {
        vector<RotatedRect>::iterator it = rotRects.begin();
        for( ; it != rotRects.end(); )
        {
            RotatedRect r = *it;
            if(pointBelongsToRect(r, keypoints[i].pt))
                it = rotRects.erase(it);
            else
                ++it;
        }
    }
}


void vectorsDifference(vector<RotatedRect>& vectorCopy, const vector<RotatedRect>& vector)
{
    for(int i=0; i<vector.size(); ++i)
    {
        Point2f vPoint = vector[i].center;
        
        std::vector<RotatedRect>::iterator it = vectorCopy.begin();
        for( ; it != vectorCopy.end(); )
        {
            // compares the RotatedRect elements from the two vectors relying on their center point
            Point2f vCopyPoint = (*it).center;
            if(vCopyPoint == vPoint)
                it = vectorCopy.erase(it);
            else
                ++it;
        }
    }
}


bool isRect(RotatedRect r)
{
    double side1 = r.size.width;
    double side2 = r.size.height;
    
    if(side1 > 2*side2 || side2 > 2*side1)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void drawRotatedBoundingBoxes(Mat& imgToDraw, const vector<RotatedRect>& rotRectangles)
{
    for(int i=0; i<rotRectangles.size(); ++i)
    {
        Point2f rectPoints[4];
        rotRectangles[i].points(rectPoints);
        for(int j=0; j<4; ++j)
        {
            if(isRect(rotRectangles[i]))
                line(imgToDraw, rectPoints[j], rectPoints[(j+1)%4], Scalar(0,0,255), 5);
            else
                line(imgToDraw, rectPoints[j], rectPoints[(j+1)%4], Scalar(255,0,0), 5);
            
            // cout << "A vertex of rotated rectangle number " << i << " is: " << rectPoints[j] << endl;
            
        }
        // cout << endl;
    }
}



}
