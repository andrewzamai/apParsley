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
    Mat histImageToDisplay(histHeight, histWidth, CV_8U, Scalar(0));
    
    // normalizes histogram to fit into histImageToDisplay
    normalize(imgDataHist, imgDataHist, 0, histImageToDisplay.rows, NORM_MINMAX, -1, Mat());
    
    // uses cv::line function to draw the computed histogram on the to display image
    for(int i = 1; i < histSize; i++)
        {
            line(histImageToDisplay, Point( binWidth*(i-1), histHeight - cvRound(imgDataHist.at<float>(i-1))),
                  Point(binWidth*(i), histHeight - cvRound(imgDataHist.at<float>(i))),
                  Scalar(255, 0, 0), 2, LINE_8, 0);
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
    
    parameters.filterByColor = 1;
    parameters.blobColor = 255;
    parameters.minDistBetweenBlobs = 1;
    parameters.minThreshold = 0;
    parameters.maxThreshold = 255;
    
    parameters.filterByArea = true;
    parameters.minArea = 150;       // DA SETTARE MAGARI CON TRACKBAR
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
    /* when I pass a reference in the main calling this function it will never be a nullptr!
    if(parameters == nullptr)
    {
        cout << "A null parameters pointer was passed: a new instance of parameters was created!" << endl;
        *parameters = instantiateBlobParams();
    }
    */
    
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
    
    // Prints intensity values
    MatIterator_<float> it, end;
    vector<float> intensities;
    for( it=imgDataHist.begin<float>(), end=imgDataHist.end<float>(); it != end; ++it)
    {
        intensities.push_back(*it);
    }
    
    //prints intensity values
    cout << "Number of intensities: " << intensities.size() << endl;
    for(int i=0; i<intensities.size(); ++i)
    {
        cout << intensities.at(i) << endl;
    }
    
    double y2 = 0.0;
    double y1 = 0.0;
    double m = 0.0;
    int window = 2; // half window size //valutare se aumentare finestra, cambiare quindi anche pendenza?
    
    for(int i=intensities.size()-window; i>window; --i)
    {
        y1 = 0.0;
        for(int j=i; j<i+window; ++j)
        {
            y1 += intensities.at(j);
        }
        y1 = y1/window;
        
        y2 = 0.0;
        for(int j=i-1; j>i-window; --j)
        {
            y2 += intensities.at(j);
        }
        y2 = y2/window;
        
        m = - (((double)(y2-y1))/window)*100; // should be negative // percentuale, dovrebbe essere fratto window
        
        cout << "pendenza:" << m << endl;
        
        if(m>1000000)
        {
            cout << "Thresholding value: " << i << endl;
            return i;
        }

    }
    
    
    /*
    double y2 = 0.0;
    double y1 = 0.0;
    double m = 0.0;
    int window = 5; // half window size //valutare se aumentare finestra, cambiare quindi anche pendenza?
    
    for(int i=intensities.size()-window; i>window; --i)
    {
        y1=0.0;
        for(int j=i; j<i+window; ++j)
        {
            y1 += intensities.at(j);
        }
        y1 = y1/window;
        
        y2=0.0;
        for(int j=i-1; j>i-window; --j)
        {
            y2 += intensities.at(j);
        }
        y2 = y2/window;
        
        m = (((double)(y2-y1))/window)*100; // should be negative // percentuale, dovrebbe essere fratto window
        //m = (y2-y1)/100; // funziona con m = 90
        //m = ((y2-y1)/window)*100;
        
        cout << "pendenza:" << m << endl;
        
        if(m>90000)
        {
            cout << "Thresholding value: " << i << endl;
            return i;
        }

    }
     */
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
        //cout << "true" << endl;
        return true;
    }
    else
    {
        //cout << "false" << endl;
        return false;
    }
}


void drawRotatedBoundingBoxes(Mat& imgToDraw, const vector<RotatedRect>& rotRectangles)
{
    for(int i=0; i<rotRectangles.size(); ++i)
    {
        Point2f rectPoints[4]; // Dichiarazione del vettore delle quattro coordinate (x,y) del rettangolo da disegnare
        rotRectangles[i].points(rectPoints); // Inizializzo il vettore mediante la funzione points applicata al vettore di RotatedRect
        for(int j=0; j<4; ++j)
        {
            if(isRect(rotRectangles[i]))
                line(imgToDraw, rectPoints[j], rectPoints[(j+1)%4], Scalar(0,0,255), 5); // Disegno il rettangolo collegando mediante dei segmenti i punti memorizzati nel vettore rectPoint, ultimo parametro è lo spessore della linea
            else
                line(imgToDraw, rectPoints[j], rectPoints[(j+1)%4], Scalar(255,0,0), 5);
            
            // cout << "A vertex of rotated rectangle number " << i << " is: " << rectPoints[j] << endl;
            
        }
        // cout << endl;
    }
}



}


