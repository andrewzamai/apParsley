#ifndef parsleyLib_hpp
#define parsleyLib_hpp

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


namespace parsleyLib{

/**
 Computes the passed image histogram and saves the ready to display data in the other image passed.
 
 @param image the image of which to compute the histogram
 @param imageToDisplayWhere the image in which to save the ready to display hist data
 */
void calculateCV_8UHistogram(const cv::Mat& image, cv::Mat& imageToDisplayWhere);


/**
 Applies  gamma correction to the passed image.
 
 @param image the image to process
 @param gamma gamma value of the correction
 */
void applyGammaCorrection(cv::Mat& image, double gamma);


/**
 Uses passed thresholding value to get a binary image of the image.
 
 @param image the image to process
 @param thresholdingValue the threshold value
 */
void toBinaryImage(cv::Mat& image, double thresholdingValue);


/**
 Use this function to create an instance of SimpleBlobDetector::Params type, with alredy all parameters setted for this specific application.
 */
cv::SimpleBlobDetector::Params instantiateBlobParams();


/**
 Use this function to get an instance of SimpleBlobDetector or each time you need to change the minArea filtering parameter.
 
 Having two distinct function: instantiateBlobParams() and getBlobDetectorInstance() does not want to be a Singleton Pattern, instead:
 permits to instantiate a Params object with alredy all parameters setted for this specific application (more encapsulation and information hiding),
 permits to instantiate only once a Params object and easily modify its minArea filtering parameter, instead of each time create a completely new Param object.
 
 @param parameters a pointer to an alredy instantiated SimpleBlobDetector::Params object
 @param minArea the new minArea filtering parameter
 @return a smart pointer (the pointed object will automatically cleaned up after the Ptr is destroyed)
 */
cv::Ptr<cv::SimpleBlobDetector> getBlobDetectorInstance(cv::SimpleBlobDetector::Params* parameters, float minArea);


/**
 This function computes BlobDetection and combines Blob Detection outputs with Bounding rotated boxes technique to give a cleaner image of the detected impurities.
 
 @param blobDetector an alredy instantiated and setted blobDetector
 @param imgWithKeypoints the image where to draw the blobdetected keypoints
 @param imgWithBoundingBoxes the image where to draw the rotated bounding boxes
 @return a vector of keypoint from which coordinates can be extracted
 */
std::vector<cv::KeyPoint> boundingBlobDetect(cv::Ptr<cv::SimpleBlobDetector> blobDetector, cv::Mat& binaryImage, cv::Mat& imgWithKeypoints, cv::Mat& imgWithBoundingBoxes);






/*------------------------------------------- Helper functions below: --------------------------------------------------*/







double getAdaptiveThreshValue(const cv::Mat& image);


/**
 Checks if a given point belongs to a given rotated rectangle.
 
 (Used in eliminateNonBlobDetectedRect funcion).
 @param rect a rotated rectangle
 @param point a point
 @return true if the point is contained by the rectangle, false otherwise
 */
bool pointBelongsToRect(const cv::RotatedRect& rect, cv::Point2f point);

/**
 Eliminates from the vector of RotatedRect elements all the ones that where also detected by the BlobDetector.
 
 This function is used in order to obtain a vector of all the "unwanted rotated rectangles" (for example too small for the given minArea filtering parameter).
 The new vector will be used as second operator in a subtraction with the original vector to get a vector of all rotRects that where also detected by the blob detector based on their area.
 
 @param keypoints a vector of keypoints obtained from a Blob detection
 @param rotRects a vector of rotRects obtained from finfContours function
 */
void eliminateBlobDetectedRect(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::RotatedRect>& rotRects);

/**
 Computes vector difference: the new first vector parameter will be a vector with all the elements that were not contained in the second vector parameter.
 
 @param vectorCopy first operand
 @param vector second operand
 */
void vectorsDifference(std::vector<cv::RotatedRect>& vectorCopy, const std::vector<cv::RotatedRect>& vector);


/**
 Checks if a given rectangle is more similar to a square or to a rectangle.
 
 @param r the rotated rectangle to check
 @return true if r is rectangle like, false if is more square like
 */
bool isRect(cv::RotatedRect r);

/**
 Draws the rotated bounding boxes on the image.
 
 @param imgToDraw a white image where to draw
 @param rotRectangles the vector of all the rectangles to draw
 */
void drawRotatedBoundingBoxes(cv::Mat& imgToDraw, const std::vector<cv::RotatedRect>& rotRectangles);



}
#endif /* parsleyLib_hpp */
