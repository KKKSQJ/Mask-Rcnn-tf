#ifndef __UTILS_HPP
#define __UTILS_HPP
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<std::vector<int>> Shape2BoundingBox(std::vector<std::vector<cv::Point2f>> points);
void ImgShowBbox(cv::Mat img, std::vector<std::vector<int>> bbox);
cv::Mat GetRoi(cv::Mat img, std::vector<std::vector<cv::Point2f>> shape);

#endif