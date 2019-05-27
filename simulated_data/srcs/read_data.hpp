#ifndef __READ_DATA_HPP
#define __READ_DATA_HPP
#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"

std::vector<std::string> GetDataSequence(std::string filename);
void StringReplace(std::string &str, std::string find, std::string repl);
void PaserAnninfo(std::string str, std::vector<std::vector<cv::Point2f>> &points, std::vector<int> &class_index);
cv::Mat PaserDepth(std::string str, int rows, int cols);
void ShowLabel(std::vector<std::vector<cv::Point2f>> vpoints, std::vector<int> vclass_index);
void PushData(std::string fname, int class_id, std::vector<cv::Point2f> shape);
#endif
