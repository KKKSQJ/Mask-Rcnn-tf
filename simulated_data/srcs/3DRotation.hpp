#ifndef __3DROTATION__HPP
#define __3DROTATION__HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#define PI acos(-1)
#define RAD PI/180


class RotateImage
{
private:
	float x;
	float y;
	float z;
public:
	RotateImage(float rx=0, float ry=0, float rz=0) :x(rx),y(ry),z(rz){}
	~RotateImage(){}
    Eigen::Matrix3f euler_to_mat(float rx,float ry, float rz)
    {
    	using namespace Eigen;
    	Quaternionf qut = AngleAxisf(rx*RAD, Vector3f::UnitX())*
    					  AngleAxisf(ry*RAD, Vector3f::UnitY())*
    					  AngleAxisf(rz*RAD, Vector3f::UnitZ());
    	return qut.toRotationMatrix();
    }
    void set_angle(float rx=0, float ry=0, float rz=0)
    {
    	x = rx; y = ry; z = rz;
    }
    float visual_to_distance(int depth)
    {
        // return std::sqrt(std::pow(h, 2)+std::pow(w,2))/2/std::tan(angle/2*RAD);
        return depth;
    }
    std::vector<cv::Point2f> MapPointByEuler(std::vector<cv::Point2f> pts, float rx, float ry, float rz, int depth);
    std::vector<std::vector<cv::Mat>> MapBboxByEuler(std::vector<std::vector<cv::Point2f>> vpts,std::vector<std::vector<std::vector<cv::Point2f>>> &vdpts, float range=30, float step=22.5);

    
};

#endif