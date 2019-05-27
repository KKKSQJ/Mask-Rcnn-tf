#include "3DRotation.hpp"

/*
* map bounding box by euler angle and compute Perspective Transform matrix
 */
std::vector<std::vector<cv::Mat>> RotateImage::MapBboxByEuler(std::vector<std::vector<cv::Point2f>> vpts, std::vector<std::vector<std::vector<cv::Point2f>>> &vdpts, float range, float step)
{
	cv::RNG rng(time(NULL));
	std::vector<std::vector<cv::Mat>> result(vpts.size());
	std::vector<std::vector<std::vector<cv::Point2f>>> result_vvdpts(vpts.size());
	for (int i = 0; i < vpts.size(); ++i)
	{
		std::vector<cv::Mat> v;
		std::vector<std::vector<cv::Point2f>> result_vdpts;
		for(int rx = -range; rx<=range; rx+=step)
			for(int ry = -range; ry<=range; ry+=step)
				{
					int rz = rng.uniform(-90,90);
					int d=rng.uniform(1000,2000);
					std::vector<cv::Point2f> target_pts=MapPointByEuler(vpts[i], rx, ry, rz, d);

					cv::Mat per_mat = cv::getPerspectiveTransform(vpts[i], target_pts);
					v.push_back(per_mat);
					result_vdpts.push_back(target_pts);
				}

		result[i] = v;
		result_vvdpts[i]=result_vdpts;
	}

	vdpts = result_vvdpts;
	return result;
}


std::vector<cv::Point2f> RotateImage::MapPointByEuler(std::vector<cv::Point2f> pts, float rx, float ry, float rz, int depth)
{
	using namespace Eigen;
	float c_x=0.f;
	float c_y=0.f;
	for (int i = 0; i < pts.size(); ++i)
	{
		c_x += pts[i].x;
		c_y += pts[i].y;
	}
	c_x = c_x/pts.size();
	c_y = c_y/pts.size();
	Vector3f v_center(c_x,c_y,0);
	std::vector<Vector3f> vpts;
	for (int i = 0; i < pts.size(); ++i)
		vpts.push_back(Vector3f(pts[i].x, pts[i].y, 0));
	std::vector<Vector3f> dvpts;
	Matrix3f mat = euler_to_mat(rx,ry,rz);
	for(int i=0; i<vpts.size();i++)
		dvpts.push_back(mat*(vpts[i]-v_center));

	//translation center and map
	std::vector<cv::Point2f> v;
	float distance = visual_to_distance(depth);
	for(int i=0; i<dvpts.size();i++)
	{
		float x = dvpts[i][0]*distance/(distance-dvpts[i][2]) + v_center[0];
		float y = dvpts[i][1]*distance/(distance-dvpts[i][2]) + v_center[1];
		v.push_back(cv::Point2f(x,y));
	}
	return v;
}