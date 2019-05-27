#include "utils.hpp"


/*
* bbox points tansfrom to bounding box
*
* 
 */

std::vector<std::vector<int>> Shape2BoundingBox(std::vector<std::vector<cv::Point2f>> points)
{
	std::vector<std::vector<int>> v;
	for(int i=0; i<points.size(); i++)
	{
		std::vector<int> bbox;	
		int xmax,ymax;
	 	int xmin=xmax=points[i][0].x;
	 	int ymin=ymax=points[i][0].y;
	 	for (int j=0; j<points[i].size(); j++)
	 	{
	 		if (points[i][j].x>xmax)xmax = points[i][j].x;
	 		if (points[i][j].x<xmin)xmin = points[i][j].x;

	 		if (points[i][j].y>ymax)ymax = points[i][j].y;
	 		if (points[i][j].y<ymin)ymin = points[i][j].y;
	 	}
	 	bbox.push_back(xmin);
	 	bbox.push_back(ymin);
	 	bbox.push_back(xmax);
	 	bbox.push_back(ymax);
	 	v.push_back(bbox);
	}

	return v;
}

/*
* show boundingbox
* 
 */
void ImgShowBbox(cv::Mat img, std::vector<std::vector<int>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		cv::line(img, cv::Point(bbox[i][0],bbox[i][1]), cv::Point(bbox[i][2],bbox[i][1]), cv::Scalar(255,100,100),2);
		cv::line(img, cv::Point(bbox[i][2],bbox[i][1]), cv::Point(bbox[i][2],bbox[i][3]), cv::Scalar(255,100,100),2);
		cv::line(img, cv::Point(bbox[i][2],bbox[i][3]), cv::Point(bbox[i][0],bbox[i][3]), cv::Scalar(255,100,100),2);
		cv::line(img, cv::Point(bbox[i][0],bbox[i][3]), cv::Point(bbox[i][0],bbox[i][1]), cv::Scalar(255,100,100),2);
	}
	cv::imshow("display", img);
	cv::waitKey(-1);
}

cv::Mat GetRoi(cv::Mat img, std::vector<std::vector<cv::Point2f>> shape)
{
	std::vector<std::vector<cv::Point>> int_shape(shape.size());
	for (int i = 0; i < shape.size(); ++i)
	{
		for (int j = 0; j < shape[i].size(); ++j)
		{
			int_shape[i].push_back(cv::Point((int)shape[i][j].x,(int)shape[i][j].y));
		}
	}
	cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	cv::fillPoly(mask, int_shape, cv::Scalar(255));
	cv::Mat src;
	img.copyTo(src, mask);
	return src;
}