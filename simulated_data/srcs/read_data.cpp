#include "read_data.hpp"

/*
* GetDataSequence
* get org data name sequence
* filename:org data name file
* return std::vector<std::string>
 */
std::vector<std::string> GetDataSequence(std::string filename)
{
	std::ifstream sequence_file(filename, std::ios::in);
	if (!sequence_file) 
	{
		std::cout<<"check the "<<filename<<std::endl;
		exit(-1);
	}
	std::vector<std::string> sequence;

	std::string buff;
	while(getline(sequence_file, buff))
		sequence.push_back(buff);

	return sequence;
}
/*
* string replace 'find' to be replaced 'repl' in str
* str : target str
* find : string to be replaced
* repl : will replaced
 */
void StringReplace(std::string &str, std::string find, std::string repl)
{
	std::size_t found = str.find(find);
	str.replace(found,find.length(), repl);
}

/*
* paser ann info 
* str : target file name
* points : the target file contain object shape
* class_index : the target file contain object class 
 */
void PaserAnninfo(std::string str, std::vector<std::vector<cv::Point2f>> &points, std::vector<int> &class_index)
{
	std::ifstream ann_file(str, std::ios::in);
	if (!ann_file)
	{
		std::cout<<"chect the "<<str<<std::endl;
		exit(-1);
	}

	std::string buff;

	while(getline(ann_file, buff))
	{
		std::vector<int> v;
		std::stringstream ann_str(buff);
		for (std::string s; ann_str>>s;)
		{
			std::istringstream iss(s);
			int num;
			iss>>num;
			v.push_back(num);
		}
		class_index.push_back(v[0]);
		v.erase(v.begin());
		std::vector<cv::Point2f> shape_points;
		for (int i=0; i<v.size(); i+=2)
			shape_points.push_back(cv::Point2f((float)v[i], (float)v[i+1]));

		points.push_back(shape_points);
	}
}
/*
* paser depth data from bin to cv::Mat
* str : target file name
* rows : depth rows
* cols : depth cols
* return mat type depth image.
 */
cv::Mat PaserDepth(std::string str, int rows, int cols)
{
	int depth_size = rows*cols;
	float *data = new float[depth_size];
	std::ifstream depth_file(str, std::ios::in|std::ios::binary);
	depth_file.read((char *)data, sizeof(float)*depth_size);
	cv::Mat depth(rows, cols, CV_32F, data);
	return depth;
}
/*
*show label infomation
* 
 */
void ShowLabel(std::vector<std::vector<cv::Point2f>> vpoints, std::vector<int> vclass_index)
{
	for (int i = 0; i < vpoints.size(); ++i)
    {
    	std::cout<<"class index: "<<vclass_index[i]<<"  ";
    	for (int j = 0; j < vpoints[i].size(); ++j)
    	{
    		std::cout<<vpoints[i][j]<<"  ";
    	}
    	std::cout<<std::endl;
    }
}

void PushData(std::string fname, int class_id, std::vector<cv::Point2f> shape)
{
	std::ofstream out(fname);
	if(out.is_open())
	{
		out<<std::to_string(class_id)+" ";
		for (int i=0;i<shape.size(); i++)
		{
			out<<(int)shape[i].x<<" "<<(int)shape[i].y<<" ";
		}
		out<<"\n";
		out.close();
	}
}