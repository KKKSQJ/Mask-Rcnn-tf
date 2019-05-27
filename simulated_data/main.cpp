#include "srcs/api.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
	if (argc<2)
	{
		std::cout << "Usage: x.exe [fold + file] \n";
		return -1;
	}
	std::string fname(argv[1]);
	api_one_shape(fname);
	//api_one_shape_thread(fname);
	return 0;
}
