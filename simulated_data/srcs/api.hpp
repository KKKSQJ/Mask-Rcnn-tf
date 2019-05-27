#ifndef __API___HPP
#define __API___HPP 
#include <iostream>
#include <vector>
#include <string>

#include <boost/bind.hpp>
#include "../boost_tool/threadpool.hpp"

#include "3DRotation.hpp"
#include "utils.hpp"
#include "read_data.hpp"

void api_one_shape(std::string filename);
void api_one_shape_thread(std::string filename);
#endif