#include "api.hpp"

void api_one_shape(std::string filename)
{
	std::vector<std::string> imgs_name_list = GetDataSequence(filename);
	for(int img_id=0; img_id<imgs_name_list.size(); img_id++)
    {
        std::cout<<imgs_name_list[img_id]<<std::endl;
    	// demp deal an image
    	std::string file_name = imgs_name_list[img_id];
    	//get image
    	cv::Mat img = cv::imread(file_name);
        //get ann
        StringReplace(file_name, "image", "labels");
        StringReplace(file_name, ".jpg", ".txt");
        std::vector<std::vector<cv::Point2f>> vpoints;
        std::vector<int> vclass_index;
        PaserAnninfo(file_name, vpoints, vclass_index);
        //only have one shape
        cv::Mat roi = GetRoi(img, vpoints);
        //ShowLabel(vpoints, vclass_index);
        std::vector<std::vector<int>> bbox;
        bbox = Shape2BoundingBox(vpoints);
        //ImgShowBbox(img, bbox);
        RotateImage ri;
        std::vector<std::vector<std::vector<cv::Point2f>>> v_v_bbox;
        std::vector<std::vector<cv::Mat>> v_per_mat = ri.MapBboxByEuler(vpoints, v_v_bbox);

        for (int k = 0; k < v_per_mat.size(); ++k)
        {
            std::vector<std::vector<int>> bbox = Shape2BoundingBox(v_v_bbox[k]);
            for (int j = 0; j < v_per_mat[k].size(); ++j)
            {
                cv::Mat img_dst;
                roi.copyTo(img_dst);
                cv::warpPerspective(img_dst, img_dst, v_per_mat[k][j], roi.size());
                //ImgShowBbox(img_dst, std::vector<std::vector<int>>({bbox[j]}));
                std::string  dst_name = imgs_name_list[img_id];
                StringReplace(dst_name, "/data/", "/simulated/");    
                StringReplace(dst_name, ".jpg", "_"+std::to_string(j)+"_.jpg");
                cv::imwrite(dst_name, img_dst);

                StringReplace(dst_name, "image", "labels");
                StringReplace(dst_name, ".jpg", ".txt");
                PushData(dst_name, vclass_index[k], v_v_bbox[k][j]);
            }
        }
    }
}


void multiple_thread(std::vector<cv::Mat> mat,
 std::vector<std::vector<cv::Point2f>> v_box,
 cv::Mat roi,
 std::string name,
 int index,
 int id_s, int id_e)
{
    int total = mat.size();
    for(int i=id_s; i<std::min(id_e, total); i++)
    {
        cv::Mat img_dst;
        roi.copyTo(img_dst);
        cv::warpPerspective(img_dst, img_dst, mat[i], img_dst.size());
        std::string dst_name(name);
        StringReplace(dst_name, "/data/", "/simulated/");    
        StringReplace(dst_name, ".jpg", "_"+std::to_string(i)+"_.jpg");
        cv::imwrite(dst_name, img_dst);
        StringReplace(dst_name, "image", "labels");
        StringReplace(dst_name, ".jpg", ".txt");
        PushData(dst_name, index, v_box[i]);
        img_dst.release();
    }
}
void api_one_shape_thread(std::string filename)
{
    std::vector<std::string> imgs_name_list = GetDataSequence(filename);
   
    for(int img_id=0; img_id<imgs_name_list.size(); img_id++)
    {
        std::cout<<imgs_name_list[img_id]<<std::endl;
        // demp deal an image
        std::string file_name = imgs_name_list[img_id];
        //get image
        cv::Mat img = cv::imread(file_name);
        //get ann
        StringReplace(file_name, "image", "labels");
        StringReplace(file_name, ".jpg", ".txt");
        std::vector<std::vector<cv::Point2f>> vpoints;
        std::vector<int> vclass_index;
        PaserAnninfo(file_name, vpoints, vclass_index);
        //only have one shape
        cv::Mat roi = GetRoi(img, vpoints);
        //ShowLabel(vpoints, vclass_index);
        //std::vector<std::vector<int>> bbox;
        //bbox = Shape2BoundingBox(vpoints);
        // ImgShowBbox(img, bbox);
        RotateImage ri;
        std::vector<std::vector<std::vector<cv::Point2f>>> v_v_bbox;
        std::vector<std::vector<cv::Mat>> v_per_mat = ri.MapBboxByEuler(vpoints, v_v_bbox);

        for (int i = 0; i < v_per_mat.size(); ++i)
        {
            std::cout<<v_per_mat[i].size()<<std::endl;
            std::vector<std::vector<int>> bbox = Shape2BoundingBox(v_v_bbox[i]);
           
            int thread_num = 10;
            //vector divided into 10 parts. The first 9 copies of same， 10 is size-9*step
            static boost::threadpool::pool pl(thread_num);
            int thread_step = v_per_mat[i].size()/(thread_num-1);
            //10 thread
            for (int k = 0; k < thread_num; ++k)
            {
                pl.schedule(boost::bind(multiple_thread, 
                    v_per_mat[i], v_v_bbox[i], roi, imgs_name_list[img_id], vclass_index[i], k*thread_step, (k+1)*thread_step));
            }
            pl.wait();
            std::vector<std::vector<int>> ().swap(bbox);
        }
        std::vector<std::vector<cv::Mat>> ().swap(v_per_mat);
        std::vector<std::vector<std::vector<cv::Point2f>>> ().swap(v_v_bbox);
    }
    
}