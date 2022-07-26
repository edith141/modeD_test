#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <fstream>
#include <string>

using namespace dlib;
using matr = std::vector<std::vector<float>>;


std::string getFileName(std::string path){
    std::string ans = "";
    bool seen_ = false;
    for(int i = (int)path.size() -1; i>=0; i--){
        if(path[i] == '/'){
            break;
        }
        ans+=path[i];
    }

    std::string fans = "";
    bool seen_dot = false;
    for(int i = (int)ans.size() -1; i>=0; i--){
        if(ans[i]=='.'){
            seen_dot = true;
        }
        else if(!seen_dot){
            fans+=ans[i];
        }
    }

    return fans;
}

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img){
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

std::ostream &operator << ( std::ostream &out, const matr &M )
{
   for ( auto &row : M )
   {
      for ( auto e : row ) out << e << '\t';
      out << '\n';
   }
   return out;
}

// Template based network structure is defined below
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;


template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;



int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        std::cout << "Provide a test data directory path, and the modelname to this program as cmd line input " << std::endl;
        return 1;
    }

    //Intializing face detector
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;


    // Initializing the network with previously trained model
    anet_type net;
    auto model_name = argv[2];
    deserialize(model_name) >> net;

    // Registering each person described as subdirectory
    std::vector<matrix<rgb_pixel>> faces;
    for (auto subdir : directory(argv[1]).get_dirs())
    {
        std::cout<<"\nRegistering "<<getFileName(subdir)<<" ... "<<std::endl;
        
        // Getting the required fingerprint images
        std::vector<matrix<rgb_pixel>> fing_print_images;
        for (auto img_path : subdir.get_files()){
            dlib::matrix<rgb_pixel> img;
            dlib::load_image(img, img_path);

	    //Detecting faces in loaded image
	    for (auto face : detector(img)){
		auto shape = sp(img, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
		faces.push_back(std::move(face_chip));
	    }

	    if (faces.size() > 1){
		std::cout << "More than 1 face found. " << std::endl;
            }

            fing_print_images.push_back(faces[0]);
	    faces.clear();
        }


        // Getting the descriptors for every image from a trained network
        std::vector<matrix<float,0,1>> descriptors = net(fing_print_images);
        
        
        // creating a 2D vector from descriptors of all image
        matr mat1;
        for(int i = 0; i< fing_print_images.size(); i++){
            std::vector<float> vec;
            for(int j = 0; j< 128 ; j++){
                vec.push_back(descriptors[i](0,j));
            }
            mat1.push_back(vec);
        }

        // writing matrix mat in a file for future use
        std::string dscrptr_path = cast_to_string(subdir)+".txt";
        std::ofstream outFile(dscrptr_path);
        outFile << mat1;
        outFile.close();

        std::cout<<"Done."<<std::endl;
    }
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
