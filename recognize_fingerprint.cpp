#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <fstream>
#include <string>
#include <cfloat>

using namespace dlib;
using matr = std::vector<std::vector<float>>;

float getDistance(std::vector<float> a, std::vector<float> b){
    if(a.size()!=b.size()){
        std::cout<<"Both vector's dimension should be same."<<std::endl;
        return 0;
    }
    
    float sum = 0;
    for(int i = 0; i < a.size(); i++){
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

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

std::istream &operator >> ( std::istream &in, matr &M )
{
   M.clear();                                          
   for ( std::string line; std::getline( in, line ); )
   {
      std::stringstream ss( line );
      std::vector<float> row;
      for ( float e; ss >> e; row.push_back( e ) );
      M.push_back( row );
   }
   return in;
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



float mean(std::vector<float> data){
  int size = data.size();
  float sum = 0.0;

  for(int i = 0; i < size; ++i) {
    sum += data[i];
  }

  return sum/size;
}

float calculateSD(std::vector<float> data) {
  int size = data.size();
  float sum = 0.0, mean, standardDeviation = 0.0;
  int i;

  for(i = 0; i < size; ++i) {
    sum += data[i];
  }

  mean = sum / size;

  for(i = 0; i < size; ++i) {
    standardDeviation += pow(data[i] - mean, 2);
  }

  return sqrt(standardDeviation / size);
}

int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        std::cout << "Provide a sample test data directory path and the descriptor's to this program as cmd line input " << std::endl;
        return 1;
    }


    // Initializing the network with previously trained model
    anet_type net;
    deserialize("metric_network_renset_Dtrans_nium_frozen_pfast2500.dat") >> net;

    for (auto img_path : directory(argv[1]).get_files()){
        std::cout<<"\nRecognizing image at path => "<<img_path<<" "<<std::endl;
        std::vector<matrix<rgb_pixel>> fing_print_images;
        dlib::matrix<rgb_pixel> img;
        dlib::load_image(img, img_path);
        fing_print_images.push_back(img);

        matrix<float,0,1> descriptor = net(fing_print_images)[0];

        float distance = FLT_MAX;
        std::string person_name = "";
        for (auto descr_path : directory(argv[2]).get_files()){
            // reading matrix from the file
            matr mat;
            std::ifstream inFile(descr_path);
            inFile >> mat;
            inFile.close();
            
            for(int i = 0; i< mat.size(); i++){
		matrix<float,0,1> saved_descriptor(1,128);
		for(int j = 0; j < 128; j++){
		    saved_descriptor(0,j) = mat[i][j];
		}

                float new_dist = length(descriptor-saved_descriptor);
                if(distance > new_dist){
                     distance = new_dist;
                     person_name = getFileName(descr_path);
                }
            }
        }
        std::cout << "Distance: "<<distance<<", Person Name: "<<(distance < 0.6? person_name: "Unknown")<<std::endl;
    }
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
