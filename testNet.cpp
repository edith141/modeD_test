#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// We will need to create some functions for loading data.  This program will
// expect to be given a directory structured as follows:
//    top_level_directory/
//        person1/
//            image1.jpg
//            image2.jpg
//            image3.jpg
//        person2/
//            image4.jpg
//            image5.jpg
//            image6.jpg
//        person3/
//            image7.jpg
//            image8.jpg
//            image9.jpg

std::vector<std::vector<string>> load_objects_list (
    const string& dir 
)
{
    std::vector<std::vector<string>> objects;
    for (auto subdir : directory(dir).get_dirs())
    {
        std::vector<string> imgs;
        for (auto img : subdir.get_files())
            imgs.push_back(img);

        if (imgs.size() != 0)
            objects.push_back(imgs);
    }
    return objects;
}

// This function takes the output of load_objects_list() as input and randomly
// selects images for training.  It should also be pointed out that it's really
// important that each mini-batch contain multiple images of each person.  This
// is because the metric learning algorithm needs to consider pairs of images
// that should be close (i.e. images of the same person) as well as pairs of
// images that should be far apart (i.e. images of different people) during each
// training step.
void load_mini_batch (
    const size_t num_people,     // how many different people to include
    const size_t samples_per_id, // how many images per person to select.
    dlib::rand& rnd,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels
)
{
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_people <= objs.size(), "The dataset doesn't have that many people in it.");

    std::vector<bool> already_selected(objs.size(), false);
    matrix<rgb_pixel> image; 
    for (size_t i = 0; i < num_people; ++i)
    {
        size_t id = rnd.get_random_32bit_number()%objs.size();
        // don't pick a person we already added to the mini-batch
        while(already_selected[id])
            id = rnd.get_random_32bit_number()%objs.size();
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j)
        {
            const auto& obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            load_image(image, obj);
            images.push_back(std::move(image));
            labels.push_back(id);
        }
    }


    // All the images going into a mini-batch have to be the same size.  And really, all
    // the images in your entire training dataset should be the same size for what we are
    // doing to make the most sense.  
    DLIB_CASSERT(images.size() > 0);
    for (auto&& img : images)
    {
        DLIB_CASSERT(img.nr() == images[0].nr() && img.nc() == images[0].nc(), 
            "All the images in a single mini-batch must be the same size.");
    }
}

// ----------------------------------------------------------------------------------------

// The next page of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and make the network somewhat smaller.

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level0 = res_down<256,SUBNET>;
template <typename SUBNET> using level1 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = res<64,res<64,res<64,res_down<64,SUBNET>>>>;
template <typename SUBNET> using level4 = res<32,res<32,res<32,SUBNET>>>;

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

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cout << "Give a folder as input.  It should contain sub-folders of images and we will " << endl;
        cout << "test the images for similarity/recognition.  " << endl;
        cout << "For example, you can run this program on the very small examples/johns dataset" << endl;
        cout << "that comes with dlib by running this command:" << endl;
        cout << "Give the name of the model file to run the test with as the second argument." << endl;
        cout << "Run the program with a command like this:" << endl;

        cout << "   ./testModel.out folder_with_images model_name.dat threshold_number" << endl;
        return 1;
    }

    auto objs = load_objects_list(argv[1]);
    auto c_threshold = atof(argv[3]);

    cout << "objs.size(): "<< objs.size() << endl;

    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    auto model_name = argv[2];
    // net_type net;
    anet_type testing_net;
    deserialize(model_name) >> testing_net;
 
    // Now, just to show an example of how you would use the network, let's check how well
    // it performs on the training data.
    dlib::rand rnd(time(0));
    load_mini_batch(10, 5, rnd, objs, images, labels);

    // Normally you would use the non-batch-normalized version of the network to do
    // testing, which is what we do here.

    // Run all the images through the network to get their vector embeddings.
    std::vector<matrix<float,0,1>> embedded = testing_net(images);

    // Now, check if the embedding puts images with the same labels near each other and
    // images with different labels far apart.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i+1; j < embedded.size(); ++j)
        {
            if (labels[i] == labels[j])
            {
                // The loss_metric layer will cause images with the same label to be less
                // than net.loss_details().get_distance_threshold() distance from each
                // other.  So we can use that distance value as our testing threshold.
                if (length(embedded[i]-embedded[j]) < c_threshold)
                    ++num_right;
                else
                    ++num_wrong;
            }
            else
            {
                if (length(embedded[i]-embedded[j]) >= c_threshold)
                    ++num_right;
                else
                    ++num_wrong;
            }
        }
    }

    cout << testing_net;
    cout << "model loss details: " << testing_net.loss_details() << endl;
    cout << "num_right: "<< num_right << endl;
    cout << "num_wrong: "<< num_wrong << endl;
    float acc = (num_right / 1.0) / ((num_right + num_wrong) / 1.0);
    cout << "accuracy: " <<  acc * 100 << endl;

}


