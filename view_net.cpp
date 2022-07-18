// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will show how to use the loss_metric layer to do
    metric learning on images.  

    The main reason you might want to use this kind of algorithm is because you
    would like to use a k-nearest neighbor classifier or similar algorithm, but
    you don't know a good way to calculate the distance between two things.  A
    popular example would be face recognition.  There are a whole lot of papers
    that train some kind of deep metric learning algorithm that embeds face
    images in some vector space where images of the same person are close to each
    other and images of different people are far apart.  Then in that vector
    space it's very easy to do face recognition with some kind of k-nearest
    neighbor classifier.  
    
    In this example we will use a version of the ResNet network from the
    dnn_imagenet_ex.cpp example to learn to map images into some vector space where
    pictures of the same person are close and pictures of different people are far
    apart.  

    You might want to read the simpler introduction to the deep metric learning
    API, dnn_metric_learning_ex.cpp, before reading this example.  You should
    also have read the examples that introduce the dlib DNN API before
    continuing.  These are dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp.

*/

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------


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


// training network type
using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            level0<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
// Next, we define a layer visitor that will modify the weight decay of a network.  The
// main interest of this class is to show how one can define custom visitors that modify
// some network parameters.
class visitor_weight_decay_multiplier
{
public:

    visitor_weight_decay_multiplier(double new_weight_decay_multiplier_) :
        new_weight_decay_multiplier(new_weight_decay_multiplier_) {}

    template <typename layer>
    void operator()(layer& l) const
    {
        set_weight_decay_multiplier(l, new_weight_decay_multiplier);
    }

private:

    double new_weight_decay_multiplier;
};

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Give a folder as input.  It should contain sub-folders of images and we will " << endl;
        cout << "learn to distinguish between these sub-folders with metric learning.  " << endl;
        cout << "For example, you can run this program on the very small examples/johns dataset" << endl;
        cout << "that comes with dlib by running this command:" << endl;
        cout << "   ./dnn_metric_learning_on_images_ex johns" << endl;
        return 1;
    }

    auto model_name = argv[1];


    // net_type net;
    anet_type dlibnet;
    anet_type freshnet;
    deserialize(model_name) >> dlibnet;
    // net.subnet().subnet() = dlibnet.subnet().subnet();

    dnn_trainer<anet_type> trainer(dlibnet, sgd(0.0001, 0.9));
    trainer.set_learning_rate(0.001);
    trainer.be_verbose();
    // trainer.set_synchronization_file("face_metric_sync_trans_nium", std::chrono::minutes(5));
    // I've set this to something really small to make the example terminate
    // sooner.  But when you really want to train a good model you should set
    // this to something like 10000 so training doesn't terminate too early.
    trainer.set_iterations_without_progress_threshold(300);

    // visit_computational_layers(freshnet, visitor_weight_decay_multiplier(0.00));

    // We can also use predefined visitors to affect the learning rate of the whole
    // network.
    // set_all_learning_rate_multipliers(freshnet, 0.0);
     visit_computational_layers(freshnet, visitor_weight_decay_multiplier(0.00));

    // // We can also use predefined visitors to affect the learning rate of the whole
    // // network.
    set_all_learning_rate_multipliers(freshnet, 0.0);
    cout << "the freshnet: " << freshnet;

    freshnet = dlibnet;
    cout << endl << endl << endl;
    cout << "after transfer: " <<endl;
    cout << freshnet;

    cout << "freezing layers" << endl << endl;
    // Usually, we want to freeze the network, except for the top layers:
    visit_computational_layers(freshnet.subnet().subnet(), visitor_weight_decay_multiplier(0));
    set_all_learning_rate_multipliers(freshnet.subnet().subnet(), 0);

    // visit_computational_layers_range<0, 2>(freshnet, visitor_weight_decay_multiplier(0.5));
    // set_learning_rate_multipliers_range<  0,   2>(freshnet, 0.5);

    // visit_computational_layers_range<2, 10>(freshnet, visitor_weight_decay_multiplier(0.1));
    // set_learning_rate_multipliers_range<  2,   10>(freshnet, 0.1);
    cout << freshnet;
        // Now let's print the details of the pnet to the screen and inspect it.
    // cout << "The anet has " << dlibnet.num_layers << " layers in it." << endl;
    // cout << dlibnet << endl;
    // cout << endl << endl << endl << endl;

    // cout << "Changing the original anetlayer 90.\n";
    // layer<90>(dlibnet).layer_details().set_learning_rate_multiplier(0.1);
    // cout << "Changed the original anet layer 90 to LRM of 0.1 instead of 1.\n";

    // cout << "Changing the original anetlayer 9.\n";
    // layer<9>(dlibnet).layer_details().set_learning_rate_multiplier(0.2);
    // cout << "Changed the original anet layer 9 to LRM of 0.2 instead of 1.\n";

    // cout << endl << endl << endl << endl;
    // cout << "The anet now has " << dlibnet.num_layers << " layers in it." << endl;
    // cout << dlibnet << endl;
    // cout << endl << endl << endl << endl;

    // cout << "The fresh net has " << freshnet.num_layers << " layers in it." << endl;
    // cout << freshnet << endl;

    // cout << "Replacing the fresh model's layer's with that of original anet.\n";
    // layer<10>(freshnet) = layer<10>(dlibnet);
    // cout << "Replaced everything but the first 10 layers of freshnet with those of OG anet.\n";

    // cout << endl << endl << endl << endl;

    // cout << "The fresh net now has " << freshnet.num_layers << " layers in it." << endl;
    // cout << freshnet << endl;
    // cout << endl << endl << endl << endl;

    // cout << "Froze everything on freshnet.\n";

    // // We can use the visit_layers function to modify the weight decay of the entire
    // // network:
    // visit_computational_layers(freshnet, visitor_weight_decay_multiplier(0.00));

    // // We can also use predefined visitors to affect the learning rate of the whole
    // // network.
    // set_all_learning_rate_multipliers(freshnet, 10.0);
    // cout << "The fresh net now has " << freshnet.num_layers << " layers in it." << endl;
    // cout << freshnet << endl;

    // cout << endl << endl << endl << endl;

    // cout << "Replacing the fresh model's layer's with that of original anet.\n";
    // layer<10>(freshnet) = layer<10>(dlibnet);
    // cout << "Replaced everything but the first 10 layers of freshnet with those of OG anet.\n";

    // cout << endl;

    // cout << "The fresh net now has " << freshnet.num_layers << " layers in it." << endl;
    // cout << freshnet << endl;

    // set_all_learning_rate_multipliers(freshnet.subnet().subnet(), 0.7);    
    // cout << "freshnet.subnet.subnet LRM: 0.7\n";

    // cout << endl;

    // cout << "The fresh net now has " << freshnet.num_layers << " layers in it." << endl;
    // cout << freshnet << endl;
}


