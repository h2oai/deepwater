#include <string>
#include <vector>
#include <map>

#include <image_train.hpp>
using namespace std;

int main() {
    int classes = 10;
    int batch_size = 4;
    vector<string> networks={"lenet","alexnet","googlenet","inception_bn","resnet","vgg"};

    for(string& network : networks) {
        string md5_first;
        {
            ImageTrain imageTrain = ImageTrain();
            imageTrain.buildNet(classes,batch_size,(char*)network.c_str());

            imageTrain.saveModel("/tmp/model");
            imageTrain.saveParam("/tmp/params");

            assert(std::system("md5sum /tmp/model /tmp/params > /tmp/log.txt")==0);
            std::ifstream ifs("/tmp/log.txt");
            md5_first=string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        }

        string md5_second;
        {
            ImageTrain imageTrain = ImageTrain();
            imageTrain.loadModel("/tmp/model");
            imageTrain.setOptimizer(classes,batch_size);
            imageTrain.loadParam("/tmp/params");

            imageTrain.saveModel("/tmp/model");
            imageTrain.saveParam("/tmp/params");

            assert(std::system("md5sum /tmp/model /tmp/params > /tmp/log.txt")==0);
            std::ifstream ifs("/tmp/log.txt");
            md5_second=string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        }

        std::cout << "\n" << network << " : ";
        if (md5_first.compare(md5_second)!=0)
            std::cout << "FAIL" << std::endl; 
        else
            std::cout << "PASS" << std::endl; 
        std::cout << md5_first << "\n" << md5_second << "\n";

    }

    MXNotifyShutdown();
}
