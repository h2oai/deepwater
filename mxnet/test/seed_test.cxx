#include <string>
#include <vector>
#include <map>

#include <image_train.hpp>
using namespace std;

string getMD5(int classes, int batch_size, string network) {
    ImageTrain imageTrain = ImageTrain(224,224,3);
    imageTrain.setSeed(1234);
    imageTrain.buildNet(classes,batch_size,(char*)network.c_str());

    imageTrain.saveModel("/tmp/model");
    imageTrain.saveParam("/tmp/params");

    assert(std::system("md5sum /tmp/model /tmp/params > /tmp/log.txt")==0);
    std::ifstream ifs("/tmp/log.txt");
    return string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

int main() {
    int classes = 10;
    int batch_size = 4;
    vector<string> networks={"lenet","alexnet","googlenet","inception_bn","resnet","vgg"};
    for(string& network : networks) {
        string md5_first = getMD5(classes,batch_size,network);
        string md5_second = getMD5(classes,batch_size,network);

        std::cout << "\n" << network << " : ";
        if (md5_first.compare(md5_second)!=0)
            std::cout << "FAIL" << std::endl; 
        else
            std::cout << "PASS" << std::endl; 
        std::cout << md5_first << "\n" << md5_second << "\n";
    }
    MXNotifyShutdown();
}
