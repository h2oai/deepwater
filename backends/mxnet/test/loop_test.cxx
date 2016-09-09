#include <string>
#include <vector>
#include <map>

#include <image_train.hpp>
using namespace std;

int main() {
    int classes = 10;
    int batch_size = 32;
    std::vector<float> data(28*28*1*batch_size);
    std::vector<float> label(batch_size);

    vector<string> networks={"lenet"};
    for(string& network : networks) {
	    int count=0;
	    ImageTrain imageTrain = ImageTrain();
	    imageTrain.buildNet(classes,batch_size,(char*)network.c_str());
	    while(1) {
		    //imageTrain.saveModel("/tmp/model");
		    //imageTrain.saveParam("/tmp/params");
		    imageTrain.train(&data[0], &label[0]);
		    std::cout << "\n" << count++ << " : ";
	    }

    }

    MXNotifyShutdown();
}
