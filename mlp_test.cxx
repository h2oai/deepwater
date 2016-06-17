
#include "mlp.hpp"

int main() {

  MLPClass m = MLPClass();
  mx_float* aptr_x = new mx_float[128 * 28];
  mx_float* aptr_y = new mx_float[128];

  // we make the data by hand, in 10 classes, with some pattern
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 28; j++) {
      aptr_x[i * 28 + j] = i % 10 * 1.0f;
    }
    aptr_y[i] = i % 10;
  }
  int lsize[2] = {512, 10};
  m.setLayers(lsize, 2);

  int dimX[2] = {128, 28};
  m.setX(aptr_x, dimX, 2);
  m.setLabel(aptr_y, 128);
  m.buildnn();
  std::cout << m.train(1000, true) << std::endl;
}
