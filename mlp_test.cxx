
#include "mlp.hpp"

#include "test_data.hpp"

int main() {

  MLPNative m = MLPNative();
  std::cout << aptr_x[123] << std::endl;
  std::cout << aptr_y[12] << std::endl;
  int lsize[1] = {10};
  m.setLayers(lsize, 1, 2);
  char * act[1] = {"tanh"};
  m.setAct(act);
  m.build_mlp();
  m.setData(aptr_x, 101, 60);
  m.setLabel(aptr_y, 101);
}
