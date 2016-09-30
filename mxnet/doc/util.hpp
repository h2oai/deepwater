/*!
 *  Copyright (c) 2016 by Contributors
 */
#ifndef DEEPWATER_MXNET_UTIL_HPP
#define DEEPWATER_MXNET_UTIL_HPP

#include <iostream>
#include <vector>
#include <string>

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mxnet/operator.h"
#include "mxnet/symbolic.h"
#include "mxnet/optimizer.h"

template<typename FunRegType>
void FunctionRegInfo(const FunRegType *e) {
  std::cout << "* Name: `" << e->name << "`" << std::endl;
  std::cout << "* Description: " << e->description << std::endl;
  if (e->arguments.size() > 0) {
    std::cout << "* Arguments: " << std::endl << std::endl;
    std::cout << "| Name | Type info | Description |" << std::endl
        << "| --- | --- | --- |" << std::endl;
    for (size_t i = 0; i < e->arguments.size(); i++) {
      std::cout << "|"
          << e->arguments[i].name << "|"
          << e->arguments[i].type_info_str << "|"
          << e->arguments[i].description << "|" << std::endl;
    }
  }
  std::cout << std::endl;
}

#endif
