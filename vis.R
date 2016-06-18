tmp <- paste(readLines("~/Desktop/h2o-native/mlp.json"), collapse="")
library(mxnet)
graph.viz(tmp)
