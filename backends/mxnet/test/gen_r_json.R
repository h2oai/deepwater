library(mxnet)

#name = "alexnet"
#name = "googlenet"
#name = "inception-bn"
#name = "lenet"
#name = "mlp"
name = "unet"
#name = "resnet"
#name = "vgg"

source(paste("symbol_", name, ".R", sep = ''))

network <- get_symbol(10)

cat(network$as.json(), file = paste("symbol_", name, "-R.json", sep = ''), sep = "")