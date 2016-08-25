library(mxnet)

#name = "alexnet"
#name = "googlenet"
#name = "inception-bn"
name = "inception-v3"
#name = "lenet"
#name = "mlp"
#name = "resnet"
#name = "vgg"

source(paste("symbol_", name, ".R", sep = ''))

network <- get_symbol(10)

cat(network$as.json(), file = paste("symbol_", name, "-R.json", sep = ''), sep = "")

name = "unet"

source(paste("symbol_", name, ".R", sep = ''))

network <- get_symbol()

cat(network$as.json(), file = paste("symbol_", name, "-R.json", sep = ''), sep = "")
