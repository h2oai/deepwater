library(mxnet)

for name in c("alexnet", "googlenet", "inception-bn", "inception-v3", "lenet", "mlp", "resnet", "vgg") {
    source(paste("symbol_", name, ".R", sep = ''))
    network <- get_symbol(10)
    cat(network$as.json(), file = paste("symbol_", name, "-R.json", sep = ''), sep = "")
}

name = "unet"

source(paste("symbol_", name, ".R", sep = ''))

network <- get_symbol()

cat(network$as.json(), file = paste("symbol_", name, "-R.json", sep = ''), sep = "")
