CURRENT_DIR = $(shell pwd)

JAVA_INCLUDE=/usr/lib/jvm/java-1.8.0-openjdk-amd64/include/linux

JNI_INCLUDE=/usr/lib/jvm/java-1.8.0-openjdk-amd64/include

SUFFIX=so

MXNET_SRCS=src/executor.cxx src/kvstore.cxx src/operator.cxx src/symbol.cxx src/io.cxx src/ndarray.cxx src/optimizer.cxx

MXNET_OBJS=$(MXNET_SRCS:.cxx=.o)

SRCS=mlp.cxx imagenet.cxx network_def.cxx image_classify.cxx

OBJS=$(SRCS:.cxx=.o)

TARGET=libNative.$(SUFFIX)

CXX=g++

INCLUDE=-I$(JAVA_INCLUDE) -I$(JNI_INCLUDE) -Iinclude

LDFLAGS=-Wl,-rpath,$(CURRENT_DIR) -L./lib -lmxnet

CXXFLAGS=-std=c++11 -O3

all: swig $(MXNET_OBJS) $(OBJS) $(TARGET)

$(MXNET_OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

$(OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

swig:
	swig -c++ -java -package water.gpu mlp.i
	swig -c++ -java -package water.gpu imagenet.i
	swig -c++ -java -package water.gpu image_classify.i

mlp_wrap.o:
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) mlp_wrap.cxx -o mlp_wrap.o

imagenet_wrap.o:
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) imagenet_wrap.cxx -o imagenet_wrap.o

image_classify_wrap.o:
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) image_classify_wrap.cxx -o image_classify_wrap.o

$(TARGET): mlp_wrap.o imagenet_wrap.o image_classify_wrap.o
	$(CXX) -shared $(MXNET_OBJS) $(OBJS) mlp_wrap.o imagenet_wrap.o image_classify_wrap.o -o $(TARGET) $(LDFLAGS)

mlp_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) mlp_test.cxx -o mlp_test.o
	$(CXX) -o mlp_test mlp_test.o $(MXNET_OBJS) $(OBJS) -L./lib -lmxnet

lstm_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) lstm_test.cxx -o lstm_test.o
	$(CXX) -o lstm_test lstm_test.o $(MXNET_OBJS) -L./lib -lmxnet

lenet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) lenet_test.cxx -o lenet_test.o
	$(CXX) -o lenet_test lenet_test.o $(MXNET_OBJS) -L./lib -lmxnet

inception_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) inception_test.cxx -o inception_test.o
	$(CXX) -o inception_test inception_test.o $(MXNET_OBJS) -L./lib -lmxnet

vgg_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) vgg_test.cxx -o vgg_test.o
	$(CXX) -o vgg_test vgg_test.o $(MXNET_OBJS) -L./lib -lmxnet

googlenet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) googlenet_test.cxx -o googlenet_test.o
	$(CXX) -o googlenet_test googlenet_test.o $(MXNET_OBJS) -L./lib -lmxnet

resnet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) resnet_test.cxx -o resnet_test.o
	$(CXX) -o resnet_test resnet_test.o $(MXNET_OBJS) -L./lib -lmxnet

alexnet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) alexnet_test.cxx -o alexnet_test.o
	$(CXX) -o alexnet_test alexnet_test.o $(MXNET_OBJS) -L./lib -lmxnet

clean:
	rm -rf $(MXNET_OBJS) $(OBJS) $(TARGET) *.java *_wrap.cxx *_wrap.o mlp_test.o mlp_test lstm_test alexnet_test alexnet_test.o

clean_test:
	rm -rf mlp_test.o mlp_test lstm_test lstm_test.o lenet_test lenet_test.o inception_test inception_test.o vgg_test vgg_test.o googlenet_test googlenet_test.o resnet_test resnet_test.o
