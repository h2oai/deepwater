CURRENT_DIR = $(shell pwd)

JAVA_INCLUDE=/usr/lib/jvm/java-1.8.0-openjdk-amd64/include/linux

JNI_INCLUDE=/usr/lib/jvm/java-1.8.0-openjdk-amd64/include

SUFFIX=so

MXNET_SRCS=src/executor.cxx src/kvstore.cxx src/operator.cxx src/symbol.cxx src/io.cxx src/ndarray.cxx src/optimizer.cxx

MXNET_OBJS=$(MXNET_SRCS:.cxx=.o)

SRCS=mlp.cxx image_pred.cxx network_def.cxx image_train.cxx

OBJS=$(SRCS:.cxx=.o)

TARGET=libNative.$(SUFFIX)

CXX=g++

INCLUDE=-I$(JAVA_INCLUDE) -I$(JNI_INCLUDE) -Iinclude

LDFLAGS=-Wl,-rpath,/tmp -L./lib -lmxnet

CXXFLAGS=-std=c++11 -O3

all: swig $(MXNET_OBJS) $(OBJS) $(TARGET)

$(MXNET_OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

$(OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

swig:
	swig -c++ -java -package water.gpu deepwater.i

deepwater_wrap.o:
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) deepwater_wrap.cxx -o deepwater_wrap.o

$(TARGET): deepwater_wrap.o
	rm -rf $(TARGET)
	$(CXX) -shared $(MXNET_OBJS) $(OBJS) deepwater_wrap.o -o $(TARGET) $(LDFLAGS)

test: mlp_test lstm_test lenet_test inception_test vgg_test googlenet_test resnet_test alexnet_test

mlp_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/mlp_test.cxx -o mlp_test.o
	$(CXX) -o mlp_test mlp_test.o $(MXNET_OBJS) $(OBJS) -L./lib -lmxnet

lstm_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/lstm_test.cxx -o lstm_test.o
	$(CXX) -o lstm_test lstm_test.o $(MXNET_OBJS) -L./lib -lmxnet

lenet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/lenet_test.cxx -o lenet_test.o
	$(CXX) -o lenet_test lenet_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

inception_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/inception_test.cxx -o inception_test.o
	$(CXX) -o inception_test inception_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

vgg_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/vgg_test.cxx -o vgg_test.o
	$(CXX) -o vgg_test vgg_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

googlenet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/googlenet_test.cxx -o googlenet_test.o
	$(CXX) -o googlenet_test googlenet_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

resnet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/resnet_test.cxx -o resnet_test.o
	$(CXX) -o resnet_test resnet_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

alexnet_test: $(TARGET) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) ./test/alexnet_test.cxx -o alexnet_test.o
	$(CXX) -o alexnet_test alexnet_test.o network_def.o $(MXNET_OBJS) -L./lib -lmxnet

lint:
	python lint.py deepwater cpp .

clean: clean_test
	rm -rf $(MXNET_OBJS) $(OBJS) $(TARGET) *_wrap.cxx *_wrap.o

clean_test:
	rm -rf *_test.o *_test water*
