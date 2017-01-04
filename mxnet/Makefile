all: pkg

UNAME_S := $(shell uname -s)

ifndef JAVA_HOME
        $(error JAVA_HOME is undefined)
endif

ifeq ($(UNAME_S), Darwin)
	#JAVA_INCLUDE := -I/System/Library/Frameworks/JavaVM.framework/Headers
	JAVA_INCLUDE := -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/darwin/
else
	#JAVA_INCLUDE := -I/usr/lib/jvm/java-1.8.0-openjdk-amd64/include/linux -I/usr/lib/jvm/java-1.8.0-openjdk-amd64/include
	JAVA_INCLUDE := -I/usr/lib/jvm/java-8-oracle/include/linux/ -I/usr/lib/jvm/java-8-oracle/include/
endif

#MXNET_ROOT:=../thirdparty/javacpp-presets/mxnet
MXNET_ROOT:=../thirdparty/mxnet

config = $(MXNET_ROOT)/config.mk
include $(config)
include $(MXNET_ROOT)/mshadow/make/mshadow.mk
include $(MXNET_ROOT)/dmlc-core/make/dmlc.mk

OBJDIR = build

VPATH = src
MXNET_SRCS := io.cxx ndarray.cxx kvstore.cxx executor.cxx symbol.cxx optimizer.cxx network_def.cxx operator.cxx
MXNET_OBJS = $(patsubst %.cxx,$(OBJDIR)/%.o,$(MXNET_SRCS)) 

$(MXNET_OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $@

SRCS := image_train.cxx image_pred.cxx

OBJS = $(patsubst %.cxx,$(OBJDIR)/%.o,$(SRCS))

TARGET := $(OBJDIR)/libNative.so

CXX := g++

SWIG := swig

INCLUDE = $(JAVA_INCLUDE) -Iinclude -I. -I./src/

#MXLIB := -L$(MXNET_ROOT)/cppbuild/linux-x86_64/lib -lmxnet
MXLIB := -L$(MXNET_ROOT)/lib -lmxnet

LDFLAGS = -Wl,-rpath,/tmp $(MXLIB)

CXXFLAGS = -std=c++11 -O3 -Wall $(INCLUDE) -Wno-unused-function
#CXXFLAGS = -std=c++11 -O0 -g -Wall $(INCLUDE) -Wno-unused-function

ifeq ($(USE_CUDA), 1)
	CXXFLAGS += -DMSHADOW_USE_CUDA=1
else
	CXXFLAGS += -DMSHADOW_USE_CUDA=0
endif

$(OBJDIR)/%.o: %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $< -MM -MF $@
	$(CXX) -c -fPIC $(CXXFLAGS) $< -o $@

DEPS=$(OBJS:.o=.d) $(MXNET_OBJS:.o=.d)

.PHONY: pkg clean


-include $(DEPS)

$(TARGET): $(MXNET_OBJS) $(OBJS)
	$(SWIG) -outdir $(OBJDIR) -c++ -java -package deepwater.backends.mxnet -o $(OBJDIR)/deepwater_wrap.cxx deepwater.i 
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $(OBJDIR)/deepwater_wrap.cxx -o $(OBJDIR)/deepwater_wrap.o
	$(CXX) -shared $(MXNET_OBJS) $(OBJS) $(SWIG_OBJS) $(OBJDIR)/deepwater_wrap.o -o $(TARGET) $(LDFLAGS)

lint:
	python ./scripts/lint.py deepwater cpp *.cxx *.hpp ./include ./src

pkg: $(TARGET)
	rm -rf $(OBJDIR)/deepwater
	mkdir -p $(OBJDIR)/deepwater/backends/mxnet
	javac -sourcepath . -d $(OBJDIR) *.java $(OBJDIR)/*.java
ifeq ($(UNAME_S), Darwin)
	install_name_tool -change lib/libmxnet.so @loader_path/libmxnet.so $(TARGET)
endif
	cp $(TARGET) $(OBJDIR)/deepwater/backends/mxnet
	cp $(MXNET_ROOT)/lib/libmxnet.so $(OBJDIR)/deepwater/backends/mxnet

mlp_test: $(MXNET_OBJS) clean_test
	bash ./test/download_mnist.sh
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/mlp_test.cxx -o mlp_test.o
	$(CXX) -o mlp_test mlp_test.o $(MXNET_OBJS) $(MXLIB)

lenet_test: $(MXNET_OBJS) clean_test
	bash ./test/download_mnist.sh
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/lenet_test.cxx -o lenet_test.o
	$(CXX) -o lenet_test lenet_test.o $(MXNET_OBJS) $(MXLIB)

inception_test: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/inception_test.cxx -o inception_test.o
	$(CXX) -o inception_test inception_test.o $(MXNET_OBJS) $(MXLIB)

inception_test2: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/inception_test2.cxx -o inception_test2.o
	$(CXX) -o inception_test2 inception_test2.o $(MXNET_OBJS) $(MXLIB)

net_def_test: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/net_def_test.cxx -o net_def_test.o
	$(CXX) -o net_def_test net_def_test.o $(MXNET_OBJS) $(MXLIB)
	./net_def_test

lstm_test: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/lstm_test.cxx -o lstm_test.o
	$(CXX) -o lstm_test lstm_test.o $(MXNET_OBJS) $(MXLIB)

swig_test: clean_test
	swig -c++ -java -package test.swig ./test/swigtest.i
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/swigtest.cxx -o ./test/swigtest.o
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/swigtest_wrap.cxx -o ./test/swigtest_wrap.o
	mkdir -p ./test/swig
ifeq ($(UNAME_S), Darwin)
	$(CXX) -shared ./test/swigtest.o ./test/swigtest_wrap.o -o ./test/swig/libswigtest.dylib
else
	$(CXX) -shared ./test/swigtest.o ./test/swigtest_wrap.o -o ./test/swig/libswigtest.so
endif
	cp ./test/testJNI.java ./test/swig
	cp ./test/test.java ./test/swig
	cp ./test/swigtest.java ./test/swig
	javac ./test/swig/*.java
	java -Djava.library.path="./test/swig" test/swig/swigtest

load_save_test: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) image_train.cxx -o image_train.o
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/load_save_test.cxx -o load_save_test.o
	$(CXX) -o load_save_test load_save_test.o image_train.o $(MXNET_OBJS) $(MXLIB)

seed_test: $(MXNET_OBJS) clean_test
	$(CXX) -c -fPIC $(CXXFLAGS) image_train.cxx -o image_train.o
	$(CXX) -c -fPIC $(CXXFLAGS) ./test/seed_test.cxx -o seed_test.o
	$(CXX) -o seed_test seed_test.o image_train.o $(MXNET_OBJS) $(MXLIB)

java_test: pkg
	javac -cp build src/test/java/deepwater/backends/mxnet/test/InceptionCLI.java
	java -cp build:src/test/java deepwater.backends.mxnet.test.InceptionCLI src/main/resources/deepwater/backends/mxnet/models/Inception $(PWD)/test/test2.jpg 

clean: clean_test
	rm -rf $(OBJDIR)

clean_test:
	rm -rf *_test.o *_test water* ./test/*_wrap.cxx ./test/*_wrap.o ./test/libswigtest.so ./test/swig/* test/test.java test/testJNI.java

