JAVA_INCLUDE=/System/Library/Frameworks/JavaVM.framework/Headers/

SUFFIX=dylib

MXNET_SRCS=src/executor.cxx src/kvstore.cxx src/operator.cxx src/symbol.cxx src/io.cxx src/ndarray.cxx src/optimizer.cxx

MXNET_OBJS=$(MXNET_SRCS:.cxx=.o)

SRCS=mlp.cxx

OBJS=$(SRCS:.cxx=.o)

TARGET=libmlp.so

CXX=g++

INCLUDE=-I$(JAVA_INCLUDE) -Iinclude

LDFLAGS=-L./lib -lmxnet

CXXFLAGS=-std=c++11 -O3

all: $(MXNET_OBJS) $(OBJS)

$(MXNET_OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

$(OBJS): %.o : %.cxx
	$(CXX) -c -fPIC $(CXXFLAGS) $(INCLUDE) $< -o $@

clean:
	rm -rf $(MXNET_OBJS) $(OBJS)