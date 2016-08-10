UNAME_S := $(shell uname -s)

MXNET_ROOT=../mxnet

config = $(MXNET_ROOT)/config.mk
include $(config)
include $(MXNET_ROOT)/mshadow/make/mshadow.mk
include $(MXNET_ROOT)/dmlc-core/make/dmlc.mk

WARNFLAGS= -Wall
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS) -std=c++11 -O3

CFLAGS += -I$(MXNET_ROOT)/mshadow/ -I$(MXNET_ROOT)/dmlc-core/include -I$(MXNET_ROOT)/include -fPIC -Iinclude $(MSHADOW_CFLAGS)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
else
	LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DMSHADOW_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

ifeq ($(USE_THREADED_ENGINE), 1)
	CFLAGS += -DMXNET_USE_THREADED_ENGINE
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

LDFLAGS += -L$(MXNET_ROOT)/lib -lmxnet

.PHONY: depend clean all

all: func_info_test

func_info_test: clean_test
	$(CXX) -c -fPIC $(CFLAGS) func_info.cxx -o func_info.o
	$(CXX) -o func_info_test func_info.o $(LDFLAGS) 

clean_test:
	rm -rf *.o *_test *_test.json
