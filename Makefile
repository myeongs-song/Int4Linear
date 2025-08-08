TARGET := run
CXX := g++
NVCC := nvcc
LINKER := $(NVCC)

SRCS_CPP := $(shell find . -name "*.cpp")
SRCS_CU  := $(shell find . -name "*.cu")

OBJS_CPP := $(SRCS_CPP:.cpp=.o)
OBJS_CU := $(SRCS_CU:.cu=.o)
OBJS := $(OBJS_CPP) $(OBJS_CU)

CXXFLAGS := -O3 -Wall -std=c++17
NVCCFLAGS := -O3 -std=c++17 -gencode arch=compute_86,code=sm_86

INCLUDES := $(shell find . -type d -printf "-I%p ")
CXXFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)

LDFLAGS :=
LDLIBS :=

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean