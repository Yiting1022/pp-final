NVCC ?= nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_80

# 可以依需要在這裡加上共用 include / lib 設定，例如：
# NVCCFLAGS += -I../include -L../lib

TARGETS := \
	gpu_pricer

all: $(TARGETS)

gpu_pricer: gpu_pricer.cu mc_pricer.cu
	$(NVCC) $(NVCCFLAGS) gpu_pricer.cu -o $@

clean:
	rm -f $(TARGETS)
