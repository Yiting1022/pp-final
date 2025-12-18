CXX ?= g++
CXXFLAGS := -O3 -std=c++17 -fopenmp

TARGETS := mc_pricer

all: $(TARGETS)

mc_pricer: mc_pricer.cpp
	$(CXX) $(CXXFLAGS) mc_pricer.cpp -o mc_pricer

clean:
	rm -f $(TARGETS)
