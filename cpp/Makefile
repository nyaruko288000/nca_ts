CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -march=native
LDFLAGS ?= -lz

all: nca_gen

nca_gen: nca_gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f nca_gen
