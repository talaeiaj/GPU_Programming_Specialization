IDIR=./
STD :=c++14
CXX=g++
CXXFLAGS=-I$(IDIR) -std=$(STD) -pthread

.PHONY: clean build run

assignment.o: assignment.cpp assignment.h
	$(CXX) assignment.cpp -c -o assignment.o

build: assignment.o
	$(CXX) assignment.o -o assignment.exe $(CXXFLAGS)

clean:
	rm -f *.exe output-*.txt

run:
	./assignment.exe $(ARGS)

all: clean build run

