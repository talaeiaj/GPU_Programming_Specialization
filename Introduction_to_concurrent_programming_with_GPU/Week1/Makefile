IDIR=./
CXX=g++
CXXFLAGS=-I$(IDIR)

build: *.cpp
	$(CXX) -o assignment.exe $(CXXFLAGS) *.cpp

.PHONY: clean

clean:
	rm -f assignment.exe

run:
	./assignment.exe $(ARGS)