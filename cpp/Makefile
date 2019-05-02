all: debug

debug:
	g++ -pg -std=c++11 -g -O3 edt.hpp -ffast-math -pthread -o edt 

shared:
	g++ -fPIC -shared -std=c++11 -O3 edt.hpp -ffast-math -pthread -o edt.so

test: FORCE
	g++ -pg -std=c++11 -g -O3 test.cpp -ffast-math -pthread -o test

FORCE: