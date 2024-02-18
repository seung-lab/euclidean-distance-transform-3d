all: debug

debug:
	g++ -pg -std=c++17 -g -O3 edt.hpp -ffast-math -pthread -o edt 

shared:
	g++ -fPIC -shared -std=c++17 -O3 edt.hpp -ffast-math -pthread -o edt.so

test: FORCE
	g++ -pg -std=c++17 -g -O3 test.cpp -ffast-math -pthread -o test

FORCE: