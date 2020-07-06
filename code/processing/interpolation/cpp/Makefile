all: clean lib

lib: extern.cpp
	g++ -std=c++17 -O3 -fPIC -shared -I./ ./*.cpp -o libinterpolation.so

clean:
	rm -f ./*.so