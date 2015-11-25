
all:
	g++ -O3 -march=native -o demo src/demo.cpp -Iinclude -Wall -lpthread -fopenmp

