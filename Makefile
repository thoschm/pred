
all: prog1 prog2

prog1:
	g++-5 -O3 -march=native -o learn src/learn.cpp -Iinclude -Wall -lpthread -fopenmp -framework OpenCL
    
prog2:
	g++-5 -O3 -march=native -o predict src/predict.cpp -Iinclude -Wall -lpthread -fopenmp -framework OpenCL
