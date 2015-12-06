
all: prog1 prog2 prog3

prog1:
	g++-5 -O3 -march=native -o learn src/learn.cpp -Iinclude -Wall -lpthread -fopenmp
    
prog2:
	g++-5 -O3 -march=native -o predict src/predict.cpp -Iinclude -Wall -lpthread -fopenmp

prog3:
	g++-5 -O3 -march=native -o src/psocl src/psocl.cpp -Iinclude -Wall -framework OpenCL
