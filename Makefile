
all: prog1 prog2

prog1:
	g++ -O3 -march=native -o learn src/learn.cpp -Iinclude -Wall -lpthread -fopenmp
    
prog2:
#	g++ -O3 -march=native -o predict src/predict.cpp -Iinclude -Wall -lpthread -fopenmp

