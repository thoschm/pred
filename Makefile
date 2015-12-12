
all: prog1 prog2

prog1:
	g++ -O3 -march=native -o learn src/learn.cpp -Iinclude -Wall -lpthread -fopenmp -lOpenCL -I/opt/AMDAPPSDK-3.0/include/ -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk
	
prog2:
	g++ -O3 -march=native -o predict src/predict.cpp -Iinclude -Wall -lpthread -fopenmp -lOpenCL -I/opt/AMDAPPSDK-3.0/include/ -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk
