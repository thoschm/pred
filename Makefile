
all: prog1 prog2 prog3

prog1:
	g++ -O3 -march=native -o learn src/learn.cpp -Iinclude -I/opt/AMDAPPSDK-3.0/include -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/ -Wall -lpthread -fopenmp -lOpenCL
    
prog2:
	g++ -O3 -march=native -o predict src/predict.cpp -Iinclude -I/opt/AMDAPPSDK-3.0/include -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/ -Wall -lpthread -fopenmp -lOpenCL

prog3:
	g++ -O3 -march=native -o src/psocl src/psocl.cpp -Iinclude -I/opt/AMDAPPSDK-3.0/include -L/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/ -Wall -lOpenCL
