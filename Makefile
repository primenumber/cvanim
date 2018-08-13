cvanim: main.cpp
	g++ -o $@ -std=c++17 -g -Og -march=native -mtune=native $^ -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp
