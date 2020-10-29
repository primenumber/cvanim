DEBUG_OPT=-pg -g -Og
RELEASE_OPT=-O2

cvanim: main.cpp
	g++ -o $@ -std=c++17 $(RELEASE_OPT) -march=native -mtune=native $^ -I /usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp

cvanim_debug: main.cpp
	g++ -o $@ -std=c++17 $(DEBUG_OPT) -march=native -mtune=native $^ -I /usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp

.PHONY: clean
clean:
	-rm cvanim cvanim_debug
