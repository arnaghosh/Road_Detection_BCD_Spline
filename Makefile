all: bcd

bcd: BCD.cpp
	g++ -g $^ -std=c++0x -o $@ -Wall `pkg-config opencv --cflags --libs`