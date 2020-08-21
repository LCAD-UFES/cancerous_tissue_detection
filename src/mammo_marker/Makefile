CC=g++
CFLAGS=-I. -std=c++0x -W -Wall `pkg-config --cflags opencv` -Wno-unused-variable -g
LFLAGS=`pkg-config --libs opencv` 
#DEPS = hellomake.h

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mammo_marker: mammo_marker.o 
	$(CC) -o mammo_marker mammo_marker.o $(LFLAGS)

clean:
	rm *.o mammo_marker
