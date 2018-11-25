
CC = mpiicc 
OPT =
CFLAGS = -Wall -std=c99 $(OPT) -qopenmp
LDFLAGS = -Wall -qopenmp
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-mpiz benchmark-3D benchmark-cir benchmark-timeskew
objects = check.o benchmark.o stencil-naive.o mpiz.o 3D.o circular.o timeskew.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
	
benchmark-cir : check.o benchmark.o circular.o
	$(CC) -o $@ $^ $(LDLIBS)	

test : test.o
	$(CC) -o $@ $^ $(LDLIBS)
	
benchmark-mpiz : check.o benchmark.o mpiz.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-3D : check.o benchmark.o 3D.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-timeskew : check.o benchmark.o timeskew.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)