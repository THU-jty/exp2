
CC = mpiicc 
OPT =
CFLAGS = -Wall -std=c99 $(OPT) -qopenmp
LDFLAGS = -Wall -qopenmp
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-optimized benchmark-3D
objects = check.o benchmark.o stencil-naive.o stencil-optimized.o 3D.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

test : test.o
	$(CC) -o $@ $^ $(LDLIBS)
	
benchmark-optimized : check.o benchmark.o stencil-optimized.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-3D : check.o benchmark.o 3D.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)