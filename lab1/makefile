MPICXX = mpicc
CXXFLAGS = -Wall -O2

all: 1-s 2-s 3-s 4-s

1-s: 1-s.c
	$(MPICXX) $(CXXFLAGS) -o 1-s 1-s.c

2-s: 2-s.c
	$(MPICXX) $(CXXFLAGS) -o 2-s 2-s.c

3-s: 3-s.c
	$(MPICXX) $(CXXFLAGS) -o 3-s 3-s.c

4-s: 4-s.c
	$(MPICXX) $(CXXFLAGS) -o 4-s 4-s.c

clean:
	rm -f 1-s 2-s 3-s 4-s

