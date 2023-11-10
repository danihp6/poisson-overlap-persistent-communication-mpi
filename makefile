CC = mpicc
CFLAGS = -Wall
LDFLAGS = -lm

TARGETS = work poisson
SRCS = work.c poisson.c

all: $(TARGETS)

work: work.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

poisson: poisson.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
