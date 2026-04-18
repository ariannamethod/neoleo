CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm

.PHONY: all clean test

all: leo

leo: leo.c
	$(CC) $(CFLAGS) leo.c $(LDFLAGS) -o $@

test: tests/test_leo
	./tests/test_leo

tests/test_leo: tests/test_leo.c leo.c
	$(CC) $(CFLAGS) tests/test_leo.c $(LDFLAGS) -I. -o $@

clean:
	rm -f leo tests/test_leo
