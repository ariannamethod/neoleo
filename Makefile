CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm

.PHONY: all clean test leogo clean-leogo

all: leo

leo: leo.c
	$(CC) $(CFLAGS) leo.c $(LDFLAGS) -o $@

test: tests/test_leo
	./tests/test_leo

tests/test_leo: tests/test_leo.c leo.c
	$(CC) $(CFLAGS) tests/test_leo.c $(LDFLAGS) -I. -o $@

# Go orchestra (async rings of thought around the C core).
# Optional target — ./leo works standalone without ever building this.
leogo: leogo/leogo

leogo/leogo: leogo/main.go leogo/leo.go leogo/leo_bridge.c leo.c
	cd leogo && go build -o leogo .

clean-leogo:
	rm -f leogo/leogo

clean: clean-leogo
	rm -f leo tests/test_leo
