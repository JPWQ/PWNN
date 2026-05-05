CC = gcc
CFLAGS = -Wall -Wextra -g -lm
BIN = pwnn
SRCS = $(wildcard src/*.c)

build:
	$(CC) $(CFLAGS) -o $(BIN) $(SRCS)

clean:
	rm -f $(BIN)
