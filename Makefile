CC = clang
SRC = src/main.c
BIN = build/main

CFLAGS_RELEASE = -O2 -mcpu=native
CFLAGS_DEBUG   = -O0 -g -fsanitize=address -fno-omit-frame-pointer

.PHONY: run debug clean

run: $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $(BIN) $(SRC)
	./$(BIN)

debug: $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -o $(BIN)_debug $(SRC)
	./$(BIN)_debug

clean:
	rm -rf build
