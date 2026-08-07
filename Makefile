make run:
	clang -O2 -mcpu=native -o build/main src/main.c
	./build/main
