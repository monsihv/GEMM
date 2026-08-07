// Arenas Memory Allocator
// Inspired by Ryan Fleury's article "Untangling Lifetimes"
// Relevant youtube videos also

#pragma once

#include "base.h"

#define KILOBYTE(n) ((sizeof(char) * 1000) * n)
#define MEGABYTE(n) ((KILOBYTE(1) * 1000) * n)
#define GIGABYTE(n) ((MEGABYTE(1) * 1000) * n)

#define MIN(a, b) (((a) < (b)) ? a : b)
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define ARENA_ALIGN (sizeof(void *))

typedef struct {
    u64 capacity;
    u64 pos;
} Arena;

Arena *arenaAlloc(u64 capacity);
void arenaFree(Arena *a);

void *pushArena(Arena *a, u64 size);

#define pushArray(arena, type, count) (type *)pushArena((arena), sizeof(type) * (count))
#define pushStruct(arena, type) pushArray((arena), type, 1)

void arenaPop(Arena *a, u64 size);
void arenaPopTo(Arena *a, u64 pos);
void clearArena(Arena *a);
